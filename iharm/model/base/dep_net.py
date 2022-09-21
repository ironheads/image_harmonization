'''
this file is used to implete DEPNet Model
'''

import torch
from functools import partial
from torch import nn as nn

from iharm.model.modeling.basic_blocks import ConvBlock, GaussianSmoothing
from iharm.model.modeling.unet import UNetEncoder, UNetDecoder
from iharm.model.ops import ChannelAttention,OutputLayer


class DEPNet(nn.Module):
    def __init__(
    self, 
    encoder_decoder_depth,
    norm_layer=nn.BatchNorm2d,
    high_resolution = 1024, 
    low_resolution = 256,
    batchnorm_from=2,
    attend_from=3, 
    attention_mid_k=2.0,
    color_mid_channels = 512,
    ch=64, 
    L=256,
    max_channels=512,
    backbone_from=-1, 
    backbone_channels=None, 
    backbone_mode='',
    ) -> None:
        super(DEPNet, self).__init__()
        # low_resolution = high_resolution // (2**(encoder_decoder_depth-1))
        assert  (high_resolution % low_resolution == 0),"high_resolution must be divisible by low_resolution"
        self.L = L
        self.hr = high_resolution
        self.lr = low_resolution
        self.lowResolutionPicture = partial(nn.functional.interpolate, size=(low_resolution,low_resolution), mode='bilinear', align_corners=True)
        self.lowResolutionMask = partial(nn.functional.interpolate, size=(low_resolution,low_resolution), mode='nearest')
        self.encoder = UNetEncoder(
            encoder_decoder_depth, ch,
            norm_layer, batchnorm_from, max_channels,
            backbone_from, backbone_channels, backbone_mode
        )
        self.decoder = UNetDecoder(
            encoder_decoder_depth, self.encoder.block_channels,
            norm_layer,
            attention_layer=partial(SpatialSeparatedAttention, mid_k=attention_mid_k),
            attend_from=attend_from,
            output_image=False,
            image_fusion=False
        )
        self.color_mid_channels = color_mid_channels
        self.p2p_blending = OutputLayer(self.decoder.output_channels,blending=True)
        self.average_pool = MaskedAveragePooling()
       

        self.get_L_length_encoder_feature = nn.Conv2d(min((2**(encoder_decoder_depth-1)) * ch,max_channels),self.L,1)
        self.downsampleMask = nn.AvgPool2d(2**(encoder_decoder_depth-1))
        
        self.color_map = nn.Sequential(
            nn.Linear(2*self.L+3,3),
            nn.LeakyReLU(0.2),
            # nn.Linear(self.color_mid_channels,3),
            # nn.LeakyReLU(0.2)
        )

        self.upsampler = nn.Upsample(scale_factor=high_resolution//low_resolution, mode='bilinear', align_corners=True)
        self.refine_module = RefinementModule(self.decoder.output_channels+7,2*self.decoder.output_channels)

    def forward(self,high_resolution_image,high_resolution_mask,backbone_features=None):
        # print(high_resolution_image.min(),high_resolution_image.max())
        low_resolution_image = self.lowResolutionPicture(high_resolution_image)
        low_resolution_mask = self.lowResolutionMask(high_resolution_mask)
        p2p_input = torch.cat([low_resolution_image,low_resolution_mask],1)
        encoder_output = self.encoder(p2p_input,backbone_features)
        decoder_output = self.decoder(encoder_output,low_resolution_image,low_resolution_mask)
        p2p_output = self.p2p_blending(decoder_output,low_resolution_image)
        foreground_mask = self.downsampleMask(low_resolution_mask)
        background_mask = 1 - foreground_mask
        
        encoder_feature = encoder_output[0]
        encoder_feature = self.get_L_length_encoder_feature(encoder_feature)
        background_feature = self.average_pool(encoder_feature,background_mask)
        foreground_feature = self.average_pool(encoder_feature,foreground_mask)
        background_feature = background_feature.reshape(-1,self.L)
        foreground_feature = foreground_feature.reshape(-1,self.L)
        feature = torch.cat([background_feature,foreground_feature],1)
        features = feature.unsqueeze(1).unsqueeze(1)
        features = features.repeat(1,self.lr,self.lr,1)
            
        c2c_input = low_resolution_image.permute(0,2,3,1).contiguous()
        c2c_input = torch.cat([c2c_input,features],3)
        c2c_output = self.color_map(c2c_input).permute(0,3,1,2).contiguous()
   
        # high_resolution_c2c_output = high_resolution_c2c_output.clamp(0,1)
        refine_input = torch.cat([self.upsampler(p2p_output),self.upsampler(c2c_output),high_resolution_mask,self.upsampler(decoder_output)],1)
        refine_output = self.refine_module(refine_input,high_resolution_image)
        return {
            'c2c_outputs':c2c_output,
            'p2p_outputs':p2p_output,
            'images': refine_output,
        }
        



class SpatialSeparatedAttention(nn.Module):
    def __init__(self, in_channels, norm_layer, activation, mid_k=2.0):
        super(SpatialSeparatedAttention, self).__init__()
        self.background_gate = ChannelAttention(in_channels)
        self.foreground_gate = ChannelAttention(in_channels)
        self.mix_gate = ChannelAttention(in_channels)

        mid_channels = int(mid_k * in_channels)
        self.learning_block = nn.Sequential(
            ConvBlock(
                in_channels, mid_channels,
                kernel_size=3, stride=1, padding=1,
                norm_layer=norm_layer, activation=activation,
                bias=False,
            ),
            ConvBlock(
                mid_channels, in_channels,
                kernel_size=3, stride=1, padding=1,
                norm_layer=norm_layer, activation=activation,
                bias=False,
            ),
        )
        self.mask_blurring = GaussianSmoothing(1, 7, 1, padding=3)

    def forward(self, x, mask):
        mask = self.mask_blurring(nn.functional.interpolate(
            mask, size=x.size()[-2:],
            mode='bilinear', align_corners=True
        ))
        background = self.background_gate(x)
        foreground = self.learning_block(self.foreground_gate(x))
        mix = self.mix_gate(x)
        output = mask * (foreground + mix) + (1 - mask) * background
        return output


class RefinementModule(nn.Module):
    def __init__(self,in_channels, mid_channels) -> None:
        super(RefinementModule,self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels,mid_channels,3,1,padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ELU()
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(mid_channels,mid_channels,3,1,padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ELU()
        )

        self.blend = OutputLayer(mid_channels,blending=True)

    def forward(self,input_image,composite_image):
        output = self.layer1(input_image)
        output = self.layer2(output)
        output = self.blend(output,composite_image)
        return output


class MaskedAveragePooling(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self,x,mask):
        x = x * mask
        x = torch.sum(x,dim=(-2,-1))
        mask_sum = torch.sum(mask,dim=(-2,-1))
        # avoid the mask == 0
        x = x / (torch.sum(mask,dim=(-2,-1))+1e-8)
        return x