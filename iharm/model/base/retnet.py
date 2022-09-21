'''
this file is used to implete RETNet Model
this model is based on retinex theory
'''

import torch
from functools import partial
import numpy as np
from torch import nn as nn

from iharm.model.modeling.basic_blocks import ConvBlock, GaussianSmoothing
from iharm.model.modeling.unet import UNetEncoder, UNetDecoder
from iharm.model.ops import ChannelAttention,OutputLayer


class RetNet(nn.Module):
    def __init__(
    self, 
    encoder_decoder_depth,
    norm_layer=nn.BatchNorm2d,
    high_resolution = 1024, 
    low_resolution = 256,
    batchnorm_from=2,
    attend_from=3, 
    attention_mid_k=0.5,
    ch=64, 
    L=256,
    max_channels=512,
    backbone_from=-1, 
    backbone_channels=None, 
    backbone_mode='',
    ) -> None:
        super(RetNet, self).__init__()
        assert  (high_resolution % low_resolution == 0),"high_resolution must be divisible by low_resolution"
        self.L = L
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
        self.to_retinex_light = nn.Conv2d(self.decoder.output_channels,3,kernel_size=1)
        self.to_attention_map = nn.Conv2d(self.decoder.output_channels,1,kernel_size=1)
 
        self.upsampler = nn.Upsample(scale_factor=high_resolution//low_resolution, mode='bilinear', align_corners=True)
        self.refine_module = RefinementModule(self.decoder.output_channels+7,2*self.decoder.output_channels)
        self.maskedLightAdaIN =MaskedLightAdaIN()

    def forward(self,high_resolution_image,high_resolution_mask,backbone_features=None):
        # print(high_resolution_image.min(),high_resolution_image.max())
        low_resolution_image = self.lowResolutionPicture(high_resolution_image)
        low_resolution_mask = self.lowResolutionMask(high_resolution_mask)
        p2p_input = torch.cat([low_resolution_image,low_resolution_mask],1)
        encoder_output = self.encoder(p2p_input,backbone_features)
        decoder_output = self.decoder(encoder_output,low_resolution_image,low_resolution_mask)
        retinex_light = self.to_retinex_light(decoder_output)
        attention_map = self.to_attention_map(decoder_output)
        transfered_light = self.maskedLightAdaIN(retinex_light,low_resolution_mask)
        low_resolution_output = low_resolution_image/retinex_light*transfered_light
        upsampled_retinex_light = self.upsampler(retinex_light)
        upsampled_transfered_light = self.upsampler(transfered_light)
        high_resolution_output = high_resolution_image/upsampled_retinex_light*upsampled_transfered_light
        return {
            'lr_images':low_resolution_output,
            'images': high_resolution_output,
            # 'weight_norm':torch.mean(LUT_weights**2),
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

#  the channel-based AdaIN
class MaskedLightAdaIN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, x, mask):
        # x: [B, C, H, W]
        # mask: [B, 1, H, W]
        # return: [B, C, H, W]
        batch, channel  = x.size(0),x.size(1)
        new_x = x.new(x.size())
        for i in range(batch):
            m = mask[i,0,:,:]
            for j in range(channel):
                img = x[i,j,:,:]
                mu_fg = torch.mean(img[m>=0.5])
                mu_bg = torch.mean(img[m<0.5])
                sigma_fg = torch.std(img[m>=0.5])
                sigma_bg = torch.std(img[m<0.5])
                new_img = img.new(img.size())
                new_img[m>=0.5] = img[m>=0.5]
                new_img[m<0.5] = (img[m<0.5] - mu_bg) / (sigma_bg+1e-8) * sigma_fg + mu_fg
                new_x[i,j,:,:] = new_img
        return new_x 

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