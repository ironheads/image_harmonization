'''
this file is used to implete CDTNet Model
'''

from turtle import forward
import torch
from functools import partial
import numpy as np
from torch import nn as nn

from iharm.model.modeling.basic_blocks import ConvBlock, GaussianSmoothing
from iharm.model.modeling.unet import UNetEncoder, UNetDecoder
from iharm.model.ops import ChannelAttention,OutputLayer

import trilinear

class CDTNet(nn.Module):
    def __init__(
    self, 
    encoder_decoder_depth,
    norm_layer=nn.BatchNorm2d,
    high_resolution = 1024, 
    low_resolution = 256,
    batchnorm_from=2,
    attend_from=3, 
    attention_mid_k=2.0,
    ch=64, 
    max_channels=512,
    backbone_from=-1, 
    backbone_channels=None, 
    backbone_mode='',
    LUT_nums = 4,
    LUT_channels = 33,
    ) -> None:
        super(CDTNet, self).__init__()
        # low_resolution = high_resolution // (2**(encoder_decoder_depth-1))
        assert  (high_resolution % low_resolution == 0),"high_resolution must be divisible by low_resolution"
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
        self.p2p_blending = OutputLayer(self.decoder.output_channels,blending=True)
        self.average_pool = MaskedAveragePooling()
        self.LUTs = nn.ModuleList()
        for i in range(LUT_nums):
            if i==0:
                self.LUTs.append(Generator3DLUT_identity(LUT_channels))
            else:
                self.LUTs.append(Generator3DLUT_zero(LUT_channels))

        # self.TV = TV_3D()
        self.LUT_nums = LUT_nums

        self.downsampleMask = nn.AvgPool2d(2**(encoder_decoder_depth-1))
        self.L = (2**(encoder_decoder_depth-1)) * ch
        self.LUT_FC = nn.Sequential(
            nn.Linear(2*self.L,LUT_nums),
            nn.LeakyReLU(0.2)
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
        # delete detach may case l_rgb turn to nan
        encoder_feature = encoder_output[0].detach()

        background_feature = self.average_pool(encoder_feature,background_mask)
        foreground_feature = self.average_pool(encoder_feature,foreground_mask)
        background_feature = background_feature.reshape(-1,self.L)
        foreground_feature = foreground_feature.reshape(-1,self.L)
        feature = torch.cat([background_feature,foreground_feature],1)
        LUT_weights = self.LUT_FC(feature)
        # batch = LUT_weights.shape[0]
        generate_c2c = []
        channel_first_high_resolution_image = high_resolution_image.permute(1,0,2,3).contiguous()
        # print(channel_first_high_resolution_image.min(),channel_first_high_resolution_image.max())
        for i in range(self.LUT_nums):
            generate_c2c.append(self.LUTs[i](channel_first_high_resolution_image))
        high_resolution_c2c_output = channel_first_high_resolution_image.new(channel_first_high_resolution_image.size())
        for b in range(high_resolution_c2c_output.size(1)):
            for i in range(self.LUT_nums):
                if i==0:
                    high_resolution_c2c_output[:,b,:,:] = generate_c2c[i][:,b,:,:]*LUT_weights[b,i]
                else:
                    high_resolution_c2c_output[:,b,:,:] += generate_c2c[i][:,b,:,:]*LUT_weights[b,i]
   
        high_resolution_c2c_output = high_resolution_c2c_output.clamp(0,1)
        high_resolution_c2c_output = high_resolution_c2c_output.permute(1,0,2,3).contiguous()
        refine_input = torch.cat([self.upsampler(p2p_output),high_resolution_c2c_output,high_resolution_mask,self.upsampler(decoder_output)],1)
        refine_output = self.refine_module(refine_input,high_resolution_image)
        return {
            'c2c_outputs':high_resolution_c2c_output,
            'p2p_outputs':p2p_output,
            'images': refine_output,
            # 'weight_norm':torch.mean(LUT_weights**2),
        }
        
    def init_weights(self, func):
        self.encoder.apply(func)
        self.decoder.apply(func)
        self.p2p_blending.apply(func)
        self.refine_module.apply(func)
        self.LUT_FC.apply(func)



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




class Generator3DLUT_identity(nn.Module):
    def __init__(self, dim=33):
        super(Generator3DLUT_identity, self).__init__()
        if dim == 33:
            file = open("preset_LUT/IdentityLUT33.txt", 'r')
        elif dim == 64:
            file = open("preset_LUT/IdentityLUT64.txt", 'r')
        lines = file.readlines()
        buffer = np.zeros((3,dim,dim,dim), dtype=np.float32)

        for i in range(0,dim):
            for j in range(0,dim):
                for k in range(0,dim):
                    n = i * dim*dim + j * dim + k
                    x = lines[n].split()
                    buffer[0,i,j,k] = float(x[0])
                    buffer[1,i,j,k] = float(x[1])
                    buffer[2,i,j,k] = float(x[2])
        self.LUT = nn.Parameter(torch.from_numpy(buffer),requires_grad=True)
        self.TrilinearInterpolation = TrilinearInterpolation()

    def forward(self, x):
        _, output = self.TrilinearInterpolation(self.LUT, x)
        #self.LUT, output = self.TrilinearInterpolation(self.LUT, x)
        return output

class Generator3DLUT_zero(nn.Module):
    def __init__(self, dim=33):
        super(Generator3DLUT_zero, self).__init__()
        self.LUT = nn.Parameter(torch.zeros(3,dim,dim,dim, dtype=torch.float),requires_grad=True)
        self.TrilinearInterpolation = TrilinearInterpolation()

    def forward(self, x):
        _, output = self.TrilinearInterpolation(self.LUT, x)

        return output

class TrilinearInterpolationFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, lut, x):
        x = x.contiguous()
        output = x.new(x.size())
        dim = lut.size()[-1]
        shift = dim ** 3
        binsize = 1.000001 / (dim-1)
        W = x.size(2)
        H = x.size(3)
        batch = x.size(1)
        # print(batch)
        assert 1 == trilinear.forward(lut, 
                                      x, 
                                      output,
                                      dim, 
                                      shift, 
                                      binsize, 
                                      W, 
                                      H, 
                                      batch)

        int_package = torch.IntTensor([dim, shift, W, H, batch])
        float_package = torch.FloatTensor([binsize])
        variables = [lut, x, int_package, float_package]
        
        ctx.save_for_backward(*variables)
        
        return lut, output
    
    @staticmethod
    def backward(ctx, lut_grad, x_grad):
        
        lut, x, int_package, float_package = ctx.saved_variables
        dim, shift, W, H, batch = int_package
        dim, shift, W, H, batch = int(dim), int(shift), int(W), int(H), int(batch)
        binsize = float(float_package[0])
            
        assert 1 == trilinear.backward(x, 
                                       x_grad, 
                                       lut_grad,
                                       dim, 
                                       shift, 
                                       binsize, 
                                       W, 
                                       H, 
                                       batch)
        return lut_grad, x_grad


class TrilinearInterpolation(torch.nn.Module):
    def __init__(self):
        super(TrilinearInterpolation, self).__init__()

    def forward(self, lut, x):
        return TrilinearInterpolationFunction.apply(lut, x)


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