import torch
import torch.nn as nn

from iharm.utils import misc
from functools import partial

class Loss(nn.Module):
    def __init__(self, pred_outputs, gt_outputs):
        super().__init__()
        self.pred_outputs = pred_outputs
        self.gt_outputs = gt_outputs


class MSE(Loss):
    def __init__(self, pred_name='images', gt_image_name='target_images'):
        super(MSE, self).__init__(pred_outputs=(pred_name,), gt_outputs=(gt_image_name,))

    def forward(self, pred, label):
        label = label.view(pred.size())
        loss = torch.mean((pred - label) ** 2, dim=misc.get_dims_with_exclusion(label.dim(), 0))
        return loss

class L1Loss(Loss):
    def __init__(self, pred_name='images', gt_image_name='target_images'):
        super(L1Loss, self).__init__(pred_outputs=(pred_name,), gt_outputs=(gt_image_name,))
    
    def forward(self, pred, label):
        label = label.view(pred.size())
        loss = torch.mean(torch.abs(pred - label), dim=misc.get_dims_with_exclusion(label.dim(), 0))
        return loss

class DownsampleLabelL1Loss(Loss):
    def __init__(self, pred_name='images', gt_image_name='target_images', low_resolution=256):
        super(DownsampleLabelL1Loss, self).__init__(pred_outputs=(pred_name,), gt_outputs=(gt_image_name,))
        self.downsample = self.downsamplePicture = partial(nn.functional.interpolate, size=(low_resolution,low_resolution), mode='bilinear', align_corners=True)
    
    def forward(self, pred, label):
        new_label = self.downsample(label)
        new_label = new_label.view(pred.size())
        loss = torch.mean(torch.abs(pred - new_label), dim=misc.get_dims_with_exclusion(new_label.dim(), 0))
        return loss

class MaskWeightedMSE(Loss):
    def __init__(self, min_area=1000.0, pred_name='images',
                 gt_image_name='target_images', gt_mask_name='masks'):
        super(MaskWeightedMSE, self).__init__(pred_outputs=(pred_name, ),
                                              gt_outputs=(gt_image_name, gt_mask_name))
        self.min_area = min_area

    def forward(self, pred, label, mask):
        label = label.view(pred.size())
        reduce_dims = misc.get_dims_with_exclusion(label.dim(), 0)

        loss = (pred - label) ** 2
        delimeter = pred.size(1) * torch.clamp_min(torch.sum(mask, dim=reduce_dims), self.min_area)
        loss = torch.sum(loss, dim=reduce_dims) / delimeter

        return loss
