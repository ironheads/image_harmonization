from statistics import mode
from iharm.model.base.cdt_net import RefinementModule,CDTNet
import torch.nn as nn
import torch
import numpy as np



input_image = torch.Tensor(np.random.rand(2,3,1024,1024))
input_mask = torch.Tensor(np.random.rand(2,1,1024,1024))

model = CDTNet(4,high_resolution=2048,low_resolution=256)

result = model(input_image,input_mask)


# print(tmp.shape)
# print(mask_sum.shape)
