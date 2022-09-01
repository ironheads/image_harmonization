from functools import partial

import torch
from torchvision import transforms
from easydict import EasyDict as edict
from albumentations import HorizontalFlip, Resize, RandomResizedCrop

from iharm.data.compose import ComposeDataset
from iharm.data.hdataset import HDataset
from iharm.data.transforms import HCompose
from iharm.engine.simple_trainer import SimpleHTrainer
from iharm.model import initializer
from iharm.model.base import SSAMImageHarmonization
from iharm.model.losses import MSE
from iharm.model.metrics import DenormalizedMSEMetric, DenormalizedPSNRMetric
from iharm.utils.log import logger

def main(cfg):
    model, model_cfg = init_model(cfg)
    train(model, cfg, model_cfg, start_epoch=cfg.start_epoch)


def init_model(cfg):
    model_cfg = edict()
    model_cfg.crop_size = (256, 256)
    model_cfg.input_normalization = {
        'mean': [.485, .456, .406],
        'std': [.229, .224, .225]
    }
    model_cfg.depth = 4
    model_cfg.input_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(model_cfg.input_normalization['mean'], model_cfg.input_normalization['std']),
    ])
    model = SSAMImageHarmonization(depth=4, batchnorm_from=2, attend_from=2)
    model.to(cfg.device)
    model.apply(initializer.XavierGluon(rnd_type='gaussian', magnitude=2.0))

    return model, model_cfg

def train(model, cfg, model_cfg, start_epoch=0):
    pass