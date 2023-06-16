import argparse
import os
import sys
from functools import partial
from pathlib import Path

import albumentations
import torch
from albumentations.pytorch import ToTensorV2
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from torch import nn
from torchvision import models as torchvision_models

from segmentation_dataset import ImageMaskModule
from aug_segmentation import DataAugmentationImgMask
from segmentation_finetune import SegmentationFinetuneModule
import pytorch_lightning as pl
import utils
from args import get_args_parser_segmentation
from resnet_unet import SResUnet18
import torchvision.transforms.functional as F


def mask_fn_single(x):
    return x.long() - 1


def mask_fn_double(x, sz=512):
    return F.center_crop(x.long(), sz) - 1


def image_fn_sz(x, sz=512):
    x_h = F.center_crop(x, sz)
    x_l = F.resize(x, sz)
    return x_h, x_l


def get_augmentation(args):
    sz = args.resize_image_size
    train_aug = albumentations.Compose([
        albumentations.Resize(sz, sz),
        albumentations.RandomResizedCrop(sz, sz, scale=(0.4, 1.0)),
        albumentations.HorizontalFlip(),
        albumentations.ColorJitter(0.4, 0.4, 0.4, 0.1, p=0.8),
        albumentations.Normalize(mean=[0.7175, 0.4949, 0.6784], std=[0.1996, 0.2345, 0.1772]),
        # albumentations.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    eval_aug = albumentations.Compose([
        albumentations.Resize(sz, sz),
        albumentations.Normalize(mean=[0.7175, 0.4949, 0.6784], std=[0.1996, 0.2345, 0.1772]),
        # albumentations.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    if args.low_scale_size > 0:
        image_fn = partial(image_fn_sz, sz=args.low_scale_size)
        mask_fn = partial(mask_fn_double, sz=args.low_scale_size)
    else:
        image_fn = None
        mask_fn = mask_fn_single
    train_aug = DataAugmentationImgMask(train_aug, img_fn=image_fn, mask_fn=mask_fn)
    eval_aug = DataAugmentationImgMask(eval_aug, img_fn=image_fn, mask_fn=mask_fn)
    return train_aug, eval_aug


def load_model(args, num_classes):
    if args.arch in ["unet_r18"]:
        if args.pretrained:
            model = SResUnet18(None, pretrained=True, out_channels=num_classes, freeze_encoder=True)
            print(f"Model {args.arch} built with ImageNet pretrained weights")
        else:
            model = SResUnet18(args.pretrained_weights, out_channels=num_classes, freeze_encoder=True)
            print(f"Model {args.arch} built.")
        return model
    else:
        print(f"Unknow architecture: {args.arch}")
        sys.exit(1)
    # load weights to evaluate
    utils.load_pretrained_weights(model, args.pretrained_weights, args.checkpoint_key, args.arch, args.patch_size)
    print(f"Model {args.arch} built.")
    return model


def get_model(args, num_classes):
    network = load_model(args, num_classes)
    loss = nn.CrossEntropyLoss()
    model = SegmentationFinetuneModule(network, loss, args)
    return model


def get_dataset(args):
    augmentations = get_augmentation(args)
    dataset = ImageMaskModule(args.data_path, args.batch_size, augmentations, args.num_workers)
    return dataset


def main(args):

    data_module = get_dataset(args)

    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    trainer_model = get_model(args, args.num_classes)

    logger = WandbLogger(project=args.project, name=args.tag, log_model=True, save_dir=args.output_dir)

    trainer = pl.Trainer(default_root_dir=args.output_dir, gpus=[int(i) for i in args.gpu_id.split(',')],
                         max_epochs=args.epochs, log_every_n_steps=10, num_sanity_val_steps=0,
                         precision=16 if args.use_fp16 else 32,
                         logger=logger,
                         accumulate_grad_batches=args.accumulate_grad_batches,
                         callbacks=lr_monitor,
                         enable_progress_bar=True if args.debug else False)

    trainer.fit(trainer_model, data_module)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('DINO', parents=[get_args_parser_segmentation()])
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)