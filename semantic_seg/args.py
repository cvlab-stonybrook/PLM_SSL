import argparse

import torch
from torchvision import models as torchvision_models

import utils


torchvision_archs = sorted(name for name in torchvision_models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(torchvision_models.__dict__[name]))


def get_args_parser_segmentation():
    parser = argparse.ArgumentParser('DINO Segmentation', add_help=False)

    # Model parameters
    parser.add_argument('--arch', default='unet_r18', type=str,
        choices=['vit_tiny', 'vit_small', 'vit_base', 'xcit', 'deit_tiny', 'deit_small'] \
                + torchvision_archs + torch.hub.list("facebookresearch/xcit:main") \
                + ['unet_r18'],
        help="""Name of architecture to train. For quick experiments with ViTs,
        we recommend using vit_tiny or vit_small.""")
    parser.add_argument('--patch_size', default=16, type=int, help="""Size in pixels
        of input square patches - default 16 (for 16x16 patches). Using smaller
        values leads to better performance but requires more memory. Applies only
        for ViTs (vit_tiny, vit_small and vit_base). If <16, we recommend disabling
        mixed precision training (--use_fp16 false) to avoid unstabilities.""")
    parser.add_argument('--pretrained', action="store_true", help='load imagenet pretrained weight')
    parser.add_argument('--pretrained_weights', default='/data07/shared/jzhang/result/LocMatCL/BCSS/resnet18_densecl_ours/bcss_alpha_0.5_ep200/segmentation/backbone.pth', type=str, help="Path to pretrained weights to evaluate.")
    parser.add_argument("--checkpoint_key", default="teacher", type=str,
                        help='Key to use in the checkpoint (example: "teacher")')

    parser.add_argument('--resize_image_size', default=256, type=int)
    parser.add_argument('--low_scale_size', default=0, type=int)
    parser.add_argument('--num_classes', default=21, type=int)

    # Training/Optimization parameters
    parser.add_argument('--use_fp16', type=utils.bool_flag, default=False, help="""Whether or not
        to use half precision for training. Improves training time and memory requirements,
        but can provoke instability and slight decay of performance. We recommend disabling
        mixed precision if the loss is unstable, if reducing the patch size or if training with bigger ViTs.""")
    parser.add_argument('--weight_decay', type=float, default=0.02, help="""Initial value of the
        weight decay. With ViT, a smaller value at the beginning of training works well.""")
    parser.add_argument('--batch_size', default=32, type=int,
        help='Per-GPU batch-size : number of distinct images loaded on one GPU.')
    parser.add_argument('--epochs', default=20, type=int, help='Number of epochs of training.')
    parser.add_argument("--lr", default=5e-4, type=float, help="""Learning rate at the end of
        linear warmup (highest LR used during training). The learning rate is linearly scaled
        with the batch size, and specified here for a reference batch size of 256.""")
    parser.add_argument('--min_lr', type=float, default=1e-6, help="""Target LR at the
        end of optimization. We use a cosine LR schedule with linear warmup.""")
    parser.add_argument('--optimizer', default='adamw', type=str,
        choices=['adamw', 'sgd', 'lars'], help="""Type of optimizer. We recommend using adamw with ViTs.""")

    parser.add_argument('--accumulate_grad_batches', type=int, default=1, help='simulate larger batch size by '
                                                                               'accumulating gradients')

    # Misc
    parser.add_argument('--data_path', default='/data07/shared/jzhang/data/segmentation/BCSS/patches/512_Imagefolder', type=str,
        help='Please specify path to the ImageNet training data.')
    parser.add_argument('--output_dir', default="/data07/shared/jzhang/result/LocMatCL/BCSS/", type=str, help='Path to save logs and checkpoints.')
    parser.add_argument('--project', default="test", type=str, help='')
    parser.add_argument('--tag', default="test_seg", type=str, help='')
    parser.add_argument('--gpu_id', type=str, default="0")
    parser.add_argument('--debug', action="store_true", help='Whether show progressive the bar')

    parser.add_argument('--num_workers', default=8, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument('--org_dino', default=True, type=utils.bool_flag, help="""""")
    return parser
