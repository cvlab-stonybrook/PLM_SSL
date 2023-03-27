_base_ = '../../base.py'
# model settings
model = dict(
    type='LocMatCL',
    pretrained=None,
    queue_len=16384,
    feat_dim=128,
    momentum=0.999,
    loss_lambda=1.0,
    backbone=dict(
        type='ResNet',
        depth=18,
        in_channels=3,
        out_indices=[4],  # 0: conv-1, x: stage-x
        norm_cfg=dict(type='BN')),
    neck=dict(
        type='DenseCLNeck',
        in_channels=512,
        hid_channels=512,
        out_channels=128,
        num_grid=None),
    neck_matching=dict(
        type='MatchingNeck',
        in_channels=128,
        out_channels=128,
        linear=True
    ),
    head=dict(type='WeightedContrastiveHead', temperature=0.2))
# dataset settings
data_source_cfg = dict(type='ImageList')
data_train_list = 'data/NCT/meta/train.txt'
data_train_root = '/gpfs/scratch/jingwezhang/data/NCT-CRC-HE/original_data/NCT-CRC-HE-100K-NONORM'
dataset_type = 'ContrastiveDatasetBoxes'
grid_size = 7
img_norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_pipeline = [
    dict(type='RandomResizedCrop', size=224, scale=(0.2, 1.)),
    dict(
        type='RandomAppliedTrans',
        transforms=[
            dict(
                type='ColorJitter',
                brightness=0.4,
                contrast=0.4,
                saturation=0.4,
                hue=0.1)
        ],
        p=0.8),
    dict(type='RandomGrayscale', p=0.2),
    dict(
        type='RandomAppliedTrans',
        transforms=[
            dict(
                type='GaussianBlur',
                sigma_min=0.1,
                sigma_max=2.0)
        ],
        p=0.5),
    dict(type='RandomHorizontalFlip'),
    #dict(type='ToTensor'),
    #dict(type='Normalize', **img_norm_cfg),
]

prefetch = True
if not prefetch:
    train_pipeline.extend([dict(type='ToTensor'), dict(type='Normalize', **img_norm_cfg)])

batch_size=256
num_gpu=4
data = dict(
    imgs_per_gpu=batch_size//num_gpu,  # total 64*4=256
    workers_per_gpu=4,
    drop_last=True,
    train=dict(
        type=dataset_type,
        data_source=dict(
            list_file=data_train_list, root=data_train_root,
            **data_source_cfg),
        pipeline=train_pipeline,
        prefetch=prefetch,
        grid_size=grid_size
    ))

# optimizer
optimizer = dict(type='SGD', lr=0.03, weight_decay=0.0001, momentum=0.9)
# learning policy
lr_config = dict(policy='CosineAnnealing', min_lr=0.)
checkpoint_config = dict(interval=20)
# runtime settings
total_epochs = 200
