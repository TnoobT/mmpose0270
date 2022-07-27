_base_ = [
    '../../../../_base_/default_runtime.py',
    '../../../../_base_/datasets/mpii.py'
]
evaluation = dict(interval=10, metric='PCKh', save_best='PCKh')

optimizer = dict(
    type='AdamW',
    lr=5e-4,
)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[170, 200])
total_epochs = 210
log_config = dict(
    interval=50, hooks=[
        dict(type='TextLoggerHook'),
    ])

channel_cfg = dict(
    num_output_channels=16,
    dataset_joints=16,
    dataset_channel=list(range(16)),
    inference_channel=list(range(16)))

# model settings
model = dict(
    type='TopDown',
    pretrained='work_dirs/2_hm_shufflenetv2_mpii_256x256/best_PCKh_epoch_200.pth',
    backbone=dict(type='ShuffleNetV2', widen_factor=1.0),
    keypoint_head=dict(
        type='IntegralPoseRegressionHead',  # 对应这个头初始化的几个参数
        in_channels=1024,
        num_joints=channel_cfg['num_output_channels'],
        loss_keypoint=dict(
            type='DSNTRLELoss',
            dsnt_param=dict(use_target_weight=True, sigma = 1, mse_weight=1, js_weight = 1, is_dsnt = True),
            rle_param=dict(use_target_weight=True, size_average=True, residual=True),
            dsnt_weight = 1, 
            rle_weight = 1,
            ),
        out_sigma=True,
        out_highres = True, # 输出高分辨率特征图，bacbone输出扩大两倍，注意扩大分辨率需修改sigma
        ),
    train_cfg=dict(),
    test_cfg=dict(flip_test=True))

data_cfg = dict(
    image_size=[256, 256],
    heatmap_size=[64, 64],
    num_output_channels=channel_cfg['num_output_channels'],
    num_joints=channel_cfg['dataset_joints'],
    dataset_channel=channel_cfg['dataset_channel'],
    inference_channel=channel_cfg['inference_channel'],
    use_gt_bbox=True,
    bbox_file=None,
)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='TopDownGetBboxCenterScale', padding=1.25),
    dict(type='TopDownRandomFlip', flip_prob=0.5),
    dict(
        type='TopDownGetRandomScaleRotation', rot_factor=40, scale_factor=0.5),
    dict(type='TopDownAffine'),
    dict(type='ToTensor'),
    dict(
        type='NormalizeTensor',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
    dict(type='TopDownGenerateTargetRegression'),
    dict(type='TopDownGenerateSimC'),  # 生成向量label
    dict(
        type='Collect',
        keys=['img', 'target', 'target_weight','target_x','target_y'],
        meta_keys=[
            'image_file', 'joints_3d', 'joints_3d_visible', 'center', 'scale',
            'rotation', 'flip_pairs',# 'target_x','target_y'
        ]),
]

val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='TopDownGetBboxCenterScale', padding=1.25),
    dict(type='TopDownAffine'),
    dict(type='ToTensor'),
    dict(
        type='NormalizeTensor',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
    dict(
        type='Collect',
        keys=['img'],
        meta_keys=['image_file', 'center', 'scale', 'rotation', 'flip_pairs']),
]

test_pipeline = val_pipeline

data_root = '/data/tfj/workspace/datasets/MPII'
data = dict(
    samples_per_gpu=64,
    workers_per_gpu=4,
    persistent_workers=True,
    val_dataloader=dict(samples_per_gpu=32),
    test_dataloader=dict(samples_per_gpu=32),
    train=dict(
        type='TopDownMpiiDataset',
        ann_file=f'{data_root}/annotations/mpii_train.json',
        img_prefix=f'{data_root}/images/',
        data_cfg=data_cfg,
        pipeline=train_pipeline,
        dataset_info={{_base_.dataset_info}}),
    val=dict(
        type='TopDownMpiiDataset',
        ann_file=f'{data_root}/annotations/mpii_val.json',
        img_prefix=f'{data_root}/images/',
        data_cfg=data_cfg,
        pipeline=val_pipeline,
        dataset_info={{_base_.dataset_info}}),
    test=dict(
        type='TopDownMpiiDataset',
        ann_file=f'{data_root}/annotations/mpii_val.json',
        img_prefix=f'{data_root}/images/',
        data_cfg=data_cfg,
        pipeline=test_pipeline,
        dataset_info={{_base_.dataset_info}}),
)
