checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segmenter/vit_base_p16_384_20220308-96dfe169.pth'
backbone_norm_cfg = dict(type='LN', eps=1e-06, requires_grad=True)
model = dict(
    type='EncoderDecoder',
    pretrained=None,
    backbone=dict(
        type='InternViTAdapter',
        pretrain_size=448,
        img_size=512,
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        drop_path_rate=0.1,
        init_values=0.1,
        with_cp=False,
        use_flash_attn=False,
        qk_normalization=True,
        layerscale_force_fp32=False,
        output_dtype='float32',
        last_feat=False,
        freeze_vit=True,
        only_feat_out=True,
        interaction_indexes=[[0, 7], [8, 11], [12, 15], [16, 23]],
        cffn_ratio=0.25,
        deform_ratio=0.25,
        qkv_bias=True,
        norm_type='layer_norm',
        with_simple_fpn=False,
        pretrained='./pretrained/InternVL3-1B/model.safetensors',
        pretrained_type='safe'),
    decode_head=dict(
        type='UPerHead',
        in_channels=[1024, 1024, 1024, 1024],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=512,
        dropout_ratio=0.1,
        num_classes=1,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(type='SoftIoULoss', smooth=1.0, loss_weight=1.0)),
    test_cfg=dict(mode='whole'))
dataset_type = 'NUDTSIRSTDataset'
data_root = '/home/intern/aaai/Infrared-Small-Target-Detection/dataset/NUDT-SIRST'
split_dir = '50_50'
reduce_zero_label = False
ignore_index = 0
background = (0, 0, 0)
train_pipeline = [
    dict(type='LoadDataFromFile', flag='color'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(
        type='DNANetTransforms', test_mode=False, base_size=256,
        crop_size=256),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]
val_pipeline = [
    dict(type='LoadDataFromFile', flag='color'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(256, 256),
        flip=False,
        transforms=[
            dict(
                type='DNANetTransforms',
                test_mode=True,
                base_size=256,
                crop_size=256),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
test_pipeline = [
    dict(type='LoadDataFromFile', flag='color'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(256, 256),
        flip=False,
        transforms=[
            dict(
                type='DNANetTransforms',
                test_mode=True,
                base_size=256,
                crop_size=256),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4,
    train=dict(
        type='NUDTSIRSTDataset',
        pipeline=[
            dict(type='LoadDataFromFile', flag='color'),
            dict(type='LoadAnnotations', reduce_zero_label=False),
            dict(
                type='DNANetTransforms',
                test_mode=False,
                base_size=256,
                crop_size=256),
            dict(type='RandomFlip', prob=0.5, direction='horizontal'),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_semantic_seg'])
        ],
        data_root=
        '/home/intern/aaai/Infrared-Small-Target-Detection/dataset/NUDT-SIRST',
        img_dir='images',
        img_suffix='.png',
        ann_dir='masks',
        seg_map_suffix='.png',
        split_dir='50_50',
        split_suffix='.txt',
        test_mode=False,
        ignore_index=0,
        reduce_zero_label=False,
        label_map=dict({255: 1})),
    val=dict(
        type='NUDTSIRSTDataset',
        pipeline=[
            dict(type='LoadDataFromFile', flag='color'),
            dict(type='LoadAnnotations', reduce_zero_label=False),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(256, 256),
                flip=False,
                transforms=[
                    dict(
                        type='DNANetTransforms',
                        test_mode=True,
                        base_size=256,
                        crop_size=256),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ],
        data_root=
        '/home/intern/aaai/Infrared-Small-Target-Detection/dataset/NUDT-SIRST',
        img_dir='images',
        img_suffix='.png',
        ann_dir='masks',
        seg_map_suffix='.png',
        split_dir='50_50',
        split_suffix='.txt',
        test_mode=True,
        ignore_index=0,
        reduce_zero_label=False,
        label_map=dict({255: 1})),
    test=dict(
        type='NUDTSIRSTDataset',
        pipeline=[
            dict(type='LoadDataFromFile', flag='color'),
            dict(type='LoadAnnotations', reduce_zero_label=False),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(256, 256),
                flip=False,
                transforms=[
                    dict(
                        type='DNANetTransforms',
                        test_mode=True,
                        base_size=256,
                        crop_size=256),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ],
        data_root=
        '/home/intern/aaai/Infrared-Small-Target-Detection/dataset/NUDT-SIRST',
        img_dir='images',
        img_suffix='.png',
        ann_dir='masks',
        seg_map_suffix='.png',
        split_dir='50_50',
        split_suffix='.txt',
        test_mode=True,
        ignore_index=0,
        reduce_zero_label=False,
        label_map=dict({255: 1})))
log_config = dict(
    interval=50, hooks=[dict(type='TextLoggerHook', by_epoch=False)])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True
optimizer_config = dict()
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=1e-07,
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=0.1,
    by_epoch=False)
interval = 2000
runner = dict(type='IterBasedRunner', max_iters=40000)
checkpoint_config = dict(
    by_epoch=False,
    interval=2000,
    max_keep_ckpts=3,
    create_symlink=False,
    save_optimizer=True)
evaluation = dict(
    interval=2000,
    gpu_collect=False,
    by_epoch=False,
    metric=['PdFa', 'ROC', 'mIoU'],
    ROC_thr=10,
    real_time=True,
    show=False,
    save_best='mIoU',
    rule='greater')
find_unused_parameters = True
deepspeed = False
deepspeed_config = 'zero_configs/adam_zero1_fp16.json'
pretrained = './pretrained/InternVL3-1B/model.safetensors'
custom_hooks = [
    dict(type='ParameterFreezeLogHook', priority=50, log_interval=1)
]
optimizer = dict(type='Adam', lr=1e-05, weight_decay=1e-05, betas=(0.9, 0.999))
work_dir = './work_dirs/internvit_nudt_freeze'
gpu_ids = range(0, 2)
auto_resume = False
