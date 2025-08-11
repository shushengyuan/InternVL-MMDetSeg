# --------------------------------------------------------
# InternVL
# Copyright (c) 2023 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

_base_ = [
    # '../_base_/models/segmenter_vit-b16_mask.py',
    "../_base_/datasets/wide_irstd_eval_aligned.py",
    '../_base_/default_runtime.py',
    "../_base_/schedules/sirst_schedule_40k.py",
]

deepspeed = True  
deepspeed_config = 'zero_configs/adam_zero2_bf16_fast.json'  # 启用BF16 + ZeRO Stage 2，稳定且更快  
pretrained = './pretrained/InternVL2_5-1B/model.safetensors'

# 添加GPU内存优化配置
gpu_memory_fraction = 0.8  # 限制GPU显存使用比例
allow_growth = True  # 允许显存动态增长

model = dict(
    pretrained=None,
    type='EncoderDecoder', 
    backbone=dict(
        type='InternViTAdapter',
        pretrain_size=448,
        img_size=512, 
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.,
        drop_path_rate=0.1,
        init_values=0.1,
        with_cp=True,
        use_flash_attn=True,
        qk_normalization=True,
        layerscale_force_fp32=False,
        output_dtype="bfloat16",
        last_feat=False,
        freeze_vit=True,  # 冻结backbone以加速训练
        only_feat_out=True,
        interaction_indexes=[[0, 7], [8, 11], [12, 15], [16, 23]],
        cffn_ratio=0.25,
        deform_ratio=0.25,
        qkv_bias=True,
        norm_type='layer_norm',
        with_simple_fpn=False,
        # 动态分辨率配置
        # use_dynamic_resolution=True,   # 启用动态分辨率
        # min_patches=1,                 # 最少patch数量
        # max_patches=6,                 # 最多patch数量（符合官方实现）
        # use_thumbnail=True,           # 是否使用缩略图
        pretrained=pretrained,
        pretrained_type="safe"),
    decode_head=dict(
        type='UPerHead',  # 或者 PSPHead, FPN等适合多尺度特征的head
        in_channels=[1024, 1024, 1024, 1024],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=512,
        dropout_ratio=0.1,
        num_classes=1,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        # norm_cfg=dict(type='BN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type='SoftIoULoss',
            smooth=1.0,
            loss_weight=1.0
        )),
    # Deep Supervision: 添加多个auxiliary heads
    # 使用滑窗推理，避免在原图尺寸下OOM，同时不缩放图像
    test_cfg=dict(mode='slide', crop_size=(512, 512), stride=(224, 224))
)
# optimizer = dict(_delete_=True, type='AdamW', lr=4e-5, betas=(0.9, 0.999), weight_decay=0.0,
#                  constructor='CustomLayerDecayOptimizerConstructor',
#                  paramwise_cfg=dict(num_layers=24, layer_decay_rate=1.0))
# lr_config = dict(_delete_=True, policy='poly',
#                  warmup='linear',
#                  warmup_iters=1500,
#                  warmup_ratio=1e-6,
#                  power=1.0, min_lr=0.0, by_epoch=False)
# By default, models are trained on 8 GPUs with 2 images per GPU
# data = dict(samples_per_gpu=1, workers_per_gpu=8, prefetch_factor=2, persistent_workers=True)

# GPU设置
data = dict(
    samples_per_gpu=1,  # 评估时使用最小batch size
    workers_per_gpu=1,  # 减少worker数量节省显存
    prefetch_factor=1,  # 减少预加载
    persistent_workers=False,  # 评估时不需要持久化
    pin_memory=False,  # 评估时关闭页锁定内存
)

# train_cfg = dict()
# test_cfg = dict(_delete_=True, mode="whole")

# runner = dict(type='IterBasedRunner', max_iters=8000*15)

# CUDA memory management for stability
gpu_multithreading = False  # 禁用GPU多线程，避免内存竞争

if deepspeed:
    checkpoint_config = dict(deepspeed=deepspeed, by_epoch=False, interval=2000, max_keep_ckpts=2)
else:
    checkpoint_config = dict(by_epoch=False, interval=2000, max_keep_ckpts=2)
evaluation = dict(
    metric=['PdFa','mIoU'], 
    save_best='mIoU',  # 保存target类IoU最佳的模型（使用类别名称而不是索引）
    rule='greater',     # 明确指定IoU.target越大越好
)
custom_hooks = [
    dict(
        type='GPUMemoryCleanupHook',
        priority=49),
    # 添加更激进的内存清理hook
    dict(
        type='EmptyCacheHook',
        priority=50),
]
optimizer = dict(type="Adam", lr=0.00005, weight_decay=1e-5, betas=(0.9, 0.999))

# 解决DDP未使用参数问题
find_unused_parameters = False

# 添加环境变量设置
env_cfg = dict(
    cudnn_benchmark=False,  # 评估时关闭cudnn benchmark
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
    deterministic=True,  # 确保结果可复现
)
