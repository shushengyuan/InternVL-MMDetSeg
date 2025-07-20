# optimizer = dict(type="AdamW", lr=0.0001, weight_decay=1e-5)
optimizer = dict(type="Adam", lr=0.0005, weight_decay=1e-5, betas=(0.9, 0.999))
optimizer_config = dict()
# learning policy
lr_config = dict(
    policy='step',
    step=[5, 10, 15, 20, 25, 30],
    gamma=0.5,  # 每次乘以 0.1
    by_epoch=True,  # 表示按 epoch 而不是 iteration 衰减
)
# lr_config = dict(policy="poly", power=0.9, min_lr=0, by_epoch=False)  # 0.7
# lr_config = dict(warmup='linear', warmup_iters=2000,
#                   warmup_by_epoch=False, policy='CosineAnnealing', min_lr=0, by_epoch=False)
# runtime settings

interval = 1
runner = dict(type="EpochBasedRunner", max_epochs=30)
checkpoint_config = dict(
    by_epoch=True,
    interval=interval,
    max_keep_ckpts=3,
    create_symlink=False,
    save_optimizer=True,
)
#  metric = 'PdFa' -> ['Pd', 'Fa']
#  metric = 'ROC'  -> ['TPR', 'FPR', 'Recall', "Precision"]
#  metric = 'mIoU' -> ['mIoU', 'OA']
# 如果 real_time=True, 单个推理结果直接进行指标计算, 默认为 False, 所有结果推理结束后统一计算
evaluation = dict(
    interval=interval,
    gpu_collect=False,
    by_epoch=True,
    metric=["PdFa", "ROC", "mIoU"],
    ROC_thr=10,
    real_time=True,
    show=False,
)  # nan_to_num=0
find_unused_parameters = True
