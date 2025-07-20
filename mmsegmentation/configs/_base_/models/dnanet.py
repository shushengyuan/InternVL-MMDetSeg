# model settings
model = dict(
    type="SimpleEncoderDecoder",
    network=dict(
        type='DNANet',
        num_classes=1,
        input_channels=3,
        channel_size='three',
        backbone='resnet_18',
        deep_supervision=False,
    ),
    pretrained=None,  # 'pretrained/mIoU__DNANet_NUDT-SIRST_epoch.pth.tar'
    align_corners=False,
    ignore_index=-100,  # 计算损失时忽略的值，只要不是 0-num_classes 就行, 默认-100, 255 也没问题！！！
    loss_cfg=dict(
        loss_decode=dict(type='SoftIoULoss', smooth=1.0, loss_weight=1.0),
        # Whether to compute losses from multiple intermediate feature maps (deep supervision)
        deep_supervision=True,
        # if deep_supervision is True
        ds_losses_cfg=dict(
            aux_loss1=dict(
                in_indices=[0, 1, 2],
                losses=[
                    dict(type='SoftIoULoss', smooth=1.0, loss_weight=1.0),
                ],
            )
        ),
    ),
)

# model training and testing settings
train_cfg = dict()
test_cfg = dict(mode="whole")


# def load_param(channel_size='three', backbone='resnet_18'):

#     if channel_size == 'one':
#         nb_filter = [4, 8, 16, 32, 64]
#     elif channel_size == 'two':
#         nb_filter = [8, 16, 32, 64, 128]
#     elif channel_size == 'three':
#         nb_filter = [16, 32, 64, 128, 256]
#     elif channel_size == 'four':
#         nb_filter = [32, 64, 128, 256, 512]

#     if backbone == 'resnet_10':
#         num_blocks = [1, 1, 1, 1]
#     elif backbone == 'resnet_18':
#         num_blocks = [2, 2, 2, 2]
#     elif backbone == 'resnet_34':
#         num_blocks = [3, 4, 6, 3]
#     elif backbone == 'vgg_10':
#         num_blocks = [1, 1, 1, 1]

#     return nb_filter, num_blocks
