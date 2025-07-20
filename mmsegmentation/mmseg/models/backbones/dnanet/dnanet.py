import torch
import torch.nn as nn
from ...builder import BACKBONES
from timm.models.layers import DropPath, trunc_normal_
import torch.nn.init as init
from mmcv.runner import load_checkpoint, load_state_dict
from mmseg.utils import get_root_logger


def load_param(channel_size='three', backbone='resnet_18'):

    if channel_size == 'one':
        nb_filter = [4, 8, 16, 32, 64]
    elif channel_size == 'two':
        nb_filter = [8, 16, 32, 64, 128]
    elif channel_size == 'three':
        nb_filter = [16, 32, 64, 128, 256]
    elif channel_size == 'four':
        nb_filter = [32, 64, 128, 256, 512]

    if backbone == 'resnet_10':
        num_blocks = [1, 1, 1, 1]
    elif backbone == 'resnet_18':
        num_blocks = [2, 2, 2, 2]
    elif backbone == 'resnet_34':
        num_blocks = [3, 4, 6, 3]
    elif backbone == 'vgg_10':
        num_blocks = [1, 1, 1, 1]

    return nb_filter, num_blocks


class VGG_CBAM_Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.ca = ChannelAttention(out_channels)
        self.sa = SpatialAttention()

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.ca(out) * out
        out = self.sa(out) * out
        out = self.relu(out)
        return out


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


@BACKBONES.register_module()
class Res_CBAM_block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(Res_CBAM_block, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if stride != 1 or out_channels != in_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = None

        self.ca = ChannelAttention(out_channels)
        self.sa = SpatialAttention()

    def forward(self, x):
        residual = x
        if self.shortcut is not None:
            residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.ca(out) * out
        out = self.sa(out) * out
        out += residual
        out = self.relu(out)
        return out


@BACKBONES.register_module()
class DNANet(nn.Module):
    """
    Paper name: Dense Nested Attention Network for Infrared Small Target Detection
    Paper link: https://arxiv.org/pdf/2106.00487
    GitHub link: https://github.com/YeRen123455/Infrared-Small-Target-Detection

    """

    def __init__(
        self,
        num_classes=1,
        input_channels=3,
        channel_size='three',
        backbone='resnet_18',
        deep_supervision=False,
    ):
        super(DNANet, self).__init__()

        # new add =======
        # 根据 backbone 类型选择 block 模块
        if 'resnet' in backbone:
            block = Res_CBAM_block
        elif 'vgg' in backbone:
            block = VGG_CBAM_Block
        else:
            raise ValueError(f"Unsupported backbone type: {backbone}")
        nb_filter, num_blocks = load_param(channel_size, backbone)
        # new add =======
        self.relu = nn.ReLU(inplace=True)
        self.deep_supervision = deep_supervision
        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.down = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=True)

        self.up_4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.up_8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.up_16 = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)

        self.conv0_0 = self._make_layer(block, input_channels, nb_filter[0])
        self.conv1_0 = self._make_layer(
            block, nb_filter[0], nb_filter[1], num_blocks[0]
        )
        self.conv2_0 = self._make_layer(
            block, nb_filter[1], nb_filter[2], num_blocks[1]
        )
        self.conv3_0 = self._make_layer(
            block, nb_filter[2], nb_filter[3], num_blocks[2]
        )
        self.conv4_0 = self._make_layer(
            block, nb_filter[3], nb_filter[4], num_blocks[3]
        )

        self.conv0_1 = self._make_layer(
            block, nb_filter[0] + nb_filter[1], nb_filter[0]
        )
        self.conv1_1 = self._make_layer(
            block,
            nb_filter[1] + nb_filter[2] + nb_filter[0],
            nb_filter[1],
            num_blocks[0],
        )
        self.conv2_1 = self._make_layer(
            block,
            nb_filter[2] + nb_filter[3] + nb_filter[1],
            nb_filter[2],
            num_blocks[1],
        )
        self.conv3_1 = self._make_layer(
            block,
            nb_filter[3] + nb_filter[4] + nb_filter[2],
            nb_filter[3],
            num_blocks[2],
        )

        self.conv0_2 = self._make_layer(
            block, nb_filter[0] * 2 + nb_filter[1], nb_filter[0]
        )
        self.conv1_2 = self._make_layer(
            block,
            nb_filter[1] * 2 + nb_filter[2] + nb_filter[0],
            nb_filter[1],
            num_blocks[0],
        )
        self.conv2_2 = self._make_layer(
            block,
            nb_filter[2] * 2 + nb_filter[3] + nb_filter[1],
            nb_filter[2],
            num_blocks[1],
        )

        self.conv0_3 = self._make_layer(
            block, nb_filter[0] * 3 + nb_filter[1], nb_filter[0]
        )
        self.conv1_3 = self._make_layer(
            block,
            nb_filter[1] * 3 + nb_filter[2] + nb_filter[0],
            nb_filter[1],
            num_blocks[0],
        )

        self.conv0_4 = self._make_layer(
            block, nb_filter[0] * 4 + nb_filter[1], nb_filter[0]
        )

        self.conv0_4_final = self._make_layer(block, nb_filter[0] * 5, nb_filter[0])

        self.conv0_4_1x1 = nn.Conv2d(
            nb_filter[4], nb_filter[0], kernel_size=1, stride=1
        )
        self.conv0_3_1x1 = nn.Conv2d(
            nb_filter[3], nb_filter[0], kernel_size=1, stride=1
        )
        self.conv0_2_1x1 = nn.Conv2d(
            nb_filter[2], nb_filter[0], kernel_size=1, stride=1
        )
        self.conv0_1_1x1 = nn.Conv2d(
            nb_filter[1], nb_filter[0], kernel_size=1, stride=1
        )

        if self.deep_supervision:
            self.final1 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final4 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
        else:
            self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

    def _make_layer(self, block, input_channels, output_channels, num_blocks=1):
        layers = []
        layers.append(block(input_channels, output_channels))
        for i in range(num_blocks - 1):
            layers.append(block(output_channels, output_channels))
        return nn.Sequential(*layers)

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """

        def _init_weights(m):
            if isinstance(m, nn.Conv2d):
                init.xavier_normal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

        if isinstance(pretrained, str):
            self.apply(_init_weights)
            logger = get_root_logger()
            logger.info(f"load model from: {pretrained}")
            checkpoint = torch.load(pretrained)
            state_dict = checkpoint.get('state_dict', checkpoint)  # 兼容直接保存的模型
            # load state_dict
            # self.load_state_dict(state_dict, strict=True) # ok
            load_state_dict(self, state_dict, strict=True, logger=logger)  # ok

        elif pretrained is None:
            self.apply(_init_weights)
        else:
            raise TypeError("pretrained must be a str or None")

    def forward(self, x):

        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0), self.down(x0_1)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0), self.down(x1_1)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1), self.down(x0_2)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0), self.down(x2_1)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1), self.down(x1_2)], 1))
        x1_3 = self.conv1_3(
            torch.cat([x1_0, x1_1, x1_2, self.up(x2_2), self.down(x0_3)], 1)
        )
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        Final_x0_4 = self.conv0_4_final(
            torch.cat(
                [
                    self.up_16(self.conv0_4_1x1(x4_0)),
                    self.up_8(self.conv0_3_1x1(x3_1)),
                    self.up_4(self.conv0_2_1x1(x2_2)),
                    self.up(self.conv0_1_1x1(x1_3)),
                    x0_4,
                ],
                1,
            )
        )

        if self.deep_supervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(Final_x0_4)

            return [output1, output2, output3, output4]
        else:
            output = self.final(Final_x0_4)

            return output


if __name__ == "__main__":
    import torch

    x = torch.randn(1, 3, 256, 256).cuda()
    nb_filter, num_blocks = load_param('three', 'resnet_18')
    net = DNANet(
        num_classes=1,
        input_channels=3,
        block=Res_CBAM_block,
        num_blocks=num_blocks,
        nb_filter=nb_filter,
        deep_supervision=False,
    ).cuda()

    out = net(x)
    print(out.shape)
