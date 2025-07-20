import torch
import torch.nn as nn
import torch.nn.functional as F
from mmseg.core import add_prefix
from mmseg.ops import resize
from .base import BaseSegmentor
from .. import builder
from ..builder import SEGMENTORS, build_loss
from mmcv.runner import auto_fp16, force_fp32
from ..losses import accuracy
from collections import OrderedDict


@SEGMENTORS.register_module()
class SimpleEncoderDecoder(BaseSegmentor):
    """Custom Simple Encoder Decoder segmentors.

    在将已有网络迁移到 MMSeg 等框架时, 通常必须把整网拆分为 Backbone、Neck、Decode Head、Auxiliary Head 等子模块,
    过程繁琐且容易出错. 现有框架也无法在 “零改动” 代码的情况下直接接入完整网络.
    为尽可能缓解这一痛点, 我们重新设计并实现了关键的 SimpleEncoderDecoder 架构.
    开发者只需将原始网络代码整体复制, 并让其继承 SimpleEncoderDecoder, 即可完成模型移植, 无需再为模块拆分而劳神.

    SimpleEncoderDecoder 一般仅包含一个完整的移植过来的网络, 包含从接受输入到网络最终输出的所有网络模块.

    注意（仅需简单修改的地方）：如果原始网络采用多个 losses, 可能需要自行实现相应的损失函数,
    即 SimpleEncoderDecoder.losses 以及其中的 loss_decode.

    Args:
        network (dict): Config of network.
            Default: dict().
        loss_decode (dict): Config of decode loss.
            Default: dict(type='CrossEntropyLoss').
    """

    def __init__(
        self,
        network=dict(
            type='DNANet',
            num_classes=1,
            input_channels=3,
            channel_size='three',
            backbone='resnet_18',
            deep_supervision=False,
        ),
        pretrained=None,
        train_cfg=None,
        test_cfg=None,
        align_corners=False,
        ignore_index=255,
        loss_cfg=dict(
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0
            ),
            # Whether to compute losses from multiple intermediate feature maps (deep supervision)
            deep_supervision=False,
            # if deep_supervision is True
            ds_losses_cfg=dict(
                aux_loss1=dict(
                    in_indices=[2],
                    losses=[
                        dict(type='CrossEntropyLoss', loss_weight=0.4),
                        # dict(type='DiceLoss', loss_weight=0.1),
                    ],
                ),
                aux_loss2=dict(
                    in_indices=[3],
                    losses=[
                        dict(type='CrossEntropyLoss', loss_weight=0.4),
                        # dict(type='DiceLoss', loss_weight=0.3),
                    ],
                ),
            ),
        ),
    ):
        super(SimpleEncoderDecoder, self).__init__()

        assert (
            isinstance(network, dict) and 'type' in network and len(network) > 0
        ), f"Invalid network config: expected a non-empty dict with'type'key, got {network}"
        self.num_classes = network.get('num_classes', -1)
        assert (
            self.num_classes >= 1
        ), "The network configuration dictionary must include the key'num_classes'."

        self.network = builder.build_backbone(network)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.loss_cfg = loss_cfg

        self.align_corners = align_corners
        self.ignore_index = ignore_index

        self.init_losses()
        self.init_weights(pretrained=pretrained)

    def init_losses(self):

        loss_cfg = self.loss_cfg
        # 主 loss
        loss_decode = loss_cfg.get('loss_decode', None)
        assert (
            loss_decode is not None
        ), "`loss_decode` must be specified in loss_cfg for main supervision."
        self.loss_decode = build_loss(loss_decode)

        # 辅助 loss 初始化
        self.deep_supervision = loss_cfg.get('deep_supervision', False)
        self.ds_losses_cfg = loss_cfg.get('ds_losses_cfg', dict())

        self.ds_loss_decodes = OrderedDict()

        if self.deep_supervision:
            for name, cfg in self.ds_losses_cfg.items():
                in_indices = cfg['in_indices']
                loss_cfgs = cfg.get('losses', [])
                if len(loss_cfgs) == 0:
                    continue

                loss_decodes = [build_loss(lc) for lc in loss_cfgs]
                self.ds_loss_decodes[name] = dict(
                    in_indices=in_indices, losses=loss_decodes
                )

    def init_weights(self, pretrained=None):
        """Initialize the weights in network.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """

        super(SimpleEncoderDecoder, self).init_weights()
        self.network.init_weights(pretrained=pretrained)

    def extract_feat(self, img, **kwargs):
        """Extract features from images."""

        x = self.network(img, **kwargs)

        return x

    def forward_train(self, img, img_metas, gt_semantic_seg):
        """Forward function for training.

        Args:
            img (Tensor): Input images. BxCxHxW
            sequence_imgs (Tensor): Input sequence_imgs. BxTxCxHxW
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """

        x = self.extract_feat(img)

        losses = dict()

        loss = self.losses(x, gt_semantic_seg, img_metas)
        losses.update(loss)

        return losses

    @force_fp32(apply_to=('seg_logit',))
    def losses(self, seg_logit, seg_label, img_metas=None):
        """Compute segmentation loss with optional deep supervision."""
        loss = dict()

        if isinstance(seg_logit, list):
            seg_main = seg_logit[-1]  # 主输出
        else:
            seg_main = seg_logit

        # resize 主分支输出尺寸与标签匹配
        target_size = seg_label.shape[-2:]  # (H, W, C)
        if seg_main.shape[-2:] != target_size:
            seg_main = resize(
                input=seg_main,
                size=seg_label.shape[-2:],  # B C=1 H W
                mode='bilinear',
                align_corners=self.align_corners,
            )

        seg_weight = None
        seg_label = seg_label.squeeze(1)  # B,1,H,W -> B,H,W

        # 主损失
        loss['loss_seg'] = self.loss_decode(
            seg_main.clone(),
            seg_label.clone(),
            weight=seg_weight,
            ignore_index=self.ignore_index,
        )
        
        # 主分支准确率 - 完美处理多维返回值
        acc = accuracy(seg_main, seg_label)
        self._add_accuracy_to_loss(loss, acc, 'acc_seg')

        # 多分支辅助损失
        if self.deep_supervision and isinstance(seg_logit, list):
            for name, cfg in self.ds_loss_decodes.items():
                for idx in cfg['in_indices']:
                    seg_aux = seg_logit[idx]
                    target_size = seg_label.shape[-2:]  # (H, W, C)
                    if seg_aux.shape[-2:] != target_size:
                        seg_aux = resize(
                            input=seg_aux,
                            size=seg_label.shape[-2:],  # B H W
                            mode='bilinear',
                            align_corners=self.align_corners,
                        )
                    
                    # 辅助分支准确率 - 完美处理多维返回值
                    acc_aux = accuracy(seg_aux, seg_label)
                    self._add_accuracy_to_loss(loss, acc_aux, f'acc_aux{name[-1]}_idx{idx}')

                    for i, loss_decode in enumerate(cfg['losses']):
                        loss_decode_name = loss_decode.__class__.__name__
                        loss_name = f'loss_aux{name[-1]}_idx{idx}_{loss_decode_name}'
                        loss[loss_name] = loss_decode(
                            seg_aux.clone(),
                            seg_label.clone(),
                            weight=seg_weight,
                            ignore_index=self.ignore_index,
                        )

        return loss
    
    def _add_accuracy_to_loss(self, loss_dict, acc, base_name):
        """完美处理accuracy返回值，确保日志记录完整且无错误
        
        Args:
            loss_dict (dict): 损失字典
            acc (torch.Tensor or tuple): accuracy函数返回值
            base_name (str): 基础名称，如'acc_seg'
        """
        if isinstance(acc, tuple):
            # 多个topk的情况
            for i, a in enumerate(acc):
                if isinstance(a, torch.Tensor):
                    if a.numel() == 1:
                        # 单个标量
                        loss_dict[f'{base_name}_top{i+1}'] = a
                    else:
                        # 多元素tensor，分别记录每个元素和均值
                        for j in range(a.numel()):
                            loss_dict[f'{base_name}_top{i+1}_cls{j}'] = a.flatten()[j]
                        loss_dict[f'{base_name}_top{i+1}_mean'] = a.mean()
                else:
                    # 标量值
                    loss_dict[f'{base_name}_top{i+1}'] = a
            # 默认使用top1作为主要指标
            if len(acc) > 0:
                main_acc = acc[0]
                if isinstance(main_acc, torch.Tensor) and main_acc.numel() > 1:
                    loss_dict[base_name] = main_acc.mean()
                else:
                    loss_dict[base_name] = main_acc
        else:
            # 单个返回值的情况
            if isinstance(acc, torch.Tensor):
                if acc.numel() == 1:
                    # 单个标量
                    loss_dict[base_name] = acc
                else:
                    # 多元素tensor，分别记录每个元素和均值
                    for i in range(acc.numel()):
                        loss_dict[f'{base_name}_cls{i}'] = acc.flatten()[i]
                    loss_dict[base_name] = acc.mean()
            else:
                # 标量值
                loss_dict[base_name] = acc

    def aug_test(self, imgs, img_metas, rescale=True):
        """Test with augmentations.

        Only rescale=True is supported.
        """
        # aug_test rescale all imgs back to ori_shape for now
        assert rescale
        # to save memory, we get augmented seg logit inplace
        seg_logit = self.inference(imgs[0], img_metas[0], rescale)
        for i in range(1, len(imgs)):
            cur_seg_logit = self.inference(imgs[i], img_metas[i], rescale)
            seg_logit += cur_seg_logit
        seg_logit /= len(imgs)
        seg_pred = seg_logit.argmax(dim=1)
        seg_pred = seg_pred.cpu().numpy()
        # unravel batch dim
        seg_pred = list(seg_pred)

        return seg_pred

    def simple_test(self, img, img_meta, rescale=True):
        """Simple test with single image."""
        seg_logit = self.inference(img, img_meta, rescale)
        if self.num_classes > 1:
            seg_pred = seg_logit.argmax(dim=1)
        else:
            seg_pred = seg_logit

        if torch.onnx.is_in_onnx_export():
            # our inference backend only support 4D output
            seg_pred = seg_pred.unsqueeze(0)
            return seg_pred
        seg_pred = seg_pred.cpu().numpy()
        # unravel batch dim
        seg_pred = list(seg_pred)

        return seg_pred

    def inference(self, img, img_meta, rescale):
        """Inference with slide/whole style.

        Args:
            img (Tensor): The input image of shape (N, 3, H, W).
            sequence_imgs (Tensor): The input sequence img of shape (B, T, 3, H, W)
            img_meta (dict): Image info dict where each dict has: 'img_shape',
                'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            rescale (bool): Whether rescale back to original shape.

        Returns:
            Tensor: The output segmentation map.
        """

        assert self.test_cfg.mode in ['slide', 'whole']
        ori_shape = img_meta[0]['ori_shape']
        assert all(_['ori_shape'] == ori_shape for _ in img_meta)
        if self.test_cfg.mode == 'slide':  # 滑动窗口推理整个图
            seg_logit = self.slide_inference(img, img_meta, rescale)
        else:  # 直接推理整个图
            seg_logit = self.whole_inference(img, img_meta, rescale)

        # 红外小目标网络类别是 1，不需要使用 softmax，尤其是计算评价指标的时候
        if self.num_classes > 1:
            output = F.softmax(seg_logit, dim=1)
        else:
            output = seg_logit

        flip = img_meta[0]['flip']
        if flip:  # todo
            flip_direction = img_meta[0]['flip_direction']
            assert flip_direction in ['horizontal', 'vertical']
            if flip_direction == 'horizontal':  # todo
                if output.ndim == 5:
                    output = output.flip(dim=(4,))
                else:
                    output = output.flip(dims=(3,))
            elif flip_direction == 'vertical':
                if output.ndim == 4:
                    output = output.flip(dim=(3,))
                else:
                    output = output.flip(dims=(2,))

        return output

    def slide_inference(self, img, img_meta, rescale):
        """Inference by sliding-window with overlap.

        If h_crop > h_img or w_crop > w_img, the small patch will be used to
        decode without padding.
        """

        h_stride, w_stride = self.test_cfg.stride
        h_crop, w_crop = self.test_cfg.crop_size
        num_classes = self.num_classes
        if img.ndim == 4:
            batch_size, c_img, h_img, w_img = img.size()
        elif img.ndim == 5:
            batch_size, c_img, t_img, h_img, w_img = img.size()
        preds = img.new_zeros((batch_size, num_classes, h_img, w_img))
        count_mat = img.new_zeros((batch_size, 1, h_img, w_img))

        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                if img.ndim == 4:  # B, C, H, W
                    crop_img = img[:, :, y1:y2, x1:x2]
                elif img.ndim == 5:  # B, C, T, H, W
                    crop_img = img[:, :, :, y1:y2, x1:x2]

                crop_seg_logit = self.encode_decode(crop_img, img_meta)
                preds += F.pad(
                    crop_seg_logit,
                    (
                        int(x1),
                        int(preds.shape[3] - x2),
                        int(y1),
                        int(preds.shape[2] - y2),
                    ),
                )
                count_mat[:, :, y1:y2, x1:x2] += 1

        assert (count_mat == 0).sum() == 0
        if torch.onnx.is_in_onnx_export():
            # cast count_mat to constant while exporting to ONNX
            count_mat = torch.from_numpy(count_mat.cpu().detach().numpy()).to(
                device=img.device
            )

        preds = preds / count_mat
        if rescale:
            preds = self.rescale_seg_logit(preds, img_meta)

        return preds

    def whole_inference(self, img, img_meta, rescale):
        """Inference with full image."""

        seg_logit = self.encode_decode(img, img_meta)

        if rescale:
            seg_logit = self.rescale_seg_logit(seg_logit, img_meta)

        return seg_logit

    def encode_decode(self, img, img_meta):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""

        seg_logit = self.extract_feat(img)

        # 如果网络输出是多尺度特征（list 或 tuple），则只取最后一个特征
        if isinstance(seg_logit, (list, tuple)):
            seg_logit = seg_logit[-1]

        return seg_logit

    def rescale_seg_logit(self, seg_logit, img_meta):

        target_size = img_meta[0]['ori_shape'][:2]  # (H, W, C)

        if seg_logit.shape[-2:] != target_size:
            seg_logit = resize(
                seg_logit,
                size=target_size,
                mode='bilinear',
                align_corners=self.align_corners,
                warning=False,
            )

        return seg_logit
