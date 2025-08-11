import numpy as np

def batch_pix_accuracy(output, target):
    """计算像素准确率"""
    assert output.shape == target.shape, "Predict and Label Shape Don't Match"
    predict = (output > 0).astype(np.float32)
    pixel_labeled = (target > 0).sum()
    pixel_correct = ((predict == target) * (target > 0)).sum()

    assert pixel_correct <= pixel_labeled, "Correct area should be smaller than Labeled"
    return pixel_correct, pixel_labeled


def batch_intersection_union(output, target, num_classes):
    """计算交并比"""
    mini = 1
    maxi = 1
    nbins = 1
    predict = (output > 0).astype(np.float32)
    intersection = predict * (predict == target)

    area_inter, _ = np.histogram(intersection, bins=nbins, range=(mini, maxi))
    area_pred, _ = np.histogram(predict, bins=nbins, range=(mini, maxi))
    area_lab, _ = np.histogram(target, bins=nbins, range=(mini, maxi))
    area_union = area_pred + area_lab - area_inter

    assert (
        area_inter <= area_union
    ).all(), "Error: Intersection area should be smaller than Union area"
    return area_inter, area_union


class mIoU:
    """计算 mIoU 指标"""
    
    def __init__(self, num_classes, threshold=None):
        super(mIoU, self).__init__()
        self.num_classes = num_classes
        self.threshold = threshold
        self.reset()

    def update(self, preds, labels):
        """更新预测结果"""
        if self.threshold is not None:
            preds = preds > self.threshold
        correct, labeled = batch_pix_accuracy(preds, labels)
        inter, union = batch_intersection_union(preds, labels, self.num_classes)
        self.total_correct += correct
        self.total_label += labeled
        self.total_inter += inter
        self.total_union += union

    def get(self):
        """获取 mIoU 结果"""
        pixAcc = 1.0 * self.total_correct / (np.spacing(1) + self.total_label)
        IoU = 1.0 * self.total_inter / (np.spacing(1) + self.total_union)
        mIoU = IoU.mean()
        return pixAcc, mIoU

    def reset(self):
        """重置统计量"""
        self.total_inter = 0
        self.total_union = 0
        self.total_correct = 0
        self.total_label = 0


def eval_pixel_metrics(
    results,
    gt_seg_maps,
    num_classes,
    ignore_index,
    metrics=['mIoU'],
    nan_to_num=0.0,
):
    """计算像素级评估指标
    
    Args:
        results (list[ndarray]): 预测分割图列表
        gt_seg_maps (list[ndarray]): 真值分割图列表
        num_classes (int): 类别数
        ignore_index (int): 忽略的索引
        metrics (list[str] | str): 要评估的指标，仅支持 'mIoU'
        nan_to_num (int, optional): NaN 替换值
        
    Returns:
        dict: 包含 mIoU 和 OA 的字典
    """
    if isinstance(metrics, str):
        metrics = [metrics]
    
    allowed_metrics = ['mIoU']
    if not set(metrics).issubset(set(allowed_metrics)):
        raise KeyError('metrics {} is not supported'.format(metrics))

    # 初始化 mIoU 计算器
    miou_metric = mIoU(num_classes=num_classes)
    
    # 逐样本更新
    for pred, label in zip(results, gt_seg_maps):
        # 确保形状匹配
        pred = pred.reshape(pred.shape[-2], pred.shape[-1])
        assert pred.shape == label.shape, "Predict and Label Shape Don't Match"
        miou_metric.update(pred, label)

    # 获取结果
    overall_acc, mean_iou = miou_metric.get()
    ret_metrics = {'mIoU': mean_iou, 'OA': overall_acc}

    return ret_metrics
