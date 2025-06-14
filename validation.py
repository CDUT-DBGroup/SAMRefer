import torch
import numpy as np
from tqdm import tqdm

def calculate_iou(pred_mask, target_mask, thresholded=False):
    """
    计算IoU。
    thresholded=False时，pred_mask是概率，target_mask是二值；
    thresholded=True时，pred_mask和target_mask都是二值。
    """
    if thresholded:
        pred_mask = pred_mask.bool()
        target_mask = target_mask.bool()
        intersection = (pred_mask & target_mask).float().sum()
        union = (pred_mask | target_mask).float().sum()
    else:
        intersection = (pred_mask * target_mask).sum()
        union = pred_mask.sum() + target_mask.sum() - intersection

    if union == 0:
        return 0.0
    return (intersection / union).item()

def calculate_point_metric(pred_mask, target_mask, num_points=100):
    """
    计算pointM：预测概率与目标0/1差异的平均，越接近1越好
    """
    h, w = target_mask.shape
    points = torch.randint(0, h * w, (num_points,))
    points_h = points // w
    points_w = points % w

    pred_values = pred_mask[points_h, points_w]
    target_values = target_mask[points_h, points_w].float()

    diff = torch.abs(pred_values - target_values)
    score = 1 - diff.mean()
    return score.item()

def validate(model, val_loader, device):
    model.eval()

    total_intersection = 0.0
    total_union = 0.0
    total_miou = 0.0
    total_pointm = 0.0
    best_ciou = 0.0
    best_giou = 0.0

    with torch.no_grad():
        for samples, targets in tqdm(val_loader, desc='Validating'):
            img = samples['img'].to(device, non_blocking=True)
            word_ids = samples['word_ids'].to(device, non_blocking=True)
            word_masks = samples['word_masks'].to(device, non_blocking=True)
            target = targets['mask'].to(device, non_blocking=True).squeeze(1)  # [B,H,W]

            pred_masks = model(img, word_ids, word_masks, target)  # 假设输出是概率，形状[B,1,H,W]或[B,H,W]
            if pred_masks.ndim == 4:
                pred_masks = pred_masks.squeeze(1)  # 变成[B,H,W]

            batch_ious = []
            batch_pointms = []

            for pred, tgt in zip(pred_masks, target):
                # 软IoU计算，不阈值化
                iou = calculate_iou(pred, tgt, thresholded=True)
                batch_ious.append(iou)

                # pointM计算基于概率
                pointm = calculate_point_metric(pred, tgt)
                batch_pointms.append(pointm)

                best_ciou = max(best_ciou, iou)
                best_giou = max(best_giou, iou)  # 你可以改成别的指标

                # 统计整体IoU时使用硬阈值0.5分割
                pred_bin = (pred > 0.5)
                tgt_bin = tgt.bool()
                intersection = (pred_bin & tgt_bin).float().sum().item()
                union = (pred_bin | tgt_bin).float().sum().item()
                if union > 0:
                    total_intersection += intersection
                    total_union += union

            batch_miou = np.mean(batch_ious)
            batch_pointm = np.mean(batch_pointms)

            total_miou += batch_miou
            total_pointm += batch_pointm

    num_batches = len(val_loader)
    avg_miou = total_miou / num_batches
    avg_pointm = total_pointm / num_batches
    overall_iou = total_intersection / total_union if total_union > 0 else 0.0

    return {
        'mIoU': avg_miou,         # batch样本平均软IoU
        'IoU': overall_iou,       # 整体硬阈值IoU
        'pointM': avg_pointm,     # point metric基于概率
        'best_cIoU': best_ciou,
        'best_gIoU': best_giou,
    }