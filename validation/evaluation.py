import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm


def calculate_iou(pred_mask, target_mask, thresholded=False):
    if not isinstance(pred_mask, torch.Tensor):
        pred_mask = torch.tensor(pred_mask)
    if not isinstance(target_mask, torch.Tensor):
        target_mask = torch.tensor(target_mask)

    device = pred_mask.device
    target_mask = target_mask.to(device)

    if thresholded:
        pred_mask = pred_mask.bool()
        target_mask = target_mask.bool()
        intersection = (pred_mask & target_mask).float().sum()
        union = (pred_mask | target_mask).float().sum()
    else:
        intersection = (pred_mask * target_mask).sum()
        union = pred_mask.sum() + target_mask.sum() - intersection

    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return (intersection / union).item()


def calculate_miou(pred_masks, target_masks, thresholded=False):
    ious = [calculate_iou(pred, target, thresholded=thresholded) for pred, target in zip(pred_masks, target_masks)]
    return np.mean(ious)


# def calculate_iou(pred_mask, target_mask, thresholded=False):
#     """
#     Calculate IoU between prediction and target masks.
#     If thresholded=False, pred_mask is float probabilities [0,1], target_mask is binary {0,1}.
#     If thresholded=True, pred_mask is binary mask.
#     """
#     if thresholded:
#         pred_mask = pred_mask.bool()
#         target_mask = target_mask.bool()
        
#         intersection = (pred_mask & target_mask).float().sum()
#         union = (pred_mask | target_mask).float().sum()
#     else:
#         # 软IoU计算，直接用概率乘积计算交集，概率和计算并集
#         intersection = (pred_mask * target_mask).sum()
#         union = pred_mask.sum() + target_mask.sum() - intersection

#     if union == 0:
#         return 0
#     return (intersection / union).item()


# def calculate_miou(pred_masks, target_masks):
#     """Calculate mean IoU across all samples in a batch"""
#     ious = [calculate_iou(pred, target) for pred, target in zip(pred_masks, target_masks)]
#     return np.mean(ious)


def calculate_point_metric(pred_mask, target_mask, num_points=100):
    pred_mask = pred_mask  # 这里不阈值化，仍是概率
    target_mask = target_mask.bool()
    
    h, w = target_mask.shape
    points = torch.randint(0, h * w, (num_points,))
    points_h = points // w
    points_w = points % w

    pred_values = pred_mask[points_h, points_w]
    target_values = target_mask[points_h, points_w].float()
    
    # 这里直接计算预测概率和目标0/1的平均差距
    diff = torch.abs(pred_values - target_values)
    score = 1 - diff.mean()  # 1 - 平均误差，越接近1越好
    return score.item()


# def calculate_point_metric(pred_mask, target_mask, num_points=100):
#     """Calculate point-based metric (pointM)"""
#     pred_mask = pred_mask.bool()
#     target_mask = target_mask.bool()
    
#     h, w = target_mask.shape
#     points = torch.randint(0, h * w, (num_points,))
#     points_h = points // w
#     points_w = points % w

#     pred_values = pred_mask[points_h, points_w]
#     target_values = target_mask[points_h, points_w]
    
#     correct = (pred_values == target_values).float().mean()
#     return correct.item()

def validate(model, val_loader, device):
    model.eval()

    total_intersection = 0.0
    total_union = 0.0
    all_ious = []
    all_pointms = []
    best_iou = 0.0

    with torch.no_grad():
        for samples, targets in tqdm(val_loader, desc='Validating'):
            img = samples['img'].to(device, non_blocking=True)
            word_ids = samples['word_ids'].to(device, non_blocking=True)
            word_masks = samples['word_masks'].to(device, non_blocking=True)
            target = targets['mask'].to(device, non_blocking=True).squeeze(1)  # [B,H,W]

            pred_masks = model(img, word_ids, word_masks, target)
            if pred_masks.ndim == 4:
                pred_masks = pred_masks.squeeze(1)
            pred_masks = (pred_masks > 0.5).float()

            for pred, tgt in zip(pred_masks, target):
                pred_bool = pred.bool()
                tgt_bool = tgt.bool()

                intersection = (pred_bool & tgt_bool).float().sum().item()
                union = (pred_bool | tgt_bool).float().sum().item()

                if union > 0:
                    total_intersection += intersection
                    total_union += union

                iou = intersection / (union + 1e-6)  # 更稳健
                all_ious.append(iou)

                all_pointms.append(calculate_point_metric(pred, tgt))
                best_iou = max(best_iou, iou)

    avg_miou = np.mean(all_ious)
    avg_pointm = np.mean(all_pointms)
    overall_iou = total_intersection / total_union if total_union > 0 else 0.0

    return {
        'mIoU': avg_miou,
        'IoU': overall_iou,
        'pointM': avg_pointm,
        'best_IoU': best_iou
    }

