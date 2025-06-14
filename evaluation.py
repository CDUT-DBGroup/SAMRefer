import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

def calculate_iou(pred_mask, target_mask):
    """Calculate IoU between prediction and target masks"""
    pred_mask = pred_mask.bool()
    target_mask = target_mask.bool()
    
    intersection = (pred_mask & target_mask).float().sum()
    union = (pred_mask | target_mask).float().sum()
    if union == 0:
        return 0
    return (intersection / union).item()

def calculate_miou(pred_masks, target_masks):
    """Calculate mean IoU across all samples in a batch"""
    ious = [calculate_iou(pred, target) for pred, target in zip(pred_masks, target_masks)]
    return np.mean(ious)

def calculate_point_metric(pred_mask, target_mask, num_points=100):
    """Calculate point-based metric (pointM)"""
    pred_mask = pred_mask.bool()
    target_mask = target_mask.bool()
    
    h, w = target_mask.shape
    points = torch.randint(0, h * w, (num_points,))
    points_h = points // w
    points_w = points % w

    pred_values = pred_mask[points_h, points_w]
    target_values = target_mask[points_h, points_w]
    
    correct = (pred_values == target_values).float().mean()
    return correct.item()

def validate(model, val_loader, device):
    """Perform validation and return metrics"""
    model.eval()
    total_miou = 0
    total_iou = 0
    total_pointm = 0
    best_ciou = 0
    best_giou = 0
    
    with torch.no_grad():
        for samples, targets in tqdm(val_loader, desc='Validating'):
            # Move data to device
            img = samples['img'].to(device, non_blocking=True)
            word_ids = samples['word_ids'].to(device, non_blocking=True)
            word_masks = samples['word_masks'].to(device, non_blocking=True)
            target = targets['mask'].to(device, non_blocking=True).squeeze(1)  # [B, H, W]
            
            # Get predictions
            pred_masks = model(img, word_ids, word_masks, target)  # assume output is [B, 1, H, W] or [B, H, W]
            if pred_masks.ndim == 4:
                pred_masks = pred_masks.squeeze(1)
            pred_masks = (pred_masks > 0.5).float()

            batch_ious = []
            batch_pointms = []

            # Per-sample metric calculation
            for pred, tgt in zip(pred_masks, target):
                iou = calculate_iou(pred, tgt)
                pointm = calculate_point_metric(pred, tgt)

                batch_ious.append(iou)
                batch_pointms.append(pointm)

                best_ciou = max(best_ciou, iou)
                best_giou = max(best_giou, iou)  # 如果你要用另一个mIoU指标替代，这里可以修改

            batch_miou = np.mean(batch_ious)
            batch_iou = batch_miou  # mean IoU and IoU在这里相等（你可以区分使用）
            batch_pointm = np.mean(batch_pointms)

            total_miou += batch_miou
            total_iou += batch_iou
            total_pointm += batch_pointm

    num_batches = len(val_loader)
    avg_miou = total_miou / num_batches
    avg_iou = total_iou / num_batches
    avg_pointm = total_pointm / num_batches

    return {
        'mIoU': avg_miou,
        'IoU': avg_iou,
        'pointM': avg_pointm,
        'best_cIoU': best_ciou,
        'best_gIoU': best_giou
    }
