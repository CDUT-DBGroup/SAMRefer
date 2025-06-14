import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

def calculate_iou(pred_mask, target_mask):
    """Calculate IoU between prediction and target masks"""
    intersection = (pred_mask & target_mask).float().sum()
    union = (pred_mask | target_mask).float().sum()
    if union == 0:
        return 0
    return (intersection / union).item()

def calculate_miou(pred_masks, target_masks):
    """Calculate mean IoU across all samples in a batch"""
    ious = []
    for pred, target in zip(pred_masks, target_masks):
        iou = calculate_iou(pred, target)
        ious.append(iou)
    return np.mean(ious)

def calculate_point_metric(pred_mask, target_mask, num_points=100):
    """Calculate point-based metric (pointM)"""
    # Sample random points
    h, w = target_mask.shape
    points = torch.randint(0, h * w, (num_points,))
    points_h = points // w
    points_w = points % w
    
    # Get values at sampled points
    pred_values = pred_mask[points_h, points_w]
    target_values = target_mask[points_h, points_w]
    
    # Calculate accuracy
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
            target = targets['mask'].to(device, non_blocking=True).squeeze(1)
            
            # Get predictions
            pred_masks = model(img, word_ids, word_masks, target)
            pred_masks = (pred_masks > 0.5).float()
            
            # Calculate metrics
            batch_miou = calculate_miou(pred_masks, target)
            batch_iou = calculate_iou(pred_masks, target)
            batch_pointm = calculate_point_metric(pred_masks, target)
            
            total_miou += batch_miou
            total_iou += batch_iou
            total_pointm += batch_pointm
            
            # Update best metrics
            best_ciou = max(best_ciou, batch_iou)
            best_giou = max(best_giou, batch_miou)
    
    # Calculate averages
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