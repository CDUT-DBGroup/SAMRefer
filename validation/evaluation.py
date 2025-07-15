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

# 这个是我新加的，不一定稳定可用，需要测试
def validate(model, val_loader, device, use_fp16=False, use_bf16=False):
    model.eval()

    total_intersection = 0.0
    total_union = 0.0
    all_ious = []
    all_gious = []

    total_nt_samples = 0
    correct_nt_preds = 0

    all_pointms = []
    best_iou = 0.0

    with torch.no_grad():
        for batch_idx, (samples, targets) in enumerate(tqdm(val_loader, desc='Validating')):
            img = samples['img'].to(device, non_blocking=True)#.half()
            # if use_fp16:
            #     img = samples['img'].to(device, non_blocking=True).half()
            # elif use_bf16:
            #     img = samples['img'].to(device, non_blocking=True).to(torch.bfloat16)
            # else:
            #     img = samples['img'].to(device, non_blocking=True)
            word_ids = samples['word_ids'].to(device, non_blocking=True)
            word_masks = samples['word_masks'].to(device, non_blocking=True)
            target = targets['mask'].to(device, non_blocking=True).squeeze(1)  # [B,H,W]

            pred_masks = model(img, word_ids, word_masks)
            if pred_masks.ndim == 4:
                pred_masks = pred_masks.squeeze(1)

            # 显式二值化处理（避免预测值不在0/1）
            pred_masks = (pred_masks > 0.5).float()

            if batch_idx < 3:
                pred_min = pred_masks.min().item()
                pred_max = pred_masks.max().item()
                pred_mean = pred_masks.float().mean().item()
                target_min = target.min().item()
                target_max = target.max().item()
                target_mean = target.float().mean().item()
                print(f"Batch {batch_idx} - Pred stats: min={pred_min:.4f}, max={pred_max:.4f}, mean={pred_mean:.4f}")
                print(f"Batch {batch_idx} - Target stats: min={target_min:.4f}, max={target_max:.4f}, mean={target_mean:.4f}")

            for pred, tgt in zip(pred_masks, target):
                pred_bool = pred.bool()
                tgt_bool = tgt.bool()

                intersection = (pred_bool & tgt_bool).float().sum().item()
                union = (pred_bool | tgt_bool).float().sum().item()
                pred_sum = pred_bool.sum().item()
                tgt_sum = tgt_bool.sum().item()

                ###### 👇 1. gIoU 计算（对所有样本）
                if tgt_sum == 0:  # 无目标样本
                    total_nt_samples += 1
                    if pred_sum == 0:
                        correct_nt_preds += 1
                        giou = 1.0
                    else:
                        giou = 0.0
                else:  # 有前景目标
                    giou = intersection / (union + 1e-6)
                    all_ious.append(giou)  # 仅对前景样本累计 mIoU
                    total_intersection += intersection
                    total_union += union
                    best_iou = max(best_iou, giou)
                    all_pointms.append(calculate_point_metric(pred, tgt))

                all_gious.append(giou)  # 所有样本都进 gIoU

    # 计算平均指标
    avg_miou = np.mean(all_ious) if all_ious else 0.0
    avg_pointm = np.mean(all_pointms) if all_pointms else 0.0
    overall_iou = total_intersection / total_union if total_union > 0 else 0.0
    avg_giou = np.mean(all_gious) if all_gious else 0.0
    acc = correct_nt_preds / total_nt_samples if total_nt_samples > 0 else 0.0

    return {
        'mIoU': avg_miou,         # 只针对前景样本
        'oIoU': overall_iou,      # 前景样本的累计交并比
        'gIoU': avg_giou,         # 所有样本（包括无目标）综合平均IoU
        'Acc': acc,               # 无目标样本准确率
        'pointM': avg_pointm,
        'best_IoU': best_iou
    }


# 这个是稳定可用的
# def validate(model, val_loader, device):
#     model.eval()

#     total_intersection = 0.0
#     total_union = 0.0
#     all_ious = []
#     all_pointms = []
#     best_iou = 0.0

#     with torch.no_grad():
#         for batch_idx, (samples, targets) in enumerate(tqdm(val_loader, desc='Validating')):
#             img = samples['img'].to(device, non_blocking=True)
#             word_ids = samples['word_ids'].to(device, non_blocking=True)
#             word_masks = samples['word_masks'].to(device, non_blocking=True)
#             target = targets['mask'].to(device, non_blocking=True).squeeze(1)  # [B,H,W]

#             pred_masks = model(img, word_ids, word_masks)
#             if pred_masks.ndim == 4:
#                 pred_masks = pred_masks.squeeze(1)
            
#             # 打印前几个batch的预测统计信息用于调试
#             if batch_idx < 3:
#                 pred_min = pred_masks.min().item()
#                 pred_max = pred_masks.max().item()
#                 pred_mean = pred_masks.float().mean().item()
#                 target_min = target.min().item()
#                 target_max = target.max().item()
#                 target_mean = target.float().mean().item()
#                 print(f"Batch {batch_idx} - Pred stats: min={pred_min:.4f}, max={pred_max:.4f}, mean={pred_mean:.4f}")
#                 print(f"Batch {batch_idx} - Target stats: min={target_min:.4f}, max={target_max:.4f}, mean={target_mean:.4f}")
            
#             # 模型已经输出了二值化结果（0或1），直接转换为float
#             pred_masks = pred_masks.float()

#             for pred, tgt in zip(pred_masks, target):
#                 pred_bool = pred.bool()
#                 tgt_bool = tgt.bool()

#                 intersection = (pred_bool & tgt_bool).float().sum().item()
#                 union = (pred_bool | tgt_bool).float().sum().item()

#                 if union > 0:
#                     total_intersection += intersection
#                     total_union += union

#                 iou = intersection / (union + 1e-6)  # 更稳健
#                 all_ious.append(iou)

#                 all_pointms.append(calculate_point_metric(pred, tgt))
#                 best_iou = max(best_iou, iou)

#     avg_miou = np.mean(all_ious)
#     avg_pointm = np.mean(all_pointms)
#     overall_iou = total_intersection / total_union if total_union > 0 else 0.0

#     return {
#         'mIoU': avg_miou,
#         'oIoU': overall_iou,
#         'IoU': overall_iou,
#         'pointM': avg_pointm,
#         'best_IoU': best_iou
#     }

