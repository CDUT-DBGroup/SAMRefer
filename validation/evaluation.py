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


def get_bbox_from_mask(mask):
    """
    从mask中提取边界框 [x1, y1, x2, y2]
    如果mask为空，返回 [0, 0, 0, 0]
    """
    if mask.sum() == 0:
        return torch.tensor([0, 0, 0, 0], device=mask.device, dtype=torch.float32)
    
    h, w = mask.shape
    coords = torch.nonzero(mask, as_tuple=False)
    if len(coords) == 0:
        return torch.tensor([0, 0, 0, 0], device=mask.device, dtype=torch.float32)
    
    y_min = coords[:, 0].min().item()
    y_max = coords[:, 0].max().item()
    x_min = coords[:, 1].min().item()
    x_max = coords[:, 1].max().item()
    
    return torch.tensor([x_min, y_min, x_max, y_max], device=mask.device, dtype=torch.float32)


def calculate_ciou(pred_mask, target_mask):
    """
    计算Complete IoU (cIoU) for segmentation masks
    cIoU = IoU - (d^2 / c^2) - alpha * v
    其中：
    - IoU: 标准IoU
    - d: 预测和真实边界框中心点的欧氏距离
    - c: 包围两个边界框的最小闭包区域的对角线长度
    - v: 长宽比相似性度量
    - alpha: 权重系数
    """
    # 计算标准IoU
    pred_bool = pred_mask.bool()
    tgt_bool = target_mask.bool()
    
    intersection = (pred_bool & tgt_bool).float().sum()
    union = (pred_bool | tgt_bool).float().sum()
    
    if union == 0:
        # 如果union为0，检查是否都是空mask
        if intersection == 0:
            return 1.0  # 两个都是空，完全匹配
        else:
            return 0.0  # 不应该发生，但安全处理
    
    iou = intersection / union
    
    # 获取边界框
    pred_bbox = get_bbox_from_mask(pred_bool)
    tgt_bbox = get_bbox_from_mask(tgt_bool)
    
    # 如果任一mask为空，返回IoU（因为没有边界框信息）
    if (pred_bbox[2] - pred_bbox[0]) == 0 or (pred_bbox[3] - pred_bbox[1]) == 0:
        if (tgt_bbox[2] - tgt_bbox[0]) == 0 or (tgt_bbox[3] - tgt_bbox[1]) == 0:
            return iou.item()  # 两个都是空，返回IoU
        else:
            return 0.0  # 预测为空但目标不为空
    if (tgt_bbox[2] - tgt_bbox[0]) == 0 or (tgt_bbox[3] - tgt_bbox[1]) == 0:
        return 0.0  # 目标为空但预测不为空
    
    # 计算中心点
    pred_cx = (pred_bbox[0] + pred_bbox[2]) / 2.0
    pred_cy = (pred_bbox[1] + pred_bbox[3]) / 2.0
    tgt_cx = (tgt_bbox[0] + tgt_bbox[2]) / 2.0
    tgt_cy = (tgt_bbox[1] + tgt_bbox[3]) / 2.0
    
    # 计算中心点距离的平方
    d_squared = (pred_cx - tgt_cx) ** 2 + (pred_cy - tgt_cy) ** 2
    
    # 计算最小闭包区域
    enclose_x1 = min(pred_bbox[0].item(), tgt_bbox[0].item())
    enclose_y1 = min(pred_bbox[1].item(), tgt_bbox[1].item())
    enclose_x2 = max(pred_bbox[2].item(), tgt_bbox[2].item())
    enclose_y2 = max(pred_bbox[3].item(), tgt_bbox[3].item())
    
    # 计算对角线长度的平方
    c_squared = (enclose_x2 - enclose_x1) ** 2 + (enclose_y2 - enclose_y1) ** 2
    
    if c_squared == 0:
        return iou.item()
    
    # 计算长宽比相似性
    pred_w = pred_bbox[2] - pred_bbox[0]
    pred_h = pred_bbox[3] - pred_bbox[1]
    tgt_w = tgt_bbox[2] - tgt_bbox[0]
    tgt_h = tgt_bbox[3] - tgt_bbox[1]
    
    # 避免除零
    if pred_h == 0 or tgt_h == 0:
        v = 0.0
    else:
        v = (4.0 / (np.pi ** 2)) * ((torch.atan(tgt_w / tgt_h) - torch.atan(pred_w / pred_h)) ** 2)
    
    # 计算alpha
    alpha = v / ((1 - iou) + v + 1e-6)
    
    # 计算cIoU
    ciou = iou - (d_squared / c_squared) - alpha * v
    
    return ciou.item()


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
def validate(model, val_loader, device, use_fp16=False, use_bf16=False, use_negative_masks=False, 
             use_best_sentence=False, sentence_aggregation='mean'):
    """
    Args:
        use_best_sentence: 如果为True，对每个样本的所有描述进行推理
        sentence_aggregation: 多描述聚合方式
            - 'best': 选择IoU最高的预测（可能不公平）
            - 'mean': 对所有描述的预测mask取平均后二值化（推荐，公平）
            - 'mean_iou': 对所有描述的IoU取平均（最公平，但需要多次推理）
            - 'median': 选择IoU中位数的预测
            - 'first': 使用第一个描述（默认行为）
    """
    model.eval()
    
    # 确保模型的所有参数都在正确的设备上
    # 获取模型实际所在的设备（通过检查第一个参数）
    model_device = next(model.parameters()).device
    if model_device != device:
        print(f"Warning: Model is on {model_device}, but data will be on {device}. Moving model to {device}...")
        model = model.to(device)
        model_device = device

    total_intersection = 0.0
    total_union = 0.0
    all_ious = []
    all_gious = []

    total_nt_samples = 0
    correct_nt_preds = 0

    all_pointms = []
    # cIoU: 所有样本的总交集像素数除以总并集像素数
    total_ciou_intersection = 0.0
    total_ciou_union = 0.0
    best_iou = 0.0

    # 获取数据集对象以访问所有描述
    dataset = val_loader.dataset
    if hasattr(dataset, 'dataset'):  # 如果是ConcatDataset，需要特殊处理
        # 对于ConcatDataset，我们无法直接访问所有描述，回退到普通模式
        if use_best_sentence:
            print("Warning: use_best_sentence is not supported for ConcatDataset, using first sentence instead")
            use_best_sentence = False

    with torch.no_grad():
        for batch_idx, (samples, targets) in enumerate(tqdm(val_loader, desc='Validating')):
            # 保持与训练阶段一致的精度设置，避免 dtype 不匹配
            # 确保数据移动到模型所在的设备
            if use_fp16:
                img = samples['img'].to(model_device, non_blocking=True).half()
            elif use_bf16:
                img = samples['img'].to(model_device, non_blocking=True).to(torch.bfloat16)
            else:
                img = samples['img'].to(model_device, non_blocking=True)
            target = targets['mask'].to(model_device, non_blocking=True).squeeze(1)  # [B,H,W]

            if use_best_sentence and 'all_word_ids' in samples:
                # 对每个样本的所有描述进行推理，根据聚合方式选择结果
                batch_size = img.shape[0]
                final_pred_masks = []
                
                for i in range(batch_size):
                    img_i = img[i:i+1]
                    target_i = target[i:i+1]
                    all_word_ids_i = samples['all_word_ids'][i]
                    all_word_masks_i = samples['all_word_masks'][i]
                    
                    if len(all_word_ids_i) == 0:
                        # 如果没有描述，使用默认方式
                        word_ids_i = samples['word_ids'][i:i+1].to(model_device, non_blocking=True)
                        word_masks_i = samples['word_masks'][i:i+1].to(model_device, non_blocking=True)
                        pred_mask_i = model(img_i, word_ids_i, word_masks_i, use_negative_masks=use_negative_masks)
                        if pred_mask_i.ndim == 4:
                            pred_mask_i = pred_mask_i.squeeze(1)
                        final_pred_masks.append(pred_mask_i[0])
                        continue
                    
                    # 对每个描述进行推理
                    all_pred_masks = []
                    sentence_ious = []  # 重命名局部变量，避免覆盖全局all_ious
                    
                    for word_ids_i, word_masks_i in zip(all_word_ids_i, all_word_masks_i):
                        # 确保数据在模型所在的设备上
                        word_ids_i = word_ids_i.unsqueeze(0).to(model_device, non_blocking=True)
                        word_masks_i = word_masks_i.unsqueeze(0).to(model_device, non_blocking=True)
                        
                        pred_mask_i = model(img_i, word_ids_i, word_masks_i, use_negative_masks=use_negative_masks)
                        if pred_mask_i.ndim == 4:
                            pred_mask_i = pred_mask_i.squeeze(1)
                        
                        pred_mask_i = pred_mask_i[0]  # [H, W]
                        # 确保是浮点数类型，以便后续计算平均值
                        if pred_mask_i.dtype != torch.float32 and pred_mask_i.dtype != torch.float16 and pred_mask_i.dtype != torch.bfloat16:
                            pred_mask_i = pred_mask_i.float()
                        all_pred_masks.append(pred_mask_i)
                        
                        # 计算IoU（用于best/median/mean_iou模式）
                        pred_bool = (pred_mask_i > 0.5).bool()
                        tgt_bool = target_i[0].bool()
                        intersection = (pred_bool & tgt_bool).float().sum().item()
                        union = (pred_bool | tgt_bool).float().sum().item()
                        
                        if union > 0:
                            iou_score = intersection / union
                        else:
                            iou_score = 1.0 if intersection == 0 else 0.0
                        sentence_ious.append(iou_score)
                    
                    # 根据聚合方式选择最终预测
                    if sentence_aggregation == 'best':
                        # 选择IoU最高的预测
                        best_idx = np.argmax(sentence_ious)
                        final_pred = all_pred_masks[best_idx]
                    elif sentence_aggregation == 'mean':
                        # 对所有预测取平均，然后二值化（推荐，公平）
                        stacked_preds = torch.stack(all_pred_masks, dim=0)  # [N, H, W]
                        mean_pred = stacked_preds.mean(dim=0)  # [H, W]
                        final_pred = mean_pred
                    elif sentence_aggregation == 'mean_iou':
                        # 对所有IoU取平均，然后选择IoU最接近平均值的预测
                        mean_iou_value = np.mean(sentence_ious)
                        # 找到IoU最接近平均值的索引
                        iou_diffs = [abs(iou - mean_iou_value) for iou in sentence_ious]
                        closest_idx = np.argmin(iou_diffs)
                        final_pred = all_pred_masks[closest_idx]
                    elif sentence_aggregation == 'median':
                        # 选择IoU中位数的预测
                        median_idx = np.argsort(sentence_ious)[len(sentence_ious) // 2]
                        final_pred = all_pred_masks[median_idx]
                    elif sentence_aggregation == 'first':
                        # 使用第一个描述（默认行为）
                        final_pred = all_pred_masks[0]
                    else:
                        # 默认使用平均方式
                        stacked_preds = torch.stack(all_pred_masks, dim=0)
                        final_pred = stacked_preds.mean(dim=0)
                    
                    final_pred_masks.append(final_pred)
                
                pred_masks = torch.stack(final_pred_masks, dim=0)
            else:
                # 使用提供的描述进行推理
                # 确保数据在模型所在的设备上
                word_ids = samples['word_ids'].to(model_device, non_blocking=True)
                word_masks = samples['word_masks'].to(model_device, non_blocking=True)
                
                pred_masks = model(img, word_ids, word_masks, use_negative_masks=use_negative_masks)
                if pred_masks.ndim == 4:
                    pred_masks = pred_masks.squeeze(1)

            # 保存未二值化的预测用于pointM计算
            pred_masks_raw = pred_masks.clone()
            
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

            for idx, (pred, tgt) in enumerate(zip(pred_masks, target)):
                pred_bool = pred.bool()
                tgt_bool = tgt.bool()

                intersection = (pred_bool & tgt_bool).float().sum().item()
                union = (pred_bool | tgt_bool).float().sum().item()
                pred_sum = pred_bool.sum().item()
                tgt_sum = tgt_bool.sum().item()

                ###### 👇 计算当前样本的IoU（用于mIoU和oIoU，包含所有样本）
                # 对于所有样本都计算IoU
                if union == 0:
                    sample_iou = 1.0 if intersection == 0 else 0.0
                else:
                    sample_iou = intersection / union
                
                # mIoU和oIoU: 累计所有样本（包括无目标样本）
                all_ious.append(sample_iou)
                total_intersection += intersection
                total_union += union
                
                ###### 👇 gIoU 计算（对所有样本）
                # 注意：这里命名为gIoU，但对于有前景样本实际计算的是普通IoU
                # 对于无目标样本，gIoU=1.0（预测为空且目标为空）或0.0（预测不为空但目标为空）
                
                # cIoU: 累计所有样本的intersection和union（包括无目标样本）
                total_ciou_intersection += intersection
                total_ciou_union += union
                
                if tgt_sum == 0:  # 无目标样本
                    total_nt_samples += 1
                    if pred_sum == 0:
                        correct_nt_preds += 1
                        giou = 1.0
                    else:
                        giou = 0.0
                else:  # 有前景目标
                    giou = sample_iou  # 有前景目标时，gIoU等于普通IoU
                    best_iou = max(best_iou, giou)
                    # pointM需要使用未二值化的预测（概率值）
                    all_pointms.append(calculate_point_metric(pred_masks_raw[idx], tgt))

                all_gious.append(giou)  # 所有样本都进 gIoU

    # 计算平均指标
    avg_miou = np.mean(all_ious) if all_ious else 0.0
    avg_pointm = np.mean(all_pointms) if all_pointms else 0.0
    overall_iou = total_intersection / total_union if total_union > 0 else 0.0
    avg_giou = np.mean(all_gious) if all_gious else 0.0
    # cIoU: 所有样本的总交集像素数除以总并集像素数
    avg_ciou = total_ciou_intersection / total_ciou_union if total_ciou_union > 0 else 0.0
    acc = correct_nt_preds / total_nt_samples if total_nt_samples > 0 else 0.0

    return {
        'mIoU': avg_miou,         # 所有样本的平均IoU（包括无目标样本）
        'oIoU': overall_iou,      # 所有样本的累计交并比（总交集/总并集）
        'gIoU': avg_giou,         # 所有样本（包括无目标）综合平均IoU
        'cIoU': avg_ciou,         # 所有样本的累计IoU（总交集/总并集）
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

