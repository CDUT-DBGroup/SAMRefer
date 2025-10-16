import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import functional as F
import math

from .criterion import dice_loss, sigmoid_ce_loss, point_sample, get_uncertain_point_coords_with_randomness, calculate_uncertainty


def focal_loss(inputs, targets, alpha=0.25, gamma=2.0):
    """
    Focal Loss for addressing class imbalance
    Args:
        inputs: A float tensor of arbitrary shape (logits)
        targets: A float tensor with the same shape as inputs
        alpha: Weighting factor for rare class (default: 0.25)
        gamma: Focusing parameter (default: 2.0)
    """
    inputs = inputs.flatten(1)
    targets = targets.flatten(1)
    
    # Compute sigmoid
    probs = torch.sigmoid(inputs)
    
    # Compute focal loss
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = probs * targets + (1 - probs) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)
    
    # Apply alpha weighting
    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    
    return loss.mean(1).mean()


def iou_loss(inputs, targets, smooth=1e-6):
    """
    IoU Loss for segmentation
    Args:
        inputs: A float tensor of arbitrary shape (logits)
        targets: A float tensor with the same shape as inputs
        smooth: Smoothing factor to avoid division by zero
    """
    inputs = torch.sigmoid(inputs)
    inputs = inputs.flatten(1)
    targets = targets.flatten(1)
    
    # Compute intersection and union
    intersection = (inputs * targets).sum(-1)
    union = inputs.sum(-1) + targets.sum(-1) - intersection
    
    # Compute IoU
    iou = (intersection + smooth) / (union + smooth)
    
    # Return 1 - IoU as loss
    return (1 - iou).mean()


def boundary_loss(inputs, targets, kernel_size=3, sdf_k=3.0):
    """
    SDF-weighted boundary loss computed on a narrow band around object boundaries.
    Args:
        inputs: [B, H, W] logits
        targets: [B, H, W] binary masks in {0,1}
        kernel_size: integer band width T (in pixels) around boundary (3~10 reasonable)
        sdf_k: normalization factor for tanh(sdf / sdf_k)
    """
    # Use probability instead of raw logits
    pred_prob = torch.sigmoid(inputs)
    target_binary = (targets > 0.5).float()

    B, H, W = target_binary.shape
    device = target_binary.device
    T = int(max(1, kernel_size))

    # Helper: limited-distance transform via iterative dilations (Chebyshev-like)
    def limited_distance_to_mask(mask: torch.Tensor, max_d: int) -> torch.Tensor:
        # mask: [B, H, W] in {0,1}
        covered = (mask > 0.5)
        dist = torch.full_like(mask, fill_value=float(max_d + 1))
        dist = torch.where(covered, torch.zeros_like(dist), dist)
        cur = covered
        for d in range(1, max_d + 1):
            cur = F.max_pool2d(cur.float().unsqueeze(1), kernel_size=3, stride=1, padding=1).squeeze(1) > 0
            newly = (~covered) & cur
            if newly.any():
                dist = torch.where(newly, torch.full_like(dist, float(d)), dist)
            covered = covered | cur
        return dist

    # Unsigned distances limited to band T (approximate)
    fg = target_binary
    bg = 1.0 - target_binary
    outside_dist = limited_distance_to_mask(fg, T)  # distance from background to nearest foreground
    inside_dist = limited_distance_to_mask(bg, T)    # distance from foreground to nearest background

    # Signed distance: positive inside object, negative outside
    sdf = torch.where(fg > 0.5, inside_dist, -outside_dist)

    # Narrow band mask |sdf| <= T
    band_mask = (sdf.abs() <= T).float()

    # Normalize/scale SDF to avoid large magnitude domination
    sdf_norm = torch.tanh(sdf / float(sdf_k))
    # Weight emphasizing boundary (|sdf| small -> weight high)
    weight = 1.0 - sdf_norm.abs()
    weight = weight * band_mask

    # Weighted BCE on probability
    bce = F.binary_cross_entropy(pred_prob, target_binary, reduction='none')
    weighted = bce * weight
    denom = weight.sum(dim=(1, 2)).clamp(min=1.0)
    loss_per_sample = weighted.sum(dim=(1, 2)) / denom
    return loss_per_sample.mean()


class AdaptiveLossWeighting(nn.Module):
    """
    Improved adaptive loss weighting mechanism with better initialization and normalization
    """
    def __init__(self, num_losses=4, init_weights=None, temperature=1.5, use_uncertainty_weighting=True):
        super().__init__()
        if init_weights is None:
            # 更合理的初始权重：基础loss权重较大，辅助loss权重较小
            init_weights = [1.0, 1.0, 0.5, 0.3, 0.2]  # [ce, dice, focal, iou, boundary]
        
        # 使用log空间初始化，避免权重过小
        self.log_vars = nn.Parameter(torch.log(torch.tensor(init_weights, dtype=torch.float32)))
        self.temperature = temperature
        self.use_uncertainty_weighting = use_uncertainty_weighting
        
        # 添加动量参数用于平滑权重变化
        self.momentum = 0.9
        self.register_buffer('running_weights', torch.ones_like(self.log_vars))
        
    def forward(self, losses):
        """
        Args:
            losses: List of loss tensors
        Returns:
            weighted_loss: Weighted sum of losses
            weights: Current loss weights
        """
        # 计算基础权重
        log_weights = self.log_vars / self.temperature
        
        # 使用sigmoid而不是softmax，避免权重过度归一化
        raw_weights = torch.sigmoid(log_weights)
        
        # 归一化权重，但保持相对比例
        weights = raw_weights / raw_weights.sum() * len(raw_weights)
        
        # 应用动量平滑
        if self.training:
            self.running_weights = self.momentum * self.running_weights + (1 - self.momentum) * weights
        
        # 使用平滑后的权重
        smooth_weights = self.running_weights if self.training else weights
        
        # 计算加权loss - 使用简单的加权求和避免计算图问题
        weighted_loss = torch.zeros_like(losses[0])
        for w, loss in zip(smooth_weights, losses):
            weighted_loss = weighted_loss + w * loss
        
        return weighted_loss, smooth_weights


class EnhancedSegMaskLoss(nn.Module):
    """
    Enhanced segmentation mask loss with multiple loss functions
    """
    def __init__(self, 
                 num_points=112*112, 
                 oversample_ratio=3.0, 
                 importance_sample_ratio=0.75,
                 use_focal=True,
                 use_iou=True,
                 use_boundary=True,
                 use_adaptive_weighting=True,
                 use_curriculum_learning=True,
                 use_dataset_aware_loss=True,
                 focal_alpha=0.25,
                 focal_gamma=2.0,
                 boundary_kernel_size=3,
                 loss_scaling_factors=None,
                 curriculum_schedule=None,
                 dataset_weights=None):
        super(EnhancedSegMaskLoss, self).__init__()
        self.num_points = num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio
        
        # Loss function flags
        self.use_focal = use_focal
        self.use_iou = use_iou
        self.use_boundary = use_boundary
        self.use_adaptive_weighting = use_adaptive_weighting
        self.use_curriculum_learning = use_curriculum_learning
        self.use_dataset_aware_loss = use_dataset_aware_loss
        
        # Loss parameters
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.boundary_kernel_size = boundary_kernel_size
        
        # Loss scaling factors to normalize different loss components
        if loss_scaling_factors is None:
            self.loss_scaling_factors = {
                'ce': 1.0,
                'dice': 1.0,
                'focal': 0.5,  # 提高focal loss的影响
                'iou': 0.3,    # 提高iou loss的影响
                'boundary': 0.2  # 提高boundary loss的影响
            }
        else:
            self.loss_scaling_factors = loss_scaling_factors
        
        # Curriculum learning schedule
        if curriculum_schedule is None:
            self.curriculum_schedule = {
                'focal_start_epoch': 2,
                'iou_start_epoch': 3,
                'boundary_start_epoch': 4,
                'full_weight_epoch': 8
            }
        else:
            self.curriculum_schedule = curriculum_schedule
        
        # Dataset weights for multi-dataset training
        if dataset_weights is None:
            self.dataset_weights = {
                'refcoco': 1.0,
                'refcoco+': 1.0,
                'refcocog': 1.0,
                'ref-zom': 0.8
            }
        else:
            self.dataset_weights = dataset_weights
        
        # Current epoch for curriculum learning
        self.current_epoch = 0
        
        # Adaptive weighting
        if self.use_adaptive_weighting:
            num_losses = 2  # base losses (ce + dice)
            if self.use_focal:
                num_losses += 1
            if self.use_iou:
                num_losses += 1
            if self.use_boundary:
                num_losses += 1
            
            # 更合理的初始权重
            init_weights = [1.0, 1.0]  # ce, dice
            if self.use_focal:
                init_weights.append(0.5)
            if self.use_iou:
                init_weights.append(0.3)
            if self.use_boundary:
                init_weights.append(0.2)
                
            self.adaptive_weighting = AdaptiveLossWeighting(
                num_losses, init_weights, temperature=1.5
            )
    
    def set_epoch(self, epoch):
        """Set current epoch for curriculum learning"""
        self.current_epoch = epoch
    
    def get_curriculum_weights(self):
        """Get curriculum learning weights based on current epoch"""
        if not self.use_curriculum_learning:
            return {
                'focal': 1.0 if self.use_focal else 0.0,
                'iou': 1.0 if self.use_iou else 0.0,
                'boundary': 1.0 if self.use_boundary else 0.0
            }
        
        # Calculate curriculum weights
        focal_weight = 0.0
        iou_weight = 0.0
        boundary_weight = 0.0
        
        if self.use_focal:
            if self.current_epoch >= self.curriculum_schedule['focal_start_epoch']:
                progress = min(1.0, (self.current_epoch - self.curriculum_schedule['focal_start_epoch']) / 
                             (self.curriculum_schedule['full_weight_epoch'] - self.curriculum_schedule['focal_start_epoch']))
                focal_weight = progress
        
        if self.use_iou:
            if self.current_epoch >= self.curriculum_schedule['iou_start_epoch']:
                progress = min(1.0, (self.current_epoch - self.curriculum_schedule['iou_start_epoch']) / 
                             (self.curriculum_schedule['full_weight_epoch'] - self.curriculum_schedule['iou_start_epoch']))
                iou_weight = progress
        
        if self.use_boundary:
            if self.current_epoch >= self.curriculum_schedule['boundary_start_epoch']:
                progress = min(1.0, (self.current_epoch - self.curriculum_schedule['boundary_start_epoch']) / 
                             (self.curriculum_schedule['full_weight_epoch'] - self.curriculum_schedule['boundary_start_epoch']))
                boundary_weight = progress
        
        return {
            'focal': focal_weight,
            'iou': iou_weight,
            'boundary': boundary_weight
        }
    
    def get_dataset_weight(self, dataset_name):
        """Get dataset-specific weight"""
        if not self.use_dataset_aware_loss:
            return 1.0
        return self.dataset_weights.get(dataset_name, 1.0)
    
    def forward(self, pred, targets, aux_pred=None, aux_target=None, dataset_name=None):
        """
        Args:
            pred: [BxHxW] - predicted masks
            targets: [BxHxW] - target masks
            aux_pred: [BxHxW] - auxiliary predictions (optional)
            aux_target: [BxHxW] - auxiliary targets (optional)
            dataset_name: str - name of the dataset for dataset-aware weighting
        """
        loss_dict = dict()
        target = targets.to(pred.dtype)
        
        # Get curriculum weights
        curriculum_weights = self.get_curriculum_weights()
        
        # Get dataset weight
        dataset_weight = self.get_dataset_weight(dataset_name) if dataset_name else 1.0
        
        # Main losses
        main_losses = self.loss_masks(pred, target, curriculum_weights)
        loss_dict.update(main_losses)
        
        # Apply dataset weight to all losses
        for key in loss_dict:
            if key != 'total_loss':
                loss_dict[key] = loss_dict[key] * dataset_weight
        
        # Auxiliary losses
        if aux_pred is not None:
            if aux_target is None:
                aux_target = target
            aux_losses = self.loss_masks(aux_pred, aux_target, curriculum_weights)
            aux_losses = {k + "_aux": v * dataset_weight for k, v in aux_losses.items()}
            loss_dict.update(aux_losses)
        
        # Compute total loss (support adaptive weighting with scaling factors)
        if self.use_adaptive_weighting:
            # main branch adaptive total
            main_loss_dict = {k: v for k, v in loss_dict.items() if k.startswith('loss_') and not k.endswith('_aux')}
            main_total, main_weights = self._compute_adaptive_total_loss(main_loss_dict)
            total_loss = main_total
            # aux branch adaptive total (if exists)
            aux_keys = [k for k in loss_dict.keys() if k.endswith('_aux')]
            if len(aux_keys) > 0:
                aux_base = {}
                for k in aux_keys:
                    base_k = k.replace('_aux', '')
                    aux_base[base_k] = loss_dict[k]
                aux_total, aux_weights = self._compute_adaptive_total_loss(aux_base)
                total_loss = total_loss + aux_total
                # store aux weights for monitoring (optional)
                loss_dict['adaptive_weights_aux'] = aux_weights.detach()
            # store main weights for monitoring
            loss_dict['adaptive_weights_main'] = main_weights.detach()
            loss_dict['total_loss'] = total_loss
        else:
            # Fallback: simple sum of all components
            total_loss = 0
            for k, v in loss_dict.items():
                if k != 'total_loss':
                    total_loss += v
            loss_dict['total_loss'] = total_loss
        
        # Add curriculum and dataset info for monitoring
        loss_dict['curriculum_weights'] = curriculum_weights
        loss_dict['dataset_weight'] = dataset_weight
        
        return loss_dict
    
    def _compute_adaptive_total_loss(self, loss_dict):
        """Compute total loss with adaptive weighting"""
        losses = []
        loss_names = []
        
        # Base losses
        losses.append(loss_dict['loss_mask'])  # CE loss
        loss_names.append('ce')
        losses.append(loss_dict['loss_dice'])  # Dice loss
        loss_names.append('dice')
        
        # Additional losses
        if self.use_focal and 'loss_focal' in loss_dict:
            losses.append(loss_dict['loss_focal'])
            loss_names.append('focal')
        if self.use_iou and 'loss_iou' in loss_dict:
            losses.append(loss_dict['loss_iou'])
            loss_names.append('iou')
        if self.use_boundary and 'loss_boundary' in loss_dict:
            losses.append(loss_dict['loss_boundary'])
            loss_names.append('boundary')
        
        # 应用scaling factors
        scaled_losses = []
        for i, loss in enumerate(losses):
            scale_factor = self.loss_scaling_factors.get(loss_names[i], 1.0)
            scaled_losses.append(loss * scale_factor)
        
        # 计算自适应权重
        weighted_loss, weights = self.adaptive_weighting(scaled_losses)
        
        return weighted_loss, weights
    
    def loss_masks(self, src_masks: torch.Tensor, target_masks: torch.Tensor, curriculum_weights=None) -> dict:
        """Compute all mask losses with curriculum learning support"""
        # Prepare masks
        src_masks_4d = src_masks[:, None]  # [B, 1, H, W]
        target_masks_4d = target_masks[:, None]  # [B, 1, H, W]
        
        # Point sampling (same as original)
        with torch.no_grad():
            point_coords = get_uncertain_point_coords_with_randomness(
                src_masks_4d,
                lambda logits: calculate_uncertainty(logits),
                self.num_points,
                self.oversample_ratio,
                self.importance_sample_ratio,
            )
            point_labels = point_sample(
                target_masks_4d,
                point_coords,
                align_corners=False,
            ).squeeze(1)
        
        point_logits = point_sample(
            src_masks_4d,
            point_coords,
            align_corners=False,
        ).squeeze(1)
        
        # Compute losses
        losses = {}
        
        # Base losses (always computed)
        losses["loss_mask"] = sigmoid_ce_loss(point_logits, point_labels)
        losses["loss_dice"] = dice_loss(point_logits, point_labels)
        
        # Additional losses with curriculum learning
        if self.use_focal and (curriculum_weights is None or curriculum_weights.get('focal', 1.0) > 0):
            focal_loss_value = focal_loss(
                point_logits, point_labels, 
                alpha=self.focal_alpha, gamma=self.focal_gamma
            )
            if curriculum_weights is not None:
                focal_loss_value = focal_loss_value * curriculum_weights.get('focal', 1.0)
            losses["loss_focal"] = focal_loss_value
        
        if self.use_iou and (curriculum_weights is None or curriculum_weights.get('iou', 1.0) > 0):
            iou_loss_value = iou_loss(point_logits, point_labels)
            if curriculum_weights is not None:
                iou_loss_value = iou_loss_value * curriculum_weights.get('iou', 1.0)
            losses["loss_iou"] = iou_loss_value
        
        if self.use_boundary and (curriculum_weights is None or curriculum_weights.get('boundary', 1.0) > 0):
            # 确保边界损失使用相同尺寸的掩码
            # 将src_masks调整到与target_masks相同的尺寸
            if src_masks.shape != target_masks.shape:
                src_masks_resized = F.interpolate(
                    src_masks.unsqueeze(1), 
                    size=target_masks.shape[-2:], 
                    mode='bilinear', 
                    align_corners=False
                ).squeeze(1)
            else:
                src_masks_resized = src_masks
                
            boundary_loss_value = boundary_loss(
                src_masks_resized, target_masks, 
                kernel_size=self.boundary_kernel_size
            )
            if curriculum_weights is not None:
                boundary_loss_value = boundary_loss_value * curriculum_weights.get('boundary', 1.0)
            losses["loss_boundary"] = boundary_loss_value
        
        return losses

# JIT compiled versions for better performance
#focal_loss_jit = torch.jit.script(focal_loss)
#iou_loss_jit = torch.jit.script(iou_loss)
#boundary_loss_jit = torch.jit.script(boundary_loss)


# Enhanced criterion dictionary
enhanced_criterion_dict = {
    'mask': EnhancedSegMaskLoss,
    'enhanced_mask': EnhancedSegMaskLoss,
}
