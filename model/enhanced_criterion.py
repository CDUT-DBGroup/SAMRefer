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


def boundary_loss(inputs, targets, kernel_size=3):
    """
    Boundary Loss for improving edge quality
    Args:
        inputs: A float tensor of shape [B, H, W] (logits)
        targets: A float tensor of shape [B, H, W]
        kernel_size: Size of the kernel for boundary detection
    """
    inputs = torch.sigmoid(inputs)
    
    # Convert to binary masks
    pred_binary = (inputs > 0.5).float()
    target_binary = (targets > 0.5).float()
    
    # Create boundary detection kernel
    kernel = torch.ones(1, 1, kernel_size, kernel_size, device=inputs.device)
    kernel[0, 0, kernel_size//2, kernel_size//2] = 0
    
    # Detect boundaries using morphological operations
    pred_boundary = F.conv2d(pred_binary.unsqueeze(1), kernel, padding=kernel_size//2)
    pred_boundary = (pred_boundary > 0).float().squeeze(1)
    
    target_boundary = F.conv2d(target_binary.unsqueeze(1), kernel, padding=kernel_size//2)
    target_boundary = (target_boundary > 0).float().squeeze(1)
    
    # Compute boundary loss
    boundary_loss = F.binary_cross_entropy(pred_boundary, target_boundary)
    
    return boundary_loss


class AdaptiveLossWeighting(nn.Module):
    """
    Adaptive loss weighting mechanism
    """
    def __init__(self, num_losses=4, init_weights=None):
        super().__init__()
        if init_weights is None:
            init_weights = [1.0, 1.0, 1.0, 1.0]  # [ce, dice, focal, iou, boundary]
        self.log_vars = nn.Parameter(torch.log(torch.tensor(init_weights)))
    
    def forward(self, losses):
        """
        Args:
            losses: List of loss tensors
        Returns:
            weighted_loss: Weighted sum of losses
            weights: Current loss weights
        """
        weights = torch.exp(-self.log_vars)
        weighted_loss = sum(w * loss for w, loss in zip(weights, losses))
        return weighted_loss, weights


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
                 focal_alpha=0.25,
                 focal_gamma=2.0,
                 boundary_kernel_size=3):
        super(EnhancedSegMaskLoss, self).__init__()
        self.num_points = num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio
        
        # Loss function flags
        self.use_focal = use_focal
        self.use_iou = use_iou
        self.use_boundary = use_boundary
        self.use_adaptive_weighting = use_adaptive_weighting
        
        # Loss parameters
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.boundary_kernel_size = boundary_kernel_size
        
        # Adaptive weighting
        if self.use_adaptive_weighting:
            num_losses = 2  # base losses (ce + dice)
            if self.use_focal:
                num_losses += 1
            if self.use_iou:
                num_losses += 1
            if self.use_boundary:
                num_losses += 1
            
            self.adaptive_weighting = AdaptiveLossWeighting(num_losses)
    
    def forward(self, pred, targets, aux_pred=None, aux_target=None):
        """
        Args:
            pred: [BxHxW] - predicted masks
            targets: [BxHxW] - target masks
            aux_pred: [BxHxW] - auxiliary predictions (optional)
            aux_target: [BxHxW] - auxiliary targets (optional)
        """
        loss_dict = dict()
        target = targets.to(pred.dtype)
        
        # Main losses
        main_losses = self.loss_masks(pred, target)
        loss_dict.update(main_losses)
        
        # Auxiliary losses
        if aux_pred is not None:
            if aux_target is None:
                aux_target = target
            aux_losses = self.loss_masks(aux_pred, aux_target)
            aux_losses = {k + "_aux": v for k, v in aux_losses.items()}
            loss_dict.update(aux_losses)
        
        # Compute total loss
        if self.use_adaptive_weighting:
            total_loss, weights = self._compute_adaptive_total_loss(loss_dict)
            loss_dict['total_loss'] = total_loss
            loss_dict['loss_weights'] = weights
        else:
            total_loss = 0
            for k, v in loss_dict.items():
                if k != 'total_loss':
                    total_loss += v
            loss_dict['total_loss'] = total_loss
        
        return loss_dict
    
    def _compute_adaptive_total_loss(self, loss_dict):
        """Compute total loss with adaptive weighting"""
        losses = []
        
        # Base losses
        losses.append(loss_dict['loss_mask'])  # CE loss
        losses.append(loss_dict['loss_dice'])  # Dice loss
        
        # Additional losses
        if self.use_focal and 'loss_focal' in loss_dict:
            losses.append(loss_dict['loss_focal'])
        if self.use_iou and 'loss_iou' in loss_dict:
            losses.append(loss_dict['loss_iou'])
        if self.use_boundary and 'loss_boundary' in loss_dict:
            losses.append(loss_dict['loss_boundary'])
        
        return self.adaptive_weighting(losses)
    
    def loss_masks(self, src_masks: torch.Tensor, target_masks: torch.Tensor) -> dict:
        """Compute all mask losses"""
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
        
        # Additional losses
        if self.use_focal:
            losses["loss_focal"] = focal_loss(
                point_logits, point_labels, 
                alpha=self.focal_alpha, gamma=self.focal_gamma
            )
        
        if self.use_iou:
            losses["loss_iou"] = iou_loss(point_logits, point_labels)
        
        if self.use_boundary:
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
                
            losses["loss_boundary"] = boundary_loss(
                src_masks_resized, target_masks, 
                kernel_size=self.boundary_kernel_size
            )
        
        # Clean up
        del src_masks_4d
        del target_masks_4d
        
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
