import torch
import torch.nn as nn
from transformers import BertModel
from transformers import CLIPTextModel
import yaml
import os

from .models import *
from .segment_anything import sam_model_registry
from .criterion import SegMaskLoss, criterion_dict
from .enhanced_criterion import EnhancedSegMaskLoss, enhanced_criterion_dict


def _segm_refersam_enhanced(pretrained, args, criterion):
    """Enhanced version of ReferSAM with improved loss functions"""
    torch_dtype = torch.float32
    if args.precision == "bf16":
        torch_dtype = torch.bfloat16
    elif args.precision == "fp16":
        torch_dtype = torch.half

    if args.clip_path:
        text_model = CLIPTextModel.from_pretrained(args.clip_path, torch_dtype=torch_dtype)
    else:
        text_model = BertModel.from_pretrained(args.ck_bert, torch_dtype=torch_dtype)

    sam_configs = {
        "vit_h": {"interaction_indexes": [[0, 7], [8, 15], [16, 23], [24, 31]], "num_heads":16, "vl_dim":1280},
        "vit_l": {"interaction_indexes": [[0, 5], [6, 11], [12, 17], [18, 23]], "num_heads":16, "vl_dim":1024},
        "vit_b": {"interaction_indexes": [[0, 2], [3, 5], [6, 8], [9, 11]], "num_heads":12, "vl_dim":768},
    }
    sam_model = sam_model_registry[args.sam_type](checkpoint=args.checkpoint)
    
    adapter_configs = {
        'drop_path_rate':0.0,
        'dropout':0.0,
        'conv_inplane':64,
        'n_points':4,
        'deform_num_heads':sam_configs[args.sam_type]["num_heads"],
        'deform_ratio':0.5,
        'cffn_ratio':2.0,
        'add_vit_feature':False,
        'interaction_indexes':sam_configs[args.sam_type]["interaction_indexes"],
        'with_cp': False,
        'init_values': 1e-6,
        'vl_dim': sam_configs[args.sam_type]["vl_dim"],
        'num_prompts': [16, 4],
        'num_extra_layers': 2,
        "num_prompt_layers": 2,
    }
    model = ReferSAM(sam_model, text_model, args, criterion=criterion, **adapter_configs)
    
    if pretrained is not None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(f"Loading model weights from {args.pre_train_path}")
        checkpoint = torch.load(args.pre_train_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()
    return model


def refersam_enhanced(pretrained=None, args=None, loss_config_path=None):
    """
    Enhanced ReferSAM with improved loss functions
    
    Args:
        pretrained: Whether to load pretrained weights
        args: Model arguments
        loss_config_path: Path to loss configuration file
    """
    # Load loss configuration
    if loss_config_path and os.path.exists(loss_config_path):
        with open(loss_config_path, 'r', encoding='utf-8') as f:
            loss_config = yaml.safe_load(f)
    else:
        # Default configuration
        loss_config = {
            'use_focal': True,
            'use_iou': True,
            'use_boundary': True,
            'use_adaptive_weighting': True,
            'focal_alpha': 0.25,
            'focal_gamma': 2.0,
            'boundary_kernel_size': 3,
            'num_points': 112*112,
            'oversample_ratio': 3.0,
            'importance_sample_ratio': 0.75
        }
    
    # Create enhanced criterion
    criterion = EnhancedSegMaskLoss(
        num_points=loss_config.get('num_points', 112*112),
        oversample_ratio=loss_config.get('oversample_ratio', 3.0),
        importance_sample_ratio=loss_config.get('importance_sample_ratio', 0.75),
        use_focal=loss_config.get('use_focal', True),
        use_iou=loss_config.get('use_iou', True),
        use_boundary=loss_config.get('use_boundary', True),
        use_adaptive_weighting=loss_config.get('use_adaptive_weighting', True),
        focal_alpha=loss_config.get('focal_alpha', 0.25),
        focal_gamma=loss_config.get('focal_gamma', 2.0),
        boundary_kernel_size=loss_config.get('boundary_kernel_size', 3),
        loss_scaling_factors=loss_config.get('loss_scaling_factors', None)
    )
    
    return _segm_refersam_enhanced(pretrained, args, criterion)


def refersam_original(pretrained=None, args=None):
    """Original ReferSAM with basic loss functions"""
    criterion = criterion_dict['mask']()
    return _segm_refersam_enhanced(pretrained, args, criterion)


# Backward compatibility
def refersam(pretrained=None, args=None):
    """Default ReferSAM - can be switched between original and enhanced"""
    # Check if enhanced loss is enabled via args
    if hasattr(args, 'use_enhanced_loss') and args.use_enhanced_loss:
        loss_config_path = getattr(args, 'loss_config_path', None)
        return refersam_enhanced(pretrained, args, loss_config_path)
    else:
        return refersam_original(pretrained, args)


def get_model_enhanced(args, loss_config_path=None):
    """Enhanced version of get_model with improved loss functions"""
    sam = sam_model_registry[args.sam_type](checkpoint=args.checkpoint)
    text_model = BertModel.from_pretrained(args.ck_bert)
    
    # Load loss configuration
    if loss_config_path and os.path.exists(loss_config_path):
        with open(loss_config_path, 'r', encoding='utf-8') as f:
            loss_config = yaml.safe_load(f)
    else:
        loss_config = {
            'use_focal': True,
            'use_iou': True,
            'use_boundary': True,
            'use_adaptive_weighting': True,
            'focal_alpha': 0.25,
            'focal_gamma': 2.0,
            'boundary_kernel_size': 3,
            'num_points': 112*112,
            'oversample_ratio': 3.0,
            'importance_sample_ratio': 0.75
        }
    
    criterion = EnhancedSegMaskLoss(
        num_points=loss_config.get('num_points', 112*112),
        oversample_ratio=loss_config.get('oversample_ratio', 3.0),
        importance_sample_ratio=loss_config.get('importance_sample_ratio', 0.75),
        use_focal=loss_config.get('use_focal', True),
        use_iou=loss_config.get('use_iou', True),
        use_boundary=loss_config.get('use_boundary', True),
        use_adaptive_weighting=loss_config.get('use_adaptive_weighting', True),
        focal_alpha=loss_config.get('focal_alpha', 0.25),
        focal_gamma=loss_config.get('focal_gamma', 2.0),
        boundary_kernel_size=loss_config.get('boundary_kernel_size', 3)
    )

    model = ReferSAM(
        sam_model=sam,
        text_encoder=text_model,
        args=args,
        num_classes=1,
        criterion=criterion
    )
    return model
