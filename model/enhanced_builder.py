import torch
import torch.nn as nn
from transformers import BertModel
from transformers import CLIPTextModel
import yaml
import os
import glob
import json
import socket

from .models import *
from .segment_anything import sam_model_registry
from .criterion import SegMaskLoss, criterion_dict
from .enhanced_criterion import EnhancedSegMaskLoss, enhanced_criterion_dict


def _find_available_port(start_port=29500, max_attempts=100):
    """
    查找一个可用的端口
    
    Args:
        start_port: 起始端口号
        max_attempts: 最大尝试次数
    
    Returns:
        可用的端口号，如果找不到则返回 None
    """
    for port in range(start_port, start_port + max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind(('', port))
                return port
        except OSError:
            continue
    return None


def _is_deepspeed_checkpoint(checkpoint_path):
    """
    检测检查点路径是否是 DeepSpeed 格式
    """
    if not os.path.exists(checkpoint_path):
        return False
    
    if os.path.isdir(checkpoint_path):
        subdirs = glob.glob(os.path.join(checkpoint_path, "global_step*"))
        if subdirs:
            latest_subdir = max(subdirs, key=os.path.getmtime)
            if os.path.exists(os.path.join(latest_subdir, "mp_rank_00")) or \
               glob.glob(os.path.join(latest_subdir, "*.pt")):
                return True
        if os.path.exists(os.path.join(checkpoint_path, "mp_rank_00")):
            return True
    return False


def _load_pytorch_checkpoint(model, checkpoint_path, device):
    """
    加载 PyTorch 格式的 checkpoint
    """
    if os.path.isdir(checkpoint_path):
        # 查找所有 global_step 子目录
        subdirs = glob.glob(os.path.join(checkpoint_path, "global_step*"))
        if subdirs:
            # 取最新的 global_step 子目录
            latest_subdir = max(subdirs, key=os.path.getmtime)
            print(f"Loading model weights from {latest_subdir}")
            checkpoint = torch.load(latest_subdir, map_location=device)
        else:
            print(f"Loading model weights from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device)
    else:
        print(f"Loading model weights from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 处理不同的 checkpoint 格式
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            # 可能是直接的 state_dict
            model.load_state_dict(checkpoint)
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    return model


def _segm_refersam_enhanced(pretrained, args, criterion):
    """
    Enhanced version of ReferSAM with improved loss functions
    支持 DeepSpeed 和 PyTorch 格式的 checkpoint 加载
    
    Returns:
        model: 模型对象
        is_deepspeed: 如果是 DeepSpeed checkpoint，返回 True，否则返回 False
        checkpoint_info: 如果是 DeepSpeed checkpoint，返回相关信息字典，否则返回 None
    """
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
        'use_lang_attention': getattr(args, 'use_lang_attention', True),  # 消融实验：默认使用文本注意力
        'use_csaf': getattr(args, 'use_csaf', True),  # 消融实验：默认使用Cross-Scale Attention Fusion (CSAF)模块
    }
    model = ReferSAM(sam_model, text_model, args, criterion=criterion, **adapter_configs)
    
    # 处理 checkpoint 加载
    is_deepspeed = False
    checkpoint_info = None
    
    if pretrained is True and hasattr(args, 'pre_train_path') and args.pre_train_path:
        checkpoint_path = args.pre_train_path
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        # 检测 checkpoint 格式
        if _is_deepspeed_checkpoint(checkpoint_path):
            is_deepspeed = True
            print(f"Detected DeepSpeed checkpoint format: {checkpoint_path}")
            
            # 读取 DeepSpeed 配置
            ds_config = getattr(args, 'deepspeed_config', 'configs/ds_config.json')
            if not os.path.exists(ds_config):
                ds_config = 'ds_config.json'
            
            use_fp16 = False
            use_bf16 = False
            if os.path.exists(ds_config):
                with open(ds_config) as f:
                    ds_conf = json.load(f)
                use_fp16 = ds_conf.get("fp16", {}).get("enabled", False)
                use_bf16 = ds_conf.get("bf16", {}).get("enabled", False)
            
            checkpoint_info = {
                'checkpoint_path': checkpoint_path,
                'ds_config': ds_config if os.path.exists(ds_config) else None,
                'use_fp16': use_fp16,
                'use_bf16': use_bf16
            }
            print(f"DeepSpeed checkpoint info: fp16={use_fp16}, bf16={use_bf16}")
        else:
            # PyTorch 格式的 checkpoint，直接加载
            model = _load_pytorch_checkpoint(model, checkpoint_path, device)
    
    # 为了保持向后兼容，如果 pretrained=True 但不是 DeepSpeed，直接返回模型
    # 如果是 DeepSpeed，返回模型和相关信息
    if is_deepspeed:
        return model, is_deepspeed, checkpoint_info
    else:
        return model


def refersam_enhanced(pretrained=None, args=None, loss_config_path=None):
    """
    Enhanced ReferSAM with improved loss functions
    
    Args:
        pretrained: Whether to load pretrained weights
        args: Model arguments
        loss_config_path: Path to loss configuration file
    
    Returns:
        如果加载的是 DeepSpeed checkpoint，返回 (model, is_deepspeed, checkpoint_info)
        否则返回 model
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
        use_curriculum_learning=loss_config.get('use_curriculum_learning', True),
        use_dataset_aware_loss=loss_config.get('use_dataset_aware_loss', True),
        focal_alpha=loss_config.get('focal_alpha', 0.25),
        focal_gamma=loss_config.get('focal_gamma', 2.0),
        boundary_kernel_size=loss_config.get('boundary_kernel_size', 3),
        loss_scaling_factors=loss_config.get('loss_scaling_factors', None),
        curriculum_schedule=loss_config.get('curriculum_schedule', None),
        dataset_weights=loss_config.get('dataset_weights', None)
    )
    
    result = _segm_refersam_enhanced(pretrained, args, criterion)
    return result


def refersam_original(pretrained=None, args=None):
    """
    Original ReferSAM with basic loss functions
    
    Returns:
        如果加载的是 DeepSpeed checkpoint，返回 (model, is_deepspeed, checkpoint_info)
        否则返回 model
    """
    criterion = criterion_dict['mask']()
    result = _segm_refersam_enhanced(pretrained, args, criterion)
    return result


# Backward compatibility
def refersam(pretrained=None, args=None):
    """
    Default ReferSAM - can be switched between original and enhanced
    
    Returns:
        如果加载的是 DeepSpeed checkpoint，返回 (model, is_deepspeed, checkpoint_info)
        否则返回 model
    """
    # Check if enhanced loss is enabled via args
    if hasattr(args, 'use_enhanced_loss') and args.use_enhanced_loss:
        loss_config_path = getattr(args, 'loss_config_path', None)
        return refersam_enhanced(pretrained, args, loss_config_path)
    else:
        return refersam_original(pretrained, args)


def load_model_with_checkpoint(model_func, pretrained, args, loss_config_path=None, device=None):
    """
    统一的模型加载函数，支持 DeepSpeed 和 PyTorch checkpoint
    
    Args:
        model_func: 模型创建函数 (refersam, refersam_enhanced, refersam_original)
        pretrained: 是否加载预训练权重
        args: 模型参数
        loss_config_path: 损失配置文件路径（仅用于 enhanced 版本）
        device: 设备，如果为 None 则自动选择
    
    Returns:
        eval_model: 用于评估的模型（如果是 DeepSpeed，返回 model_engine.module，否则返回 model）
        use_fp16: 是否使用 fp16
        use_bf16: 是否使用 bf16
        model_engine: DeepSpeed 引擎（如果是 DeepSpeed checkpoint），否则为 None
    """
    if device is None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # 创建模型
    result = model_func(pretrained=pretrained, args=args)
    
    # 检查返回值格式
    if isinstance(result, tuple) and len(result) == 3:
        # DeepSpeed checkpoint - 只在需要时才导入和初始化 DeepSpeed
        model, is_deepspeed, checkpoint_info = result
        assert is_deepspeed, "Expected DeepSpeed checkpoint but got False"
        
        model = model.to(device)
        
        # 设置环境变量以避免 MPI 检测问题（单机单卡测试时）
        # 如果未设置分布式相关环境变量，设置为单机单卡模式
        if 'RANK' not in os.environ:
            os.environ['RANK'] = '0'
        if 'LOCAL_RANK' not in os.environ:
            os.environ['LOCAL_RANK'] = '0'
        if 'WORLD_SIZE' not in os.environ:
            os.environ['WORLD_SIZE'] = '1'
        if 'MASTER_ADDR' not in os.environ:
            os.environ['MASTER_ADDR'] = 'localhost'
        
        # 设置 MASTER_PORT：如果环境变量已设置则使用，否则自动查找可用端口
        if 'MASTER_PORT' not in os.environ:
            # 首先尝试默认端口 29500
            default_port = 29500
            available_port = _find_available_port(start_port=default_port, max_attempts=100)
            if available_port is None:
                # 如果默认端口范围都不可用，尝试从 20000 开始
                available_port = _find_available_port(start_port=20000, max_attempts=100)
            
            if available_port is not None:
                os.environ['MASTER_PORT'] = str(available_port)
                print(f"Using available port: {available_port}")
            else:
                # 如果仍然找不到可用端口，使用默认值（可能会失败，但至少尝试）
                os.environ['MASTER_PORT'] = str(default_port)
                print(f"Warning: Could not find available port, using default: {default_port}")
        else:
            # 如果用户已经设置了端口，验证它是否可用
            try:
                user_port = int(os.environ['MASTER_PORT'])
                test_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                test_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                try:
                    test_socket.bind(('', user_port))
                    test_socket.close()
                    print(f"Using user-specified port: {user_port}")
                except OSError:
                    # 用户指定的端口不可用，尝试查找替代端口
                    print(f"Warning: User-specified port {user_port} is not available, finding alternative...")
                    available_port = _find_available_port(start_port=20000, max_attempts=100)
                    if available_port is not None:
                        os.environ['MASTER_PORT'] = str(available_port)
                        print(f"Using alternative port: {available_port}")
                    else:
                        print(f"Warning: Could not find alternative port, keeping user-specified: {user_port}")
            except (ValueError, OSError) as e:
                print(f"Warning: Invalid or unavailable port {os.environ['MASTER_PORT']}: {e}")
        
        # 禁用 MPI 检测，避免 MPI 初始化错误
        os.environ['PMI_RANK'] = '0'
        os.environ['PMI_SIZE'] = '1'
        # 禁用 DeepSpeed 的自动 MPI 检测
        os.environ['DEEPSPEED_AUTOTUNE'] = '0'
        
        import deepspeed
        
        # 尝试初始化分布式环境（如果尚未初始化）
        try:
            deepspeed.init_distributed()
        except (RuntimeError, ValueError):
            # 如果已经初始化或初始化失败，继续执行
            pass
        
        # 初始化 DeepSpeed 引擎
        if hasattr(model, 'params_to_optimize'):
            param_groups = model.params_to_optimize()
        else:
            param_groups = model.parameters()
        
        # 使用单机模式初始化 DeepSpeed
        model_engine, optimizer, _, _ = deepspeed.initialize(
            model=model,
            model_parameters=param_groups,
            config=checkpoint_info['ds_config'] if checkpoint_info['ds_config'] else None
        )
        
        # 加载 DeepSpeed checkpoint
        checkpoint_path = checkpoint_info['checkpoint_path']
        if os.path.isdir(checkpoint_path):
            subdirs = glob.glob(os.path.join(checkpoint_path, "global_step*"))
            if subdirs:
                latest_subdir = max(subdirs, key=os.path.getmtime)
                print(f"Loading DeepSpeed checkpoint from {checkpoint_path}, latest: {os.path.basename(latest_subdir)}")
            else:
                print(f"Loading DeepSpeed checkpoint from {checkpoint_path}")
        else:
            print(f"Loading DeepSpeed checkpoint from {checkpoint_path}")
        
        try:
            _, client_state = model_engine.load_checkpoint(checkpoint_path)
            print(f"Successfully loaded DeepSpeed checkpoint")
            if client_state:
                print(f"Client state: {client_state}")
        except RuntimeError as e:
            error_msg = str(e)
            if "position_ids" in error_msg:
                print(f"Warning: Encountered position_ids in checkpoint (likely due to transformers version mismatch)")
                print("Attempting to load checkpoint with filtered state_dict...")
                
                # 使用 DeepSpeed 工具提取 state_dict 并过滤 position_ids
                from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
                try:
                    state_dict = get_fp32_state_dict_from_zero_checkpoint(checkpoint_path)
                    # 只过滤掉 position_ids 相关的键
                    filtered_state_dict = {k: v for k, v in state_dict.items() if 'position_ids' not in k}
                    filtered_count = len(state_dict) - len(filtered_state_dict)
                    if filtered_count > 0:
                        print(f"Filtered out {filtered_count} position_ids key(s)")
                    
                    # 直接加载到模型（使用 strict=False 允许部分键不匹配）
                    missing_keys, unexpected_keys = model_engine.module.load_state_dict(filtered_state_dict, strict=False)
                    if missing_keys:
                        print(f"Missing keys (will use default values): {len(missing_keys)} keys")
                    if unexpected_keys:
                        print(f"Unexpected keys (ignored): {len(unexpected_keys)} keys")
                    print("Successfully loaded DeepSpeed checkpoint with filtered state_dict")
                    client_state = {}
                except Exception as e2:
                    raise RuntimeError(f"Failed to load checkpoint even with filtering: {e2}")
            else:
                # 其他类型的错误，直接抛出
                raise
        
        model_engine.eval()
        return model_engine.module, checkpoint_info['use_fp16'], checkpoint_info['use_bf16'], model_engine
    else:
        # PyTorch checkpoint 或未加载 checkpoint
        model = result
        if not hasattr(model, 'device') or next(model.parameters()).device != device:
            model = model.to(device)
        if pretrained:
            model.eval()
        return model, False, False, None


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
