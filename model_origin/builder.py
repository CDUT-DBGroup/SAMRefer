import torch
import torch.nn as nn
import os
import glob
from transformers import BertModel
from transformers import CLIPTextModel

from .models import *
from .segment_anything import sam_model_registry
from .criterion import SegMaskLoss, criterion_dict


def _segm_refersam(pretrained, args, criterion):
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
    # sam_model = 
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
        
        # 检查路径是文件还是目录（DeepSpeed检查点）
        if os.path.isdir(args.pre_train_path):
            # DeepSpeed检查点目录，使用DeepSpeed工具加载
            try:
                from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
                print(f"Detected DeepSpeed checkpoint directory, converting to fp32 state_dict...")
                state_dict = get_fp32_state_dict_from_zero_checkpoint(args.pre_train_path)
                # 处理state_dict的键名，可能需要移除'module.'前缀
                if any(key.startswith('module.') for key in state_dict.keys()):
                    state_dict = {key[7:] if key.startswith('module.') else key: value 
                                 for key, value in state_dict.items()}
                model.load_state_dict(state_dict, strict=False)
            except Exception as e:
                print(f"Failed to load DeepSpeed checkpoint: {e}")
                print("Trying alternative method: loading from mp_rank_00_model_states.pt...")
                # 备用方法：尝试从global_step子目录加载
                subdirs = glob.glob(os.path.join(args.pre_train_path, "global_step*"))
                if subdirs:
                    latest_subdir = max(subdirs, key=os.path.getmtime)
                    model_state_file = os.path.join(latest_subdir, "mp_rank_00_model_states.pt")
                    if os.path.exists(model_state_file):
                        checkpoint = torch.load(model_state_file, map_location=device)
                        if 'module' in checkpoint:
                            model.load_state_dict(checkpoint['module'], strict=False)
                        else:
                            model.load_state_dict(checkpoint, strict=False)
                    else:
                        raise FileNotFoundError(f"Model state file not found: {model_state_file}")
                else:
                    raise ValueError(f"Could not find global_step subdirectory in {args.pre_train_path}")
        else:
            # 普通PyTorch检查点文件
            checkpoint = torch.load(args.pre_train_path, map_location=device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            else:
                # 如果checkpoint直接是state_dict
                model.load_state_dict(checkpoint, strict=False)
        
        model = model.to(device)
        model.eval()  # Set model to evaluation mode
    return model


def refersam(pretrained=None, args=None):
    criterion = criterion_dict['mask']()
    return _segm_refersam(pretrained, args, criterion)


# 获取的是原始的模型
def get_model(args):
    sam = sam_model_registry[args.sam_type](checkpoint=args.checkpoint)
    text_model = BertModel.from_pretrained(args.ck_bert)
    criterion = SegMaskLoss(num_points=112*112, oversample_ratio=3.0, importance_sample_ratio=0.75)

    model = ReferSAM(
        sam_model=sam,
        text_encoder=text_model,
        args=args,
        num_classes=1,
        criterion=criterion
    )
    return model