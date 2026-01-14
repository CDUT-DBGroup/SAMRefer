import argparse
import yaml
import os
from types import SimpleNamespace

def get_args():
    # Step 1: 预解析 config 文件路径
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=False, help='Path to config file', default='configs/main_refersam_bert.yaml')
    # parser.add_argument('--deepspeed_config', type=str, default='configs/ds_config.json', help='Path to DeepSpeed config file')
    args_config_only, remaining_argv = parser.parse_known_args()

    # Step 2: 加载 YAML 配置文件
    with open(args_config_only.config, 'r') as f:
        config_dict = yaml.safe_load(f)

    for key, value in config_dict.items():
        arg_type = type(value) if value is not None else str
        parser.add_argument(f'--{key}', type=arg_type, default=value)

    # Add DeepSpeed specific parameters
    parser.add_argument('--deepspeed_config', type=str, default='ds_config.json', help='Path to DeepSpeed config file')
    parser.add_argument('--local_rank', type=int, default=-1, help='Local rank for distributed training')
    # parser.add_argument('--resume', type=str, help='Resume training from checkpoint')
    parser.add_argument('--use_enhanced_loss', action='store_true', 
                       help='Use enhanced loss functions (Focal, IoU, Boundary)')
    parser.add_argument('--loss_config_path', type=str, 
                       default='configs/enhanced_loss_config.yaml',
                       help='Path to loss configuration file')
    parser.add_argument('--loss_ablation', type=str, choices=['all', 'focal', 'iou', 'boundary', 'adaptive'], 
                       default='all', help='Loss ablation study mode')
    parser.add_argument('--use_negative_masks', action='store_true',
                       help='Use negative mask selection and adaptive fusion (BMS method)')
    parser.add_argument('--use_best_sentence', action='store_true',
                       help='Use best sentence for validation')
    parser.add_argument('--sentence_aggregation', type=str, choices=['best', 'mean', 'mean_iou', 'median'],
                       default='mean', help='Sentence aggregation method for validation')
    # 消融实验参数
    parser.add_argument('--use_lang_attention', type=lambda x: (str(x).lower() == 'true'), default=True,
                       help='Use language attention mechanism for text feature aggregation (for ablation study)')
    parser.add_argument('--use_csaf', type=lambda x: (str(x).lower() == 'true'), default=True,
                       help='Use Cross-Scale Attention Fusion (CSAF) module (for ablation study, controls ViT-C3 fusion etc.)')
    parser.add_argument('--use_multi_scale_fusion', type=lambda x: (str(x).lower() == 'true') if x is not None else None, default=None,
                       help='Use MultiScaleFusion module for c2, c3, c4 cross-scale attention fusion (for ablation study). If None, uses use_csaf value.')
    parser.add_argument('--use_enhanced_c1c2_fusion', type=lambda x: (str(x).lower() == 'true') if x is not None else None, default=None,
                       help='Use EnhancedC1C2Fusion module for spatial attention fusion of c1 and c2 (for ablation study). If None, uses use_csaf value.')
    parser.add_argument('--use_lang_fusion_weights', type=lambda x: (str(x).lower() == 'true') if x is not None else None, default=None,
                       help='Use learnable fusion weights to combine base token and attention-aggregated text features (for ablation study). If None, uses use_lang_attention value.')
    # Step 4: 解析所有参数（命令行优先）
    args = parser.parse_args()
    return args
