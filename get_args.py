import argparse
import yaml
import os
from types import SimpleNamespace

def get_args():
    # Step 1: 预解析 config 文件路径
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=False, help='Path to config file', default='configs/new_refersam_bert.yaml')
    args_config_only, remaining_argv = parser.parse_known_args()

    # Step 2: 加载 YAML 配置文件
    with open(args_config_only.config, 'r') as f:
        config_dict = yaml.safe_load(f)

    # Step 3: 用 argparse 定义所有可能的可覆盖参数
    parser = argparse.ArgumentParser()
    for key, value in config_dict.items():
        arg_type = type(value) if value is not None else str
        parser.add_argument(f'--{key}', type=arg_type, default=value)

    # Add resume training parameter
    parser.add_argument('--resume', action='store_true', help='Resume training from checkpoint')

    # Step 4: 解析所有参数（命令行优先）
    args = parser.parse_args()
    return args
