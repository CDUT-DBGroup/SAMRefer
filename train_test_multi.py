#!/usr/bin/env python3
"""
Enhanced training script for multi-dataset training with improved loss functions
使用增强损失函数的多数据集训练脚本
"""
import filelock
filelock.FileLock = filelock.SoftFileLock
import deepspeed.ops.transformer.inference.triton.matmul_ext as matmul_ext

matmul_ext.FileLock = filelock.SoftFileLock
print(">>> Patched matmul_ext.FileLock -> SoftFileLock")
import os
import sys
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
import numpy as np
from dataset.GRefDataset import GRefDataset
from dataset.Dataset_referit import ReferitDataset
from dataset.ReferDataset import ReferDataset
from dataset.RefzomDataset import ReferzomDataset
from get_args import get_args
from model.enhanced_builder import refersam_enhanced, refersam_original
from validate_bert import evaluate_four_datasets
from validation.evaluation import validate
import logging
import datetime
import random
import torch.nn as nn
import torch.distributed as dist
import deepspeed
from deepspeed.ops.adam import DeepSpeedCPUAdam
import json
import shutil
import argparse
torch.set_num_threads(16)
torch.set_num_interop_threads(16)

# 内存优化设置
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'


def add_enhanced_loss_args(parser):
    """Add enhanced loss specific arguments"""
    parser.add_argument('--use_enhanced_loss', action='store_true', 
                       help='Use enhanced loss functions (Focal, IoU, Boundary)')
    parser.add_argument('--loss_config_path', type=str, 
                       default='configs/enhanced_loss_config.yaml',
                       help='Path to loss configuration file')
    parser.add_argument('--loss_ablation', type=str, choices=['all', 'focal', 'iou', 'boundary', 'adaptive'], 
                       default='all', help='Loss ablation study mode')
    return parser


def set_seed(seed=123456):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logger(output_dir, rank=0):
    if rank != 0:
        return None
    log_file = os.path.join(output_dir, f'enhanced_multi_dataset_training_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)


def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


class MultiDatasetWrapper:
    """Wrapper to add dataset name to samples"""
    def __init__(self, dataset, dataset_name):
        self.dataset = dataset
        self.dataset_name = dataset_name
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        sample, target = self.dataset[idx]
        # Add dataset name to sample
        if isinstance(sample, dict):
            sample['dataset_name'] = self.dataset_name
        return sample, target


def main():
    # 设置分布式超时，防止卡死（30分钟超时）
    os.environ['NCCL_TIMEOUT'] = '1800'  # 30分钟
    os.environ['GLOO_TIMEOUT_SECONDS'] = '1800'  # 30分钟
    # 初始化 DeepSpeed
    deepspeed.init_distributed()
    
    # 获取原始参数
    args = get_args()
    
    # 确保 config 是文件路径字符串
    if hasattr(args, 'deepspeed_config') and args.deepspeed_config:
        ds_config = args.deepspeed_config
    else:
        ds_config = 'ds_config.json'

    with open(ds_config) as f:
        ds_conf = json.load(f)
    use_fp16 = ds_conf.get("fp16", {}).get("enabled", False)
    use_bf16 = ds_conf.get("bf16", {}).get("enabled", False)

    assert os.path.exists(ds_config), f"DeepSpeed config 文件不存在: {ds_config}"
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logger(args.output_dir, rank)
    
    if logger:
        logger.info(f"Starting Enhanced Multi-Dataset DeepSpeed training: rank {rank}, world_size {world_size}")
        logger.info(f"Enhanced loss enabled: {getattr(args, 'use_enhanced_loss', True)}")
        logger.info(f"Loss config path: {getattr(args, 'loss_config_path', 'None')}")
        logger.info(f"Loss ablation mode: {getattr(args, 'loss_ablation', 'all')}")

    # 创建模型
    if logger:
        logger.info("Creating Enhanced ReferSAM model...")
    
    if getattr(args, 'use_enhanced_loss', False):
        model = refersam_enhanced(
            pretrained=None, 
            args=args, 
            loss_config_path=getattr(args, 'loss_config_path', None)
        )
        if logger:
            logger.info("Using Enhanced Loss Functions: Focal + IoU + Boundary + Adaptive Weighting + Curriculum Learning")
    else:
        model = refersam_original(pretrained=None, args=args)
        if logger:
            logger.info("Using Original Loss Functions: CE + Dice")
    
    model = model.to(device)
    
    # 根据DeepSpeed配置设置模型数据类型
    if use_fp16:
        if logger:
            logger.info("Converting model to fp16 for DeepSpeed")
        model = model.half()
    elif use_bf16:
        if logger:
            logger.info("Converting model to bf16 for DeepSpeed")
        model = model.to(torch.bfloat16)
    else:
        if logger:
            logger.info("Using fp32 precision")
        for param in model.parameters():
            if param.dtype != torch.float32:
                param.data = param.data.float()

    # 创建4个数据集
    if logger:
        logger.info("Creating multi-dataset...")
    
    # RefCOCO
    train_dataset_coco = MultiDatasetWrapper(
        ReferDataset(
            refer_data_root=args.data_root,
            dataset='refcoco',
            splitBy='unc',
            bert_tokenizer=args.tokenizer_type,
            max_tokens=getattr(args, 'max_tokens', 30),
            split='train',
            eval_mode=False,
            size=getattr(args, 'img_size', 320),
            precision=args.precision
        ), 'refcoco'
    )
    
    # RefCOCO+
    train_dataset_refcocoplus = MultiDatasetWrapper(
         ReferDataset(
             refer_data_root=args.data_root,
             dataset='refcoco+',
             splitBy='unc',
             bert_tokenizer=args.tokenizer_type,
             max_tokens=getattr(args, 'max_tokens', 30),
             split='train',
             eval_mode=False,
             size=getattr(args, 'img_size', 320),
             precision=args.precision
         ), 'refcoco+'
     )
    
    # # RefCOCOg
    train_dataset_refcocog = MultiDatasetWrapper(
         ReferDataset(
             refer_data_root=args.data_root,
             dataset='refcocog',
             splitBy='umd',
             bert_tokenizer=args.tokenizer_type,
             max_tokens=getattr(args, 'max_tokens', 30),
             split='train',
             eval_mode=False,
             size=getattr(args, 'img_size', 320),
             precision=args.precision
         ), 'refcocog'
    )
    
    # # Ref-ZOM
    train_dataset_zom = MultiDatasetWrapper(
         ReferzomDataset(
             refer_data_root=args.data_root,
             dataset='ref-zom',
             splitBy='final',
             bert_tokenizer=args.tokenizer_type,
             max_tokens=getattr(args, 'max_tokens', 30),
             split='train',
             eval_mode=False,
             size=getattr(args, 'img_size', 320),
             precision=args.precision
         ), 'ref-zom'
     )

    train_dataset_gref = MultiDatasetWrapper(
         GRefDataset(
             refer_data_root=args.data_root,
             dataset='grefcoco',
             splitBy='unc',
             bert_tokenizer=args.tokenizer_type,
             max_tokens=getattr(args, 'max_tokens', 30),
             split='train',
             eval_mode=False,
             size=getattr(args, 'img_size', 320),
             precision=args.precision
         ), 'grefcoco'
    )
    # 合并所有训练数据集
    train_dataset = torch.utils.data.ConcatDataset([
        train_dataset_refcocog,
        train_dataset_zom,
        train_dataset_gref,
        train_dataset_coco,
        train_dataset_refcocoplus
    ])
    
    # 验证数据集（使用RefCOCO）
    val_dataset_coco = ReferDataset(
        refer_data_root=args.data_root,
        dataset='refcoco',
        splitBy='unc',
        bert_tokenizer=args.tokenizer_type,
        max_tokens=getattr(args, 'max_tokens', 30),
        split='val',
        eval_mode=False,
        size=getattr(args, 'img_size', 320),
        precision=args.precision
    )
    # val_dataset_refcocoplus = ReferDataset(
    #     refer_data_root=args.data_root,
    #     dataset='refcoco+',
    #     splitBy='unc',
    #     bert_tokenizer=args.tokenizer_type,
    #     max_tokens=getattr(args, 'max_tokens', 30),
    #     split='val',
    #     eval_mode=False,
    #     size=getattr(args, 'img_size', 320),
    #     precision=args.precision
    # )
    val_dataset = torch.utils.data.ConcatDataset([val_dataset_coco])

    if logger:
        logger.info("Creating data loaders...")
        logger.info(f"Total training samples: {len(train_dataset)}")
        logger.info(f"  - RefCOCO: {len(train_dataset_coco)}")
        # logger.info(f"  - RefCOCO+: {len(train_dataset_refcocoplus)}")
        # logger.info(f"  - RefCOCOg: {len(train_dataset_refcocog)}")
        # logger.info(f"  - Ref-ZOM: {len(train_dataset_zom)}")
        logger.info(f"Total validation samples: {len(val_dataset)}")
    
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=12,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        sampler=val_sampler,
        num_workers=8,
        pin_memory=True
    )

    # 初始化 DeepSpeed 引擎
    if logger:
        logger.info("Initializing DeepSpeed engine...")
    
    # 创建参数组
    if hasattr(model, 'params_to_optimize'):
        param_groups = model.params_to_optimize()
    else:
        param_groups = model.parameters()

    if logger:
        total_params, trainable_params = count_parameters(model)
        logger.info(f"\nModel Parameters:")
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        logger.info(f"Non-trainable parameters: {total_params - trainable_params:,}")

    # 初始化 DeepSpeed 引擎
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=param_groups,
        config=ds_config
    )

        # resume support
    start_epoch = 0
    best_iou_miou_sum = 0
    if rank == 0 and hasattr(args, 'resume') and args.resume:
        import glob
        resume_path = args.resume
        # 如果 resume_path 是 best_iou_miou_model 目录，自动查找 global_step 子目录
        if os.path.isdir(resume_path):
            # 查找所有 global_step 子目录
            subdirs = glob.glob(os.path.join(resume_path, "global_step*"))
            if subdirs:
                # 取最新的 global_step 子目录
                latest_subdir = max(subdirs, key=os.path.getmtime)
                logger.info(f"Resuming from checkpoint: {latest_subdir}")
            else:
                logger.info(f"Resuming from checkpoint: {resume_path}")
        else:
            logger.info(f"Resuming from checkpoint: {resume_path}")
        
        # 尝试加载检查点，如果 world_size 不匹配则只加载模型权重
        try:
            # 实际上 load_checkpoint 只需要传父目录，deepspeed 会自动找最新 global_step
            _, client_state = model_engine.load_checkpoint(resume_path)
            # 处理 client_state 为 None 的情况（checkpoint 加载失败）
            if client_state is not None:
                start_epoch = client_state.get('epoch', 0)
                best_iou_miou_sum = client_state.get('best_iou_miou_sum', 0)
                logger.info(f"Resumed from epoch {start_epoch}, best_iou_miou_sum={best_iou_miou_sum}")
            else:
                logger.warning(f"Failed to load checkpoint from {resume_path}, starting from epoch 0")
                start_epoch = 0
                best_iou_miou_sum = 0
        except Exception as e:
            error_msg = str(e)
            # 检查是否是 world_size 不匹配的错误
            if "world size" in error_msg.lower() or "ZeRO" in error_msg or "ZeRORuntimeException" in error_msg:
                logger.warning(f"Checkpoint was saved with different world_size. Loading model weights only (skipping optimizer states)...")
                logger.warning(f"Original error: {error_msg}")
                try:
                    # 只加载模型权重，不加载优化器状态和调度器状态
                    _, client_state = model_engine.load_checkpoint(
                        resume_path, 
                        load_optimizer_states=False, 
                        load_lr_scheduler_states=False
                    )
                    if client_state is not None:
                        start_epoch = client_state.get('epoch', 0)
                        best_iou_miou_sum = client_state.get('best_iou_miou_sum', 0)
                        logger.info(f"Loaded model weights only. Resumed from epoch {start_epoch}, best_iou_miou_sum={best_iou_miou_sum}")
                        logger.info("Note: Optimizer states were not loaded due to world_size mismatch. Training will continue with new optimizer states.")
                    else:
                        logger.warning(f"Failed to load checkpoint even without optimizer states, starting from epoch 0")
                        start_epoch = 0
                        best_iou_miou_sum = 0
                except Exception as e2:
                    logger.error(f"Failed to load checkpoint even without optimizer states: {e2}")
                    logger.warning("Starting training from scratch...")
                    start_epoch = 0
                    best_iou_miou_sum = 0
            else:
                # 其他类型的错误
                logger.error(f"Failed to load checkpoint: {e}")
                logger.warning("Starting training from scratch...")
                start_epoch = 0
                best_iou_miou_sum = 0
    # broadcast resume info to all ranks
    start_epoch_tensor = torch.tensor([start_epoch], dtype=torch.int, device=device)
    best_iou_miou_sum_tensor = torch.tensor([best_iou_miou_sum], dtype=torch.float, device=device)
    dist.broadcast(start_epoch_tensor, src=0)
    dist.broadcast(best_iou_miou_sum_tensor, src=0)
    start_epoch = int(start_epoch_tensor.item())
    best_iou_miou_sum = float(best_iou_miou_sum_tensor.item())
    
    if logger:
        logger.info("Starting enhanced multi-dataset training...")
        
        # 打印loss配置信息
        if getattr(args, 'use_enhanced_loss', False):
            logger.info("=== Enhanced Loss Configuration ===")
            logger.info(f"Loss config path: {getattr(args, 'loss_config_path', 'None')}")
            
            # 打印loss缩放因子
            if hasattr(model_engine.module, 'criterion') and hasattr(model_engine.module.criterion, 'loss_scaling_factors'):
                scaling_factors = model_engine.module.criterion.loss_scaling_factors
                logger.info(f"Loss scaling factors: {scaling_factors}")
            
            # 打印自适应权重信息
            if hasattr(model_engine.module, 'criterion') and hasattr(model_engine.module.criterion, 'adaptive_weighting'):
                if hasattr(model_engine.module.criterion.adaptive_weighting, 'log_vars'):
                    # 修复BFloat16到numpy转换问题：先转换为float32
                    init_weights = torch.exp(model_engine.module.criterion.adaptive_weighting.log_vars).detach().cpu().float().numpy()
                    logger.info(f"Initial adaptive weights: {init_weights}")
                    logger.info(f"Temperature: {model_engine.module.criterion.adaptive_weighting.temperature}")
            
            # 打印课程学习配置
            if hasattr(model_engine.module, 'criterion') and hasattr(model_engine.module.criterion, 'curriculum_schedule'):
                curriculum_schedule = model_engine.module.criterion.curriculum_schedule
                logger.info(f"Curriculum schedule: {curriculum_schedule}")
            
            # 打印数据集权重配置
            if hasattr(model_engine.module, 'criterion') and hasattr(model_engine.module.criterion, 'dataset_weights'):
                dataset_weights = model_engine.module.criterion.dataset_weights
                logger.info(f"Dataset weights: {dataset_weights}")
            
            logger.info("=" * 40)
    
    global_step = 0
    for epoch in range(start_epoch, args.epochs):
        model_engine.train()
        train_sampler.set_epoch(epoch)
        
        # 设置当前epoch用于课程学习
        if getattr(args, 'use_enhanced_loss', False) and hasattr(model_engine.module, 'criterion'):
            if hasattr(model_engine.module.criterion, 'set_epoch'):
                model_engine.module.criterion.set_epoch(epoch)
        
        total_loss = 0
        total_mask_loss = 0
        total_dice_loss = 0
        
        # 增强损失的统计
        if getattr(args, 'use_enhanced_loss', False):
            total_focal_loss = 0
            total_iou_loss = 0
            total_boundary_loss = 0
            total_curriculum_info = {}
            total_dataset_weights = {}
            dataset_loss_stats = {}
        
        if logger:
            pbar = tqdm(train_loader, disable=not sys.stdout.isatty(), desc=f'Enhanced Multi-Dataset Epoch {epoch+1}/{args.epochs}')
        else:
            pbar = train_loader
            
        for batch_idx, (samples, targets) in enumerate(pbar):
            # 数据准备
            if use_fp16:
                img = samples['img'].to(device, non_blocking=True).half()
            elif use_bf16:
                img = samples['img'].to(device, non_blocking=True).to(torch.bfloat16)
            else:
                img = samples['img'].to(device, non_blocking=True)
            
            word_ids = samples['word_ids'].to(device, non_blocking=True)
            word_masks = samples['word_masks'].to(device, non_blocking=True)
            target = targets['mask'].to(device, non_blocking=True).squeeze(1)
            orig_size = samples['orig_size']
            
            # 获取数据集名称
            dataset_name = samples.get('dataset_name', 'unknown')
            # 修复：如果dataset_name是列表，取第一个元素
            if isinstance(dataset_name, list):
                dataset_name = dataset_name[0] if dataset_name else 'unknown'
            
            if target.sum() < 10:
                if logger:
                    logger.warning("Skipping sample with empty/too small mask")
                continue

            # 前向传播
            loss_dict = model_engine(img, word_ids, word_masks, target)
            loss = loss_dict['total_loss']
            
            # 确保loss是标量张量
            if not loss.requires_grad:
                continue
                
            # 检查损失值
            if torch.isnan(loss) or torch.isinf(loss):
                if logger:
                    logger.error(f"NaN/Inf loss detected: {loss.item()}")
                continue
            
            # 在backward之前提取所有需要的损失值
            loss_item = loss.item()
            mask_loss_item = loss_dict['loss_mask'].item()
            dice_loss_item = loss_dict['loss_dice'].item()
            
            # 提取增强损失值（如果存在）
            focal_loss_item = loss_dict.get('loss_focal', torch.tensor(0.0)).item() if 'loss_focal' in loss_dict else 0.0
            iou_loss_item = loss_dict.get('loss_iou', torch.tensor(0.0)).item() if 'loss_iou' in loss_dict else 0.0
            boundary_loss_item = loss_dict.get('loss_boundary', torch.tensor(0.0)).item() if 'loss_boundary' in loss_dict else 0.0
            
            # 提取课程学习和数据集权重信息（如果存在）
            curriculum_weights = loss_dict.get('curriculum_weights', {})
            dataset_weight = loss_dict.get('dataset_weight', 1.0)
                
            # DeepSpeed 反向传播和优化器步骤
            # 确保loss是标量且不保留计算图
            if loss.dim() > 0:
                loss = loss.mean()
            model_engine.backward(loss)
            model_engine.step()
            
            # 统计损失（使用预先提取的值）
            total_loss += loss_item
            total_mask_loss += mask_loss_item
            total_dice_loss += dice_loss_item
            
            # 按数据集统计损失
            if dataset_name not in dataset_loss_stats:
                dataset_loss_stats[dataset_name] = {
                    'total_loss': 0, 'count': 0,
                    'mask_loss': 0, 'dice_loss': 0
                }
            dataset_loss_stats[dataset_name]['total_loss'] += loss_item
            dataset_loss_stats[dataset_name]['count'] += 1
            dataset_loss_stats[dataset_name]['mask_loss'] += mask_loss_item
            dataset_loss_stats[dataset_name]['dice_loss'] += dice_loss_item
            
            # 增强损失统计
            if getattr(args, 'use_enhanced_loss', False):
                total_focal_loss += focal_loss_item
                total_iou_loss += iou_loss_item
                total_boundary_loss += boundary_loss_item
                
                # 统计课程学习信息
                for key, value in curriculum_weights.items():
                    if key not in total_curriculum_info:
                        total_curriculum_info[key] = 0
                    total_curriculum_info[key] += value
                
                # 统计数据集权重信息
                if dataset_name not in total_dataset_weights:
                    total_dataset_weights[dataset_name] = []
                total_dataset_weights[dataset_name].append(dataset_weight)
            
            # 更新进度条
            current_loss = total_loss / (batch_idx + 1)
            current_mask_loss = total_mask_loss / (batch_idx + 1)
            current_dice_loss = total_dice_loss / (batch_idx + 1)
            
            postfix_dict = {
                'loss': current_loss,
                'mask_loss': current_mask_loss,
                'dice_loss': current_dice_loss
            }
            
            if getattr(args, 'use_enhanced_loss', False):
                if focal_loss_item > 0:
                    postfix_dict['focal_loss'] = total_focal_loss / (batch_idx + 1)
                if iou_loss_item > 0:
                    postfix_dict['iou_loss'] = total_iou_loss / (batch_idx + 1)
                if boundary_loss_item > 0:
                    postfix_dict['boundary_loss'] = total_boundary_loss / (batch_idx + 1)
            
            if logger:
                pbar.set_postfix(postfix_dict)
                if (batch_idx + 1) % 20 == 0:
                    # 详细的loss打印
                    log_msg = f"Enhanced Multi-Dataset Epoch {epoch+1}/{args.epochs} - Batch {batch_idx+1}/{len(train_loader)} - "
                    log_msg += f"Total Loss: {current_loss:.4f} - Mask Loss: {current_mask_loss:.4f} - Dice Loss: {current_dice_loss:.4f}"
                    
                    # 添加增强loss组件
                    if getattr(args, 'use_enhanced_loss', False):
                        if focal_loss_item > 0:
                            current_focal = total_focal_loss / (batch_idx + 1)
                            log_msg += f" - Focal Loss: {current_focal:.4f}"
                        if iou_loss_item > 0:
                            current_iou = total_iou_loss / (batch_idx + 1)
                            log_msg += f" - IoU Loss: {current_iou:.4f}"
                        if boundary_loss_item > 0:
                            current_boundary = total_boundary_loss / (batch_idx + 1)
                            log_msg += f" - Boundary Loss: {current_boundary:.4f}"
                        
                        # 打印课程学习权重
                        if curriculum_weights:
                            log_msg += f" - Curriculum: {curriculum_weights}"
                        
                        # 打印数据集权重
                        if dataset_name:
                            log_msg += f" - Dataset({dataset_name}): {dataset_weight:.3f}"
                    
                    logger.info(log_msg)

        # 打印epoch结束时的详细loss统计
        if rank == 0 and logger:
            avg_loss = total_loss / len(train_loader)
            avg_mask_loss = total_mask_loss / len(train_loader)
            avg_dice_loss = total_dice_loss / len(train_loader)
            
            logger.info(f"\n=== Enhanced Multi-Dataset Epoch {epoch+1} Loss Summary ===")
            logger.info(f"Average Total Loss: {avg_loss:.4f}")
            logger.info(f"Average Mask Loss: {avg_mask_loss:.4f}")
            logger.info(f"Average Dice Loss: {avg_dice_loss:.4f}")
            
            # 按数据集统计
            logger.info("Dataset-wise Loss Statistics:")
            for ds_name, stats in dataset_loss_stats.items():
                if stats['count'] > 0:
                    avg_ds_loss = stats['total_loss'] / stats['count']
                    avg_ds_mask = stats['mask_loss'] / stats['count']
                    avg_ds_dice = stats['dice_loss'] / stats['count']
                    logger.info(f"  {ds_name}: Total={avg_ds_loss:.4f}, Mask={avg_ds_mask:.4f}, Dice={avg_ds_dice:.4f} (samples: {stats['count']})")
            
            if getattr(args, 'use_enhanced_loss', False):
                if 'loss_focal' in loss_dict:
                    avg_focal_loss = total_focal_loss / len(train_loader)
                    logger.info(f"Average Focal Loss: {avg_focal_loss:.4f}")
                if 'loss_iou' in loss_dict:
                    avg_iou_loss = total_iou_loss / len(train_loader)
                    logger.info(f"Average IoU Loss: {avg_iou_loss:.4f}")
                if 'loss_boundary' in loss_dict:
                    avg_boundary_loss = total_boundary_loss / len(train_loader)
                    logger.info(f"Average Boundary Loss: {avg_boundary_loss:.4f}")
                
                # 打印课程学习统计
                if total_curriculum_info:
                    logger.info("Curriculum Learning Progress:")
                    for key, total_value in total_curriculum_info.items():
                        avg_value = total_value / len(train_loader)
                        logger.info(f"  {key}: {avg_value:.4f}")
                
                # 打印数据集权重统计
                if total_dataset_weights:
                    logger.info("Dataset Weight Statistics:")
                    for ds_name, weights_list in total_dataset_weights.items():
                        avg_weight = sum(weights_list) / len(weights_list)
                        logger.info(f"  {ds_name}: {avg_weight:.3f} (samples: {len(weights_list)})")
            
            logger.info("=" * 50)

        # 验证和保存
        save_best = False
        metrics = None

        if rank == 0:
            logger.info(f"\nValidating enhanced multi-dataset epoch {epoch+1}...")
            model_engine.eval()
            metrics = validate(model_engine.module, val_loader, device, use_fp16, use_bf16)
            model_engine.train()
            
            logger.info(f"Enhanced multi-dataset validation metrics for epoch {epoch+1}:")
            logger.info(f"mIoU: {metrics['mIoU']:.4f}")
            logger.info(f"oIoU: {metrics['oIoU']:.4f}")
            logger.info(f"gIoU: {metrics['gIoU']:.4f}")
            logger.info(f"Acc: {metrics['Acc']:.4f}")
            logger.info(f"pointM: {metrics['pointM']:.4f}")
            logger.info(f"best_IoU: {metrics['best_IoU']:.4f}")

            # 判断是否为当前最优
            iou_miou_sum = metrics['oIoU'] + metrics['mIoU']
            if iou_miou_sum > best_iou_miou_sum:
                best_iou_miou_sum = iou_miou_sum
                save_best = True

        # 广播保存决策和 metrics 到所有进程
        save_best_tensor = torch.tensor([int(save_best)], device=device)
        dist.broadcast(save_best_tensor, src=0)
        save_best = bool(save_best_tensor.item())

        # 广播 metrics 到所有进程
        if rank != 0:
            metrics = {
                'mIoU': torch.zeros(1, device=device),
                'oIoU': torch.zeros(1, device=device),
                'gIoU': torch.zeros(1, device=device),
                'Acc': torch.zeros(1, device=device),
                'pointM': torch.zeros(1, device=device),
                'best_IoU': torch.zeros(1, device=device)
            }
        for key in metrics:
            tensor = metrics[key] if isinstance(metrics[key], torch.Tensor) else torch.tensor([metrics[key]], device=device)
            dist.broadcast(tensor, src=0)
            metrics[key] = tensor.item()

        # 所有进程准备 client_state
        client_state = {
            'epoch': epoch + 1,
            'best_iou_miou_sum': best_iou_miou_sum,
            'metrics': metrics
        }

        # 保存模型
        if save_best:
            best_path = os.path.join(args.output_dir, 'enhanced_multi_dataset_best_iou_miou_model')
            if rank == 0:
                os.makedirs(args.output_dir, exist_ok=True)
                if os.path.exists(best_path):
                    shutil.rmtree(best_path)
                os.makedirs(best_path, exist_ok=True)
            dist.barrier()
            model_engine.save_checkpoint(best_path, client_state=client_state)
            if rank == 0:
                logger.info(f"Saved new enhanced multi-dataset best iou+miou model with score: {best_iou_miou_sum:.4f}")

        # 保存当前 epoch 的 checkpoint
        current_checkpoint = os.path.join(args.output_dir, f'enhanced_multi_dataset_checkpoint_epoch_{epoch+1}')
        if rank == 0:
            os.makedirs(args.output_dir, exist_ok=True)
        dist.barrier()
        model_engine.save_checkpoint(current_checkpoint, client_state=client_state)
        if rank == 0:
            logger.info(f"Saved enhanced multi-dataset checkpoint for epoch {epoch+1}")

        # 删除上一个 checkpoint
        prev_checkpoint = os.path.join(args.output_dir, f'enhanced_multi_dataset_checkpoint_epoch_{epoch}')
        if rank == 0 and os.path.exists(prev_checkpoint):
            shutil.rmtree(prev_checkpoint)
        dist.barrier()
    
    dist.destroy_process_group()


if __name__ == '__main__':
    import torch.multiprocessing as mp
    mp.set_start_method('fork', force=True)
    try:
        main()
        # evaluate_four_datasets()
    except Exception as e:
        import traceback
        print(f"训练过程中发生错误: {e}")
        traceback.print_exc()
        # 确保分布式进程组被正确清理
        if dist.is_initialized():
            dist.destroy_process_group()
        # 退出时使用非零状态码，表示错误
        sys.exit(1)
