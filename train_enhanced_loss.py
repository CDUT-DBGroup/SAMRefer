#!/usr/bin/env python3
"""
Enhanced training script with improved loss functions
使用增强损失函数的训练脚本
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
    log_file = os.path.join(output_dir, f'enhanced_training_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
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


def main():
    # 解析命令行参数
    # parser = argparse.ArgumentParser(description='Enhanced ReferSAM Training')
    # parser = add_enhanced_loss_args(parser)
    # args, unknown = parser.parse_known_args()
    
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
        logger.info(f"Starting Enhanced DeepSpeed training: rank {rank}, world_size {world_size}")
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
            logger.info("Using Enhanced Loss Functions: Focal + IoU + Boundary + Adaptive Weighting")
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

    # 创建数据集 (保持原有逻辑)
    if logger:
        logger.info("Creating datasets...")
    train_dataset_coco = ReferDataset(
        refer_data_root=args.data_root,
        dataset='refcoco',
        splitBy='unc',
        bert_tokenizer=args.tokenizer_type,
        max_tokens=getattr(args, 'max_tokens', 30),
        split='train',
        eval_mode=False,
        size=getattr(args, 'img_size', 320),
        precision=args.precision
    )
    
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
    
    train_dataset = torch.utils.data.ConcatDataset([train_dataset_coco])
    val_dataset = torch.utils.data.ConcatDataset([val_dataset_coco])

    if logger:
        logger.info("Creating data loaders...")
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

    # 训练循环 (保持原有逻辑，但添加增强损失的日志)
    start_epoch = 0
    best_iou_miou_sum = 0
    
    if logger:
        logger.info("Starting enhanced training...")
    
    global_step = 0
    for epoch in range(start_epoch, args.epochs):
        model_engine.train()
        train_sampler.set_epoch(epoch)
        total_loss = 0
        total_mask_loss = 0
        total_dice_loss = 0
        
        # 增强损失的统计
        if getattr(args, 'use_enhanced_loss', False):
            total_focal_loss = 0
            total_iou_loss = 0
            total_boundary_loss = 0
        
        if logger:
            pbar = tqdm(train_loader, disable=not sys.stdout.isatty(), desc=f'Enhanced Epoch {epoch+1}/{args.epochs}')
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
            
            if target.sum() < 10:
                if logger:
                    logger.warning("Skipping sample with empty/too small mask")
                continue

            # 前向传播
            loss_dict = model_engine(img, word_ids, word_masks, target)
            loss = loss_dict['total_loss']
                
            # 检查损失值
            if torch.isnan(loss) or torch.isinf(loss):
                if logger:
                    logger.error(f"NaN/Inf loss detected: {loss.item()}")
                continue
                
            # DeepSpeed 反向传播和优化器步骤
            model_engine.backward(loss)
            model_engine.step()
            
            # 统计损失
            total_loss += loss.item()
            total_mask_loss += loss_dict['loss_mask'].item()
            total_dice_loss += loss_dict['loss_dice'].item()
            
            # 增强损失统计
            if getattr(args, 'use_enhanced_loss', False):
                if 'loss_focal' in loss_dict:
                    total_focal_loss += loss_dict['loss_focal'].item()
                if 'loss_iou' in loss_dict:
                    total_iou_loss += loss_dict['loss_iou'].item()
                if 'loss_boundary' in loss_dict:
                    total_boundary_loss += loss_dict['loss_boundary'].item()
            
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
                if 'loss_focal' in loss_dict:
                    postfix_dict['focal_loss'] = total_focal_loss / (batch_idx + 1)
                if 'loss_iou' in loss_dict:
                    postfix_dict['iou_loss'] = total_iou_loss / (batch_idx + 1)
                if 'loss_boundary' in loss_dict:
                    postfix_dict['boundary_loss'] = total_boundary_loss / (batch_idx + 1)
            
            if logger:
                pbar.set_postfix(postfix_dict)
                if (batch_idx + 1) % 10 == 0:
                    logger.info(f"Enhanced Epoch {epoch+1}/{args.epochs} - Batch {batch_idx+1}/{len(train_loader)} - "
                                f"Loss: {current_loss:.4f} - Mask Loss: {current_mask_loss:.4f} - "
                                f"Dice Loss: {current_dice_loss:.4f}")

        # 验证和保存 (保持原有逻辑)
        save_best = False
        metrics = None

        if rank == 0:
            logger.info(f"\nValidating enhanced epoch {epoch+1}...")
            model_engine.eval()
            metrics = validate(model_engine.module, val_loader, device, use_fp16, use_bf16)
            model_engine.train()
            
            logger.info(f"Enhanced validation metrics for epoch {epoch+1}:")
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
            best_path = os.path.join(args.output_dir, 'enhanced_best_iou_miou_model')
            if rank == 0:
                os.makedirs(args.output_dir, exist_ok=True)
                if os.path.exists(best_path):
                    shutil.rmtree(best_path)
                os.makedirs(best_path, exist_ok=True)
            dist.barrier()
            model_engine.save_checkpoint(best_path, client_state=client_state)
            if rank == 0:
                logger.info(f"Saved new enhanced best iou+miou model with score: {best_iou_miou_sum:.4f}")

        # 保存当前 epoch 的 checkpoint
        current_checkpoint = os.path.join(args.output_dir, f'enhanced_checkpoint_epoch_{epoch+1}')
        if rank == 0:
            os.makedirs(args.output_dir, exist_ok=True)
        dist.barrier()
        model_engine.save_checkpoint(current_checkpoint, client_state=client_state)
        if rank == 0:
            logger.info(f"Saved enhanced checkpoint for epoch {epoch+1}")

        # 删除上一个 checkpoint
        prev_checkpoint = os.path.join(args.output_dir, f'enhanced_checkpoint_epoch_{epoch}')
        if rank == 0 and os.path.exists(prev_checkpoint):
            shutil.rmtree(prev_checkpoint)
        dist.barrier()
    
    dist.destroy_process_group()


if __name__ == '__main__':
    import torch.multiprocessing as mp
    mp.set_start_method('fork', force=True)
    main()

