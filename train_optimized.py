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
from model.builder import refersam
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
from model.data_augmentation import AdaptiveAugmentation
import torch.optim.lr_scheduler as lr_scheduler
import torchvision.transforms as transforms

# 内存优化设置
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

def set_seed(seed=123456):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed()

def setup_logger(output_dir, rank=0):
    if rank != 0:
        return None
    log_file = os.path.join(output_dir, f'optimized_training_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
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

def create_optimized_scheduler(optimizer, args, num_training_steps):
    """创建优化的学习率调度器"""
    if hasattr(args, 'cosine_lr') and args.cosine_lr:
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, 
            T_0=num_training_steps // 4,
            T_mult=2,
            eta_min=1e-7
        )
    else:
        scheduler = lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=args.lr,
            total_steps=num_training_steps,
            pct_start=0.1,
            anneal_strategy='cos'
        )
    return scheduler

def main():
    # 初始化 DeepSpeed
    deepspeed.init_distributed()
    args = get_args()
    
    # 确保 config 是文件路径字符串
    if hasattr(args, 'deepspeed_config') and args.deepspeed_config:
        ds_config = args.deepspeed_config
    else:
        ds_config = 'configs/ds_config.json'

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
        logger.info(f"Starting Optimized DeepSpeed training: rank {rank}, world_size {world_size}")
        logger.info(f"Batch size: {args.batch_size}")
        logger.info(f"Epochs: {args.epochs}")
        logger.info(f"Learning rate: {args.lr}")
        logger.info(f"Weight decay: {args.weight_decay}")
        logger.info(f"Image size: {args.img_size}")
        logger.info(f"Using device: {device}")

    if logger:
        logger.info("Creating optimized ReferSAM model...")
    model = refersam(args=args)
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

    if logger:
        logger.info("Creating optimized datasets...")
    
    # 创建数据集
    train_dataset = ReferDataset(
        data_root=args.data_root,
        dataset='refcoco',
        split='train',
        image_size=args.img_size,
        max_tokens=args.max_tokens,
        tokenizer_type=args.tokenizer_type,
        ck_bert=args.ck_bert
    )
    
    val_dataset = ReferDataset(
        data_root=args.data_root,
        dataset='refcoco',
        split='val',
        image_size=args.img_size,
        max_tokens=args.max_tokens,
        tokenizer_type=args.tokenizer_type,
        ck_bert=args.ck_bert
    )

    if logger:
        logger.info("Creating optimized data loaders...")
    
    # 创建数据加载器
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        sampler=val_sampler,
        num_workers=4,
        pin_memory=True,
        drop_last=False
    )

    if logger:
        logger.info("Initializing optimized DeepSpeed engine...")
    
    # 初始化DeepSpeed引擎
    model_engine, optimizer, train_loader, lr_scheduler = deepspeed.initialize(
        model=model,
        training_data=train_dataset,
        config=ds_config,
    )

    # 计算训练步数
    num_training_steps = len(train_loader) * args.epochs
    
    if logger:
        total_params, trainable_params = count_parameters(model)
        logger.info(f"Model Parameters:")
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        logger.info(f"Non-trainable parameters: {total_params - trainable_params:,}")
        logger.info(f"Training steps: {num_training_steps:,}")

    # 初始化数据增强
    data_augmentation = AdaptiveAugmentation(max_epochs=args.epochs, img_size=args.img_size)
    
    # 训练循环
    best_score = 0.0
    global_step = 0
    
    for epoch in range(args.epochs):
        model_engine.train()
        train_sampler.set_epoch(epoch)
        
        if logger:
            logger.info(f"Starting epoch {epoch + 1}/{args.epochs}")
        
        epoch_loss = 0.0
        epoch_mask_loss = 0.0
        epoch_dice_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")
        
        for batch_idx, batch in enumerate(progress_bar):
            images = batch['image'].to(device, non_blocking=True)
            texts = batch['text']
            text_masks = batch['text_mask'].to(device, non_blocking=True)
            targets = batch['mask'].to(device, non_blocking=True)
            orig_sizes = batch['orig_size']
            
            # 应用数据增强
            if epoch < args.epochs * 0.8:  # 前80%的epoch使用数据增强
                for i in range(images.shape[0]):
                    img_pil = transforms.ToPILImage()(images[i])
                    mask_pil = transforms.ToPILImage()(targets[i].float())
                    img_pil, mask_pil = data_augmentation(img_pil, mask_pil, epoch)
                    images[i] = transforms.ToTensor()(img_pil)
                    targets[i] = transforms.ToTensor()(mask_pil)
            
            # 前向传播
            losses = model_engine(images, texts, text_masks, targets=targets, orig_size=orig_sizes)
            
            if isinstance(losses, dict):
                total_loss = losses['total_loss']
                mask_loss = losses.get('loss_mask', torch.tensor(0.0))
                dice_loss = losses.get('loss_dice', torch.tensor(0.0))
            else:
                total_loss = losses
                mask_loss = torch.tensor(0.0)
                dice_loss = torch.tensor(0.0)
            
            # 反向传播
            model_engine.backward(total_loss)
            model_engine.step()
            
            # 更新统计信息
            epoch_loss += total_loss.item()
            epoch_mask_loss += mask_loss.item()
            epoch_dice_loss += dice_loss.item()
            global_step += 1
            
            # 更新进度条
            progress_bar.set_postfix({
                'Loss': f'{total_loss.item():.4f}',
                'Mask': f'{mask_loss.item():.4f}',
                'Dice': f'{dice_loss.item():.4f}',
                'LR': f'{lr_scheduler.get_lr()[0]:.2e}'
            })
            
            # 定期打印日志
            if global_step % 10 == 0 and logger:
                logger.info(f"Epoch {epoch + 1}/{args.epochs} - Batch {batch_idx + 1}/{len(train_loader)} - "
                           f"Loss: {total_loss.item():.4f} - Mask Loss: {mask_loss.item():.4f} - "
                           f"Dice Loss: {dice_loss.item():.4f}")
        
        # 计算平均损失
        avg_loss = epoch_loss / len(train_loader)
        avg_mask_loss = epoch_mask_loss / len(train_loader)
        avg_dice_loss = epoch_dice_loss / len(train_loader)
        
        if logger:
            logger.info(f"Epoch {epoch + 1} completed - Avg Loss: {avg_loss:.4f}, "
                       f"Avg Mask Loss: {avg_mask_loss:.4f}, Avg Dice Loss: {avg_dice_loss:.4f}")
        
        # 验证
        if (epoch + 1) % 2 == 0:  # 每2个epoch验证一次
            if logger:
                logger.info(f"Validating epoch {epoch + 1}...")
            
            model_engine.eval()
            val_metrics = validate(model_engine, val_loader, device, use_fp16, use_bf16)
            
            if logger:
                logger.info(f"Validation metrics for epoch {epoch + 1}:")
                for key, value in val_metrics.items():
                    logger.info(f"{key}: {value:.4f}")
            
            # 保存最佳模型
            current_score = val_metrics.get('mIoU', 0.0) + val_metrics.get('oIoU', 0.0)
            if current_score > best_score:
                best_score = current_score
                if logger:
                    logger.info(f"Saved new best model with score: {best_score:.4f}")
                
                # 保存检查点
                checkpoint_dir = os.path.join(args.output_dir, "best_optimized_model")
                model_engine.save_checkpoint(checkpoint_dir)
        
        # 定期保存检查点
        if (epoch + 1) % 5 == 0:
            checkpoint_dir = os.path.join(args.output_dir, f"checkpoint_epoch_{epoch + 1}")
            model_engine.save_checkpoint(checkpoint_dir)
            if logger:
                logger.info(f"Saved checkpoint for epoch {epoch + 1}")

    if logger:
        logger.info("Training completed!")
        logger.info(f"Best score achieved: {best_score:.4f}")

if __name__ == "__main__":
    main()
