import os
import sys
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
import numpy as np
from dataset.Dataset_referit import ReferitDataset
from dataset.ReferDataset import ReferDataset
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
    log_file = os.path.join(output_dir, f'training_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
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
    # 初始化 DeepSpeed
    ds_config = deepspeed.init_distributed()
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logger(args.output_dir, rank)
    if logger:
        logger.info(f"Starting DeepSpeed training: rank {rank}, world_size {world_size}")
        logger.info(f"Batch size: {args.batch_size}")
        logger.info(f"Epochs: {args.epochs}")
        logger.info(f"Learning rate: {args.lr}")
        logger.info(f"Weight decay: {args.weight_decay}")
        logger.info(f"Using device: {device}")

    if logger:
        logger.info("Creating ReferSAM model...")
    model = refersam(args=args)
    model = model.to(device)
    # 确保模型参数使用正确的数据类型
    for param in model.parameters():
        if param.dtype != torch.float32:
            param.data = param.data.float()
    
    if logger:
        total_params, trainable_params = count_parameters(model)
        logger.info(f"\nModel Parameters:")
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        logger.info(f"Non-trainable parameters: {total_params - trainable_params:,}")

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
    train_referit = ReferitDataset(root = args.data_referit_root, split="train", max_tokens=getattr(args, 'max_tokens', 30), size=getattr(args, 'img_size', 320))
    val_referit = ReferitDataset(root = args.data_referit_root, split="val", max_tokens=getattr(args, 'max_tokens', 30), size=getattr(args, 'img_size', 320))
    train_dataset = torch.utils.data.ConcatDataset([
        train_referit, train_dataset_coco
    ])
    val_dataset = torch.utils.data.ConcatDataset([
        val_referit,
    ])

    if logger:
        logger.info("Creating data loaders...")
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=0,  # 如果内存不足，设置为0
        pin_memory=False  # 如果内存不足，设置为False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        sampler=val_sampler,
        num_workers=0,  # 如果内存不足，设置为0
        pin_memory=False  # 如果内存不足，设置为False
    )

    # 初始化 DeepSpeed 引擎
    if logger:
        logger.info("Initializing DeepSpeed engine...")
    
    # 创建参数组
    if hasattr(model, 'params_to_optimize'):
        param_groups = model.params_to_optimize()
    else:
        param_groups = model.parameters()

    # 初始化 DeepSpeed 引擎
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=param_groups,
        config=args.deepspeed_config if hasattr(args, 'deepspeed_config') else 'ds_config.json'
    )

    # resume support
    start_epoch = 0
    best_iou_miou_sum = 0
    if rank == 0 and hasattr(args, 'resume') and args.resume and os.path.isfile(args.resume):
        logger.info(f"Resuming from checkpoint: {args.resume}")
        _, client_state = model_engine.load_checkpoint(args.resume)
        start_epoch = client_state.get('epoch', 0)
        best_iou_miou_sum = client_state.get('best_iou_miou_sum', 0)
        logger.info(f"Resumed from epoch {start_epoch}, best_iou_miou_sum={best_iou_miou_sum}")
    
    # broadcast resume info to all ranks
    start_epoch_tensor = torch.tensor([start_epoch], dtype=torch.int, device=device)
    best_iou_miou_sum_tensor = torch.tensor([best_iou_miou_sum], dtype=torch.float, device=device)
    dist.broadcast(start_epoch_tensor, src=0)
    dist.broadcast(best_iou_miou_sum_tensor, src=0)
    start_epoch = int(start_epoch_tensor.item())
    best_iou_miou_sum = float(best_iou_miou_sum_tensor.item())

    if logger:
        logger.info("Starting training...")
    global_step = 0
    for epoch in range(start_epoch, args.epochs):
        model_engine.train()
        train_sampler.set_epoch(epoch)
        total_loss = 0
        total_mask_loss = 0
        total_dice_loss = 0
        if logger:
            pbar = tqdm(train_loader, disable=not sys.stdout.isatty(), desc=f'Epoch {epoch+1}/{args.epochs}')
        else:
            pbar = train_loader
        for batch_idx, (samples, targets) in enumerate(pbar):
            img = samples['img'].to(device, non_blocking=True).half()
            word_ids = samples['word_ids'].to(device, non_blocking=True)
            word_masks = samples['word_masks'].to(device, non_blocking=True)
            target = targets['mask'].to(device, non_blocking=True).squeeze(1)
            orig_size = samples['orig_size']
            
            if target.sum() < 10:  # 面积太小的 mask
                if logger:
                    logger.warning("Skipping sample with empty/too small mask")
                continue

            # DeepSpeed 前向传播和反向传播
            loss_dict = model_engine(img, word_ids, word_masks, target)
            loss = loss_dict['total_loss']
                
            # 检查损失值是否为NaN或Inf
            if torch.isnan(loss) or torch.isinf(loss):
                if logger:
                    logger.error(f"NaN/Inf loss detected: {loss.item()}")
                    logger.error(f"Loss dict: {loss_dict}")
                    logger.error(f"Image stats: min={img.min().item():.6f}, max={img.max().item():.6f}, mean={img.mean().item():.6f}")
                    logger.error(f"Target stats: min={target.min().item():.6f}, max={target.max().item():.6f}, mean={target.mean().item():.6f}")
                continue
                
            # 检查各个损失分量
            for loss_name, loss_value in loss_dict.items():
                if torch.isnan(loss_value) or torch.isinf(loss_value):
                    if logger:
                        logger.error(f"NaN/Inf detected in {loss_name}: {loss_value.item()}")
                    continue
            
            # DeepSpeed 反向传播和优化器步骤
            model_engine.backward(loss)
            model_engine.step()
            
            # 调试：检查梯度是否正常
            if batch_idx == 0 and epoch == start_epoch:
                total_grad_norm = 0
                grad_info = []
                for name, param in model_engine.module.named_parameters():
                    if param.grad is not None:
                        param_norm = param.grad.data.norm(2)
                        total_grad_norm += param_norm.item() ** 2
                        grad_dtype = param.grad.dtype
                        grad_info.append(f"{name}: norm={param_norm.item():.6f}, dtype={grad_dtype}")
                total_grad_norm = total_grad_norm ** (1. / 2)
                if logger:
                    logger.info(f"Initial gradient norm: {total_grad_norm:.6f}")
                    logger.info("Gradient info (first 5 layers):")
                    for info in grad_info[:5]:
                        logger.info(f"  {info}")
            
            total_loss += loss.item()
            total_mask_loss += loss_dict['loss_mask'].item()
            total_dice_loss += loss_dict['loss_dice'].item()
            current_loss = total_loss / (batch_idx + 1)
            current_mask_loss = total_mask_loss / (batch_idx + 1)
            current_dice_loss = total_dice_loss / (batch_idx + 1)
            if logger:
                pbar.set_postfix({
                    'loss': current_loss,
                    'mask_loss': current_mask_loss,
                    'dice_loss': current_dice_loss
                })
                if (batch_idx + 1) % 10 == 0:
                    logger.info(f"Epoch {epoch+1}/{args.epochs} - Batch {batch_idx+1}/{len(train_loader)} - "
                                f"Loss: {current_loss:.4f} - Mask Loss: {current_mask_loss:.4f} - "
                                f"Dice Loss: {current_dice_loss:.4f}")

        # Validation and checkpointing only on rank 0
        if rank == 0:
            logger.info(f"\nValidating epoch {epoch+1}...")
            # 临时切换到评估模式
            model_engine.eval()
            metrics = validate(model_engine.module, val_loader, device)
            model_engine.train()
            
            logger.info(f"Validation metrics for epoch {epoch+1}:")
            logger.info(f"mIoU: {metrics['mIoU']:.4f}")
            logger.info(f"oIoU: {metrics['oIoU']:.4f}")
            logger.info(f"gIoU: {metrics['gIoU']:.4f}")
            logger.info(f"Acc: {metrics['Acc']:.4f}")
            logger.info(f"pointM: {metrics['pointM']:.4f}")
            logger.info(f"best_IoU: {metrics['best_IoU']:.4f}")
            
            # Save best iou+miou model
            iou_miou_sum = metrics['oIoU'] + metrics['mIoU']
            if iou_miou_sum > best_iou_miou_sum:
                best_iou_miou_sum = iou_miou_sum
                os.makedirs(args.output_dir, exist_ok=True)
                best_path = os.path.join(args.output_dir, 'best_iou_miou_model')
                
                # 使用 DeepSpeed 保存检查点
                client_state = {
                    'epoch': epoch + 1,
                    'best_iou_miou_sum': best_iou_miou_sum,
                    'metrics': metrics
                }
                model_engine.save_checkpoint(best_path, client_state=client_state)
                logger.info(f"Saved new best iou+miou model with score: {best_iou_miou_sum:.4f}")
            
            # 保存当前检查点
            os.makedirs(args.output_dir, exist_ok=True)
            current_checkpoint = os.path.join(args.output_dir, f'checkpoint_epoch_{epoch+1}')
            client_state = {
                'epoch': epoch + 1,
                'best_iou_miou_sum': best_iou_miou_sum
            }
            model_engine.save_checkpoint(current_checkpoint, client_state=client_state)
            
            # 删除前一个检查点
            prev_checkpoint = os.path.join(args.output_dir, f'checkpoint_epoch_{epoch}')
            if os.path.exists(prev_checkpoint):
                import shutil
                shutil.rmtree(prev_checkpoint)
    
    dist.destroy_process_group()

if __name__ == '__main__':
    import torch.multiprocessing as mp
    mp.set_start_method('fork', force=True)
    main()
    # 在训练结束后评估四个数据集
    evaluate_four_datasets()