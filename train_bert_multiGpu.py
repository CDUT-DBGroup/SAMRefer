import os
import sys
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
import numpy as np
from dataset.ReferDataset import ReferDataset
from get_args import get_args
from model.builder import refersam
from validation.evaluation import validate
import logging
import datetime
import random
import torch.nn as nn
import torch.distributed as dist
from torch.optim.lr_scheduler import CosineAnnealingLR

def set_seed(seed=123456):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed()
# Configure logging

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
    # DDP: initialize process group
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    from torch.cuda.amp import autocast, GradScaler

    scaler = GradScaler()


    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logger(args.output_dir, rank)
    if logger:
        logger.info(f"Starting DDP training: rank {rank}, world_size {world_size}")
        logger.info(f"Batch size: {args.batch_size}")
        logger.info(f"Epochs: {args.epochs}")
        logger.info(f"Learning rate: {args.lr}")
        logger.info(f"Weight decay: {args.weight_decay}")
        logger.info(f"Using device: {device}")

    if logger:
        logger.info("Creating ReferSAM model...")
    model = refersam(args=args)
    model = model.to(device)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
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
    val_dataset_coco = ReferDataset(
        refer_data_root=args.data_root,
        dataset='refcoco',
        splitBy='unc',
        bert_tokenizer=args.tokenizer_type,
        max_tokens=getattr(args, 'max_tokens', 30),
        split='val',
        eval_mode=True,
        size=getattr(args, 'img_size', 320),
        precision=args.precision
    )
    train_dataset_cocoplus = ReferDataset(
        refer_data_root=args.data_root,
        dataset='refcoco+',
        splitBy='unc',
        bert_tokenizer=args.tokenizer_type,
        max_tokens=getattr(args, 'max_tokens', 30),
        split='train',
        eval_mode=False,
        size=getattr(args, 'img_size', 320),
        precision=args.precision
    )
    # val_dataset_cocoplus = ReferDataset(
    #     refer_data_root=args.data_root,
    #     dataset='refcoco+',
    #     splitBy='unc',
    #     bert_tokenizer=args.tokenizer_type,
    #     max_tokens=getattr(args, 'max_tokens', 30),
    #     split='val',
    #     eval_mode=True,
    #     size=getattr(args, 'img_size', 320),
    #     precision=args.precision
    # )
    train_dataset_cocog = ReferDataset(
        refer_data_root=args.data_root,
        dataset='refcocog',
        splitBy='umd',
        bert_tokenizer=args.tokenizer_type,
        max_tokens=getattr(args, 'max_tokens', 30),
        split='train',
        eval_mode=False,
        size=getattr(args, 'img_size', 320),
        precision=args.precision
    )
    # val_dataset_cocog = ReferDataset(
    #     refer_data_root=args.data_root,
    #     dataset='refcocog',
    #     splitBy='umd',
    #     bert_tokenizer=args.tokenizer_type,
    #     max_tokens=getattr(args, 'max_tokens', 30),
    #     split='val',
    #     eval_mode=True,
    #     size=getattr(args, 'img_size', 320),
    #     precision=args.precision
    # )
    train_dataset = torch.utils.data.ConcatDataset([
        train_dataset_coco, train_dataset_cocoplus, train_dataset_cocog
    ])
    val_dataset = torch.utils.data.ConcatDataset([
        val_dataset_coco #, val_dataset_cocoplus, val_dataset_cocog
    ])

    if logger:
        logger.info("Creating data loaders...")
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=8,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        sampler=val_sampler,
        num_workers=4,
        pin_memory=True
    )

    if logger:
        logger.info("Initializing optimizer...")

    param_groups = model.parameters()
    # # 如果你自定义了 param 分组逻辑
    # if hasattr(model, 'module') and hasattr(model.module, 'params_to_optimize'):
    #     param_groups = model.module.params_to_optimize()
    # else:
    #     param_groups = model.parameters()

    optimizer = AdamW(
        param_groups,
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        betas=(0.9, 0.999),
        eps=1e-8
    )

    # 推荐：Cosine 学习率调度器（可替换为 ReduceLROnPlateau）
    # scheduler = CosineAnnealingLR(
    #     optimizer,
    #     T_max=args.epochs,   # 或 T_max=len(train_loader) * args.epochs （按step调整）
    #     eta_min=1e-6         # 最低学习率
    # )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    # resume support
    start_epoch = 0
    best_iou_miou_sum = 0
    if rank == 0 and hasattr(args, 'resume') and args.resume and os.path.isfile(args.resume):
        logger.info(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.module.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if 'torch_amp_scaler' in checkpoint:
            scaler.load_state_dict(checkpoint['torch_amp_scaler'])

        start_epoch = checkpoint.get('epoch', 0)
        best_iou_miou_sum = checkpoint.get('best_iou_miou_sum', 0)
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
    for epoch in range(start_epoch, args.epochs):
        model.train()
        train_sampler.set_epoch(epoch)
        total_loss = 0
        total_mask_loss = 0
        total_dice_loss = 0
        if logger:
            pbar = tqdm(train_loader, disable=not sys.stdout.isatty(), desc=f'Epoch {epoch+1}/{args.epochs}')
        else:
            pbar = train_loader
        for batch_idx, (samples, targets) in enumerate(pbar):
            img = samples['img'].to(device, non_blocking=True)
            word_ids = samples['word_ids'].to(device, non_blocking=True)
            word_masks = samples['word_masks'].to(device, non_blocking=True)
            target = targets['mask'].to(device, non_blocking=True).squeeze(1)
            orig_size = samples['orig_size']
            
            # 清零梯度
            optimizer.zero_grad()
            
            with autocast():
                loss_dict = model(img, word_ids, word_masks, target)
                loss = loss_dict['total_loss']
            # AMP backward + optimizer step
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # 调试：检查梯度是否正常
            if batch_idx == 0 and epoch == start_epoch:
                total_grad_norm = 0
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        param_norm = param.grad.data.norm(2)
                        total_grad_norm += param_norm.item() ** 2
                total_grad_norm = total_grad_norm ** (1. / 2)
                if logger:
                    logger.info(f"Initial gradient norm: {total_grad_norm:.6f}")
            
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
                # logger.info(f"Sample target stats: min={target.min()}, max={target.max()}, mean={target.float().mean()}")
                # logger.info(f"Sample img stats: min={img.min()}, max={img.max()}, mean={img.float().mean()}")
                if (batch_idx + 1) % 10 == 0:
                    logger.info(f"Epoch {epoch+1}/{args.epochs} - Batch {batch_idx+1}/{len(train_loader)} - "
                                f"Loss: {current_loss:.4f} - Mask Loss: {current_mask_loss:.4f} - "
                                f"Dice Loss: {current_dice_loss:.4f}")
        scheduler.step()

        # Validation and checkpointing only on rank 0
        if rank == 0:
            logger.info(f"\nValidating epoch {epoch+1}...")
            metrics = validate(model, val_loader, device)
            logger.info(f"Validation metrics for epoch {epoch+1}:")
            logger.info(f"mIoU: {metrics['mIoU']:.4f}")
            logger.info(f"IoU: {metrics['IoU']:.4f}")
            logger.info(f"pointM: {metrics['pointM']:.4f}")
            logger.info(f"best_IoU: {metrics['best_IoU']:.4f}")
            # Save best iou+miou model
            iou_miou_sum = metrics['IoU'] + metrics['mIoU']
            if iou_miou_sum > best_iou_miou_sum:
                best_iou_miou_sum = iou_miou_sum
                os.makedirs(args.output_dir, exist_ok=True)
                best_path = os.path.join(args.output_dir, 'best_iou_miou_model.pt')
                if os.path.exists(best_path):
                    os.remove(best_path)
                # torch.save({
                #     'epoch': epoch + 1,
                #     'model_state_dict': model.module.state_dict(),
                #     'optimizer_state_dict': optimizer.state_dict(),
                #     'scheduler_state_dict': scheduler.state_dict(),
                #     'metrics': metrics,
                #     'best_iou_miou_sum': best_iou_miou_sum
                # }, best_path)
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'torch_amp_scaler': scaler.state_dict(),
                    'best_iou_miou_sum': best_iou_miou_sum
                }, best_path)

                logger.info(f"Saved new best iou+miou model with score: {best_iou_miou_sum:.4f}")
            os.makedirs(args.output_dir, exist_ok=True)
            current_checkpoint = os.path.join(args.output_dir, f'checkpoint_epoch_{epoch+1}.pt')
            prev_checkpoint = os.path.join(args.output_dir, f'checkpoint_epoch_{epoch}.pt')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'torch_amp_scaler': scaler.state_dict(),
                'best_iou_miou_sum': best_iou_miou_sum
            }, current_checkpoint)
            if os.path.exists(prev_checkpoint):
                os.remove(prev_checkpoint)
            # 打印当前 best_iou_miou_model.pt 的 epoch 和指标
            best_path = os.path.join(args.output_dir, 'best_iou_miou_model.pt')
            if os.path.exists(best_path):
                best_ckpt = torch.load(best_path, map_location='cpu')
                best_epoch = best_ckpt.get('epoch', '-')
                best_metrics = best_ckpt.get('metrics', {})
                logger.info(f"Current best iou+miou sum: {best_ckpt.get('best_iou_miou_sum', '-'):.4f} (epoch {best_epoch})")
                def safe_format(value):
                    return f"{value:.4f}" if isinstance(value, float) else str(value)

                logger.info(
                    f"Best model metrics: "
                    f"mIoU={safe_format(best_metrics.get('mIoU', '-'))}, "
                    f"IoU={safe_format(best_metrics.get('IoU', '-'))}, "
                    f"pointM={safe_format(best_metrics.get('pointM', '-'))}"
                )
    dist.destroy_process_group()

if __name__ == '__main__':
    import torch.multiprocessing as mp
    mp.set_start_method('fork', force=True)
    main() 