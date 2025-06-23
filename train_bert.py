import os
import sys
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from dataset.ReferDataset import ReferDataset
from get_args import get_args
from model.builder import refersam
from evaluation import validate
import logging
import datetime
import random

def set_seed(seed=123456):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed()
# Configure logging
def setup_logger(output_dir):
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
    # Fixed arguments for BERT configuration
    args = get_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup logger
    logger = setup_logger(args.output_dir)
    logger.info("Starting training with configuration:")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Learning rate: {args.lr}")
    logger.info(f"Weight decay: {args.weight_decay}")

    # Set device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    logger.info("Creating ReferSAM model...")
    model = refersam(args=args)
    
    # Print model parameters
    total_params, trainable_params = count_parameters(model)
    logger.info(f"\nModel Parameters:")
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    logger.info(f"Non-trainable parameters: {total_params - trainable_params:,}")
    
    model = model.to(device)

    # Create datasets
    logger.info("Creating datasets...")
    train_dataset = ReferDataset(
        refer_data_root=args.data_root,
        dataset='refcoco',
        splitBy='unc',
        bert_tokenizer=args.tokenizer_type,
        max_tokens=30,
        split='train',
        eval_mode=False,
        size=320,
        precision=args.precision
    )
    
    val_dataset = ReferDataset(
        refer_data_root=args.data_root,
        dataset='refcoco',
        splitBy='unc',
        bert_tokenizer=args.tokenizer_type,
        max_tokens=30,
        split='val',
        eval_mode=True,
        size=320,
        precision=args.precision
    )

    # Create data loaders
    logger.info("Creating data loaders...")
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    # Initialize optimizer
    logger.info("Initializing optimizer...")
    optimizer = optim.Adam(
        model.parameters(),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay)
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    # Initialize training state
    start_epoch = 0
    best_giou = 0
    best_iou_miou_sum = 0  # 新增
    
    # Check for resume training
    if hasattr(args, 'resume') and args.resume:
        checkpoint_path = os.path.join(args.output_dir, 'best_giou_model.pt')
        if os.path.exists(checkpoint_path):
            logger.info(f"Loading checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch']
            best_giou = checkpoint['metrics']['best_gIoU']
            logger.info(f"Resuming from epoch {start_epoch} with best gIoU: {best_giou:.4f}")
        else:
            logger.warning(f"Resume flag is set but checkpoint not found at {checkpoint_path}")
            logger.info("Starting training from scratch")

    # Training loop
    if start_epoch == 0:
        logger.info("Starting training from epoch 1...")
    else:
        logger.info(f"Resuming training from epoch {start_epoch + 1}...")
    
    for epoch in range(start_epoch, args.epochs):
        model.train()
        total_loss = 0
        total_mask_loss = 0
        total_dice_loss = 0
        
        with tqdm(train_loader, disable=not sys.stdout.isatty(),desc=f'Epoch {epoch+1}/{args.epochs}') as pbar:
            for batch_idx, (samples, targets) in enumerate(pbar):
                img = samples['img'].to(device, non_blocking=True)         # tensor
                word_ids = samples['word_ids'].to(device, non_blocking=True)
                word_masks = samples['word_masks'].to(device, non_blocking=True)
                target = targets['mask'].to(device, non_blocking=True).squeeze(1)     # tensor
                orig_size = samples['orig_size']                           # numpy array, shape (B, 2)

                # Forward pass
                loss_dict = model(img, word_ids, word_masks, target)
                loss = loss_dict['total_loss']

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Update progress bar
                total_loss += loss.item()
                total_mask_loss += loss_dict['loss_mask'].item()
                total_dice_loss += loss_dict['loss_dice'].item()

                current_loss = total_loss / (batch_idx + 1)
                current_mask_loss = total_mask_loss / (batch_idx + 1)
                current_dice_loss = total_dice_loss / (batch_idx + 1)

                pbar.set_postfix({
                    'loss': current_loss,
                    'mask_loss': current_mask_loss,
                    'dice_loss': current_dice_loss
                })
                # logger.info(f"Sample target stats: min={target.min()}, max={target.max()}, mean={target.float().mean()}")
                # logger.info(f"Sample img stats: min={img.min()}, max={img.max()}, mean={img.float().mean()}")


                if (batch_idx + 1) % 10 == 0:  # 每10个batch记录一次
                    logger.info(f"Epoch {epoch+1}/{args.epochs} - Batch {batch_idx+1}/{len(train_loader)} - "
                                f"Loss: {current_loss:.4f} - Mask Loss: {current_mask_loss:.4f} - "
                                f"Dice Loss: {current_dice_loss:.4f}")
                # Clear memory
                # del img, word_ids, word_masks, target, loss_dict, loss
                # torch.cuda.empty_cache()
        scheduler.step()

        # Validation
        logger.info(f"\nValidating epoch {epoch+1}...")
        metrics = validate(model, val_loader, device)
        logger.info(f"Validation metrics for epoch {epoch+1}:")
        logger.info(f"mIoU: {metrics['mIoU']:.4f}")
        logger.info(f"IoU: {metrics['IoU']:.4f}")
        logger.info(f"pointM: {metrics['pointM']:.4f}")
        logger.info(f"best_cIoU: {metrics['best_cIoU']:.4f}")
        logger.info(f"best_gIoU: {metrics['best_gIoU']:.4f}")

        # Save best model based on gIoU only
        if metrics['best_gIoU'] > best_giou:
            best_giou = metrics['best_gIoU']
            best_giou_path = os.path.join(args.output_dir, 'best_giou_model.pt')
            # Delete previous best gIoU model if exists
            if os.path.exists(best_giou_path):
                os.remove(best_giou_path)
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': metrics
            }, best_giou_path)
            logger.info(f"Epoch {epoch+1}/{args.epochs}: Saved new best gIoU model with score: {best_giou:.4f}")
        else:
            logger.info(f"Epoch {epoch+1}/{args.epochs}: No improvement in gIoU. Best gIoU remains: {best_giou:.4f}")

        # 新增：保存 IoU + mIoU 最大的模型
    iou_miou_sum = metrics['IoU'] + metrics['mIoU']
    if iou_miou_sum > best_iou_miou_sum:
        best_iou_miou_sum = iou_miou_sum
        best_iou_miou_path = os.path.join(args.output_dir, 'best_iou_miou_model.pt')
        if os.path.exists(best_iou_miou_path):
            os.remove(best_iou_miou_path)
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics
        }, best_iou_miou_path)
        logger.info(f"Epoch {epoch+1}/{args.epochs}: Saved new best (IoU + mIoU) model with sum: {best_iou_miou_sum:.4f}")
    else:
        logger.info(f"Epoch {epoch+1}/{args.epochs}: No improvement in (IoU + mIoU). Best sum remains: {best_iou_miou_sum:.4f}")

        # Delete previous epoch checkpoint (no longer saving regular checkpoints)
        # prev_checkpoint = os.path.join(args.output_dir, f'checkpoint_epoch_{epoch}.pt')
        # if os.path.exists(prev_checkpoint):
        #     os.remove(prev_checkpoint)

if __name__ == '__main__':
    main() 