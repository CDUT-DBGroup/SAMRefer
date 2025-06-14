import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import argparse
from tqdm import tqdm
import numpy as np
from transformers import BertTokenizer, BertModel
from dataset.ReferDataset import ReferDataset
from model.models.refersam import ReferSAM
from model.segment_anything.build_sam import sam_model_registry
from model.criterion import SegMaskLoss
from evaluation import validate

def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

class ImageMaskTransform:
    def __init__(self, size):
        self.size = size
        self.image_transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.mask_transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor()
        ])

    def __call__(self, img, mask):
        img = self.image_transform(img)
        mask = self.mask_transform(mask)
        return img, mask

def main():
    # Fixed arguments for BERT configuration
    args = argparse.Namespace(
        batch_size=8,
        epochs=40,
        lr=2e-5,
        weight_decay=0.01,
        data_root='/root/autodl-tmp/paper_data/coco_data',
        output_dir='output/refersam_bert',
        model_type='vit_b',
        checkpoint='/root/autodl-tmp/paper_data/weight/sam/sam_vit_b_01ec64.pth',
        tokenizer_type='bert',
        precision='fp32',
        clip_path=None,
        ck_bert='bert-base-uncased'
    )

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize models and criterion
    print("Initializing models...")
    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    text_model = BertModel.from_pretrained(args.ck_bert)
    criterion = SegMaskLoss(num_points=112*112, oversample_ratio=3.0, importance_sample_ratio=0.75)

    # Create model
    print("Creating ReferSAM model...")
    model = ReferSAM(
        sam_model=sam,
        text_encoder=text_model,
        args=args,
        num_classes=1,
        criterion=criterion
    )
    
    # Print model parameters
    total_params, trainable_params = count_parameters(model)
    print(f"\nModel Parameters:")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {total_params - trainable_params:,}")
    
    model = model.to(device)

    # Define image transforms
    image_transforms = ImageMaskTransform(size=480)

    # Create datasets
    print("Creating datasets...")
    train_dataset = ReferDataset(
        refer_data_root=args.data_root,
        dataset='refcoco',
        splitBy='unc',
        bert_tokenizer=args.tokenizer_type,
        image_transforms=image_transforms,
        max_tokens=30,
        split='train',
        eval_mode=False,
        size=480,
        precision=args.precision
    )
    
    val_dataset = ReferDataset(
        refer_data_root=args.data_root,
        dataset='refcoco',
        splitBy='unc',
        bert_tokenizer=args.tokenizer_type,
        image_transforms=image_transforms,
        max_tokens=30,
        split='val',
        eval_mode=True,
        size=480,
        precision=args.precision
    )

    # Create data loaders
    print("Creating data loaders...")
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
    print("Initializing optimizer...")
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # Initialize best metrics
    best_ciou = 0
    best_giou = 0

    # Training loop
    print("Starting training...")
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        
        with tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.epochs}') as pbar:
            for samples, targets in pbar:
                # Move data to device
                img = samples['img'].to(device, non_blocking=True)
                word_ids = samples['word_ids'].to(device, non_blocking=True)
                word_masks = samples['word_masks'].to(device, non_blocking=True)
                target = targets['mask'].to(device, non_blocking=True).squeeze(1)

                # Forward pass
                loss_dict = model(img, word_ids, word_masks, target)
                loss = loss_dict['total_loss']

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Update progress bar
                total_loss += loss.item()
                pbar.set_postfix({
                    'loss': total_loss / (pbar.n + 1),
                    'mask_loss': loss_dict['loss_mask'].item(),
                    'dice_loss': loss_dict['loss_dice'].item()
                })

                # Clear memory
                del img, word_ids, word_masks, target, loss_dict, loss
                torch.cuda.empty_cache()

        # Validation
        print(f"\nValidating epoch {epoch+1}...")
        metrics = validate(model, val_loader, device)
        print(f"Validation metrics:")
        print(f"mIoU: {metrics['mIoU']:.4f}")
        print(f"IoU: {metrics['IoU']:.4f}")
        print(f"pointM: {metrics['pointM']:.4f}")
        print(f"best_cIoU: {metrics['best_cIoU']:.4f}")
        print(f"best_gIoU: {metrics['best_gIoU']:.4f}")

        # Save checkpoint
        print(f"Saving checkpoint for epoch {epoch+1}...")
        checkpoint_path = os.path.join(args.output_dir, f'checkpoint_epoch_{epoch+1}.pt')
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': total_loss / len(train_loader),
            'metrics': metrics
        }, checkpoint_path)

        # Save best model based on cIoU and gIoU
        if metrics['best_cIoU'] > best_ciou:
            best_ciou = metrics['best_cIoU']
            best_ciou_path = os.path.join(args.output_dir, 'best_ciou_model.pt')
            # Delete previous best cIoU model if exists
            if os.path.exists(best_ciou_path):
                os.remove(best_ciou_path)
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': metrics
            }, best_ciou_path)
            print(f"Saved new best cIoU model with score: {best_ciou:.4f}")

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
            print(f"Saved new best gIoU model with score: {best_giou:.4f}")

        # Delete previous epoch checkpoint
        prev_checkpoint = os.path.join(args.output_dir, f'checkpoint_epoch_{epoch}.pt')
        if os.path.exists(prev_checkpoint):
            os.remove(prev_checkpoint)

if __name__ == '__main__':
    main() 