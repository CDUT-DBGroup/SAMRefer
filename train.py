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
from transformers import CLIPTextModel, CLIPTokenizer
from dataset.ReferDataset import ReferDataset
from model.models.refersam import ReferSAM
from model.segment_anything.build_sam import sam_model_registry
from model.criterion import SegMaskLoss
import deepspeed
from deepspeed.ops.adam import DeepSpeedCPUAdam

def get_args_parser():
    parser = argparse.ArgumentParser('ReferSAM training script', add_help=False)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--epochs', default=40, type=int)
    parser.add_argument('--lr', default=2e-5, type=float)
    parser.add_argument('--weight_decay', default=0.01, type=float)
    parser.add_argument('--data_root', default='data', type=str)
    parser.add_argument('--output_dir', default='output', type=str)
    parser.add_argument('--model_type', default='vit_b', type=str, choices=['vit_b', 'vit_h', 'vit_l'])
    parser.add_argument('--checkpoint', default=None, type=str)
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--deepspeed_config', default='ds_config.json', type=str)
    parser.add_argument('--tokenizer_type', default='clip', type=str, choices=['clip', 'bert'])
    parser.add_argument('--precision', default='fp32', type=str, choices=['fp32', 'fp16', 'bf16'])
    parser.add_argument('--clip_path', default=None, type=str)
    parser.add_argument('--ck_bert', default='bert-base-uncased', type=str)
    return parser

def main(args):
    # Initialize deepspeed
    deepspeed.init_distributed()

    # Create output directory
    if args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir, exist_ok=True)

    # Initialize SAM model
    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    
    # Initialize text encoder
    if args.clip_path:
        text_model = CLIPTextModel.from_pretrained(args.clip_path)
    else:
        text_model = BertModel.from_pretrained(args.ck_bert)
    
    # Initialize criterion
    criterion = SegMaskLoss(num_points=112*112, oversample_ratio=3.0, importance_sample_ratio=0.75)

    # Create model
    model = ReferSAM(
        sam_model=sam,
        text_encoder=text_model,
        args=args,
        num_classes=1,
        criterion=criterion
    )

    # Define image transforms
    image_transforms = transforms.Compose([
        transforms.Resize((480, 480)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create datasets
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

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    # Initialize optimizer
    optimizer = DeepSpeedCPUAdam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # Initialize deepspeed
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        config=args.deepspeed_config
    )

    # Training loop
    for epoch in range(args.epochs):
        model_engine.train()
        total_loss = 0
        
        with tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.epochs}') as pbar:
            for samples, targets in pbar:
                # Move data to device
                img = samples['img'].to(model_engine.device)
                word_ids = samples['word_ids'].to(model_engine.device)
                word_masks = samples['word_masks'].to(model_engine.device)
                target = targets['mask'].to(model_engine.device)

                # Forward pass
                loss_dict = model_engine(img, word_ids, word_masks, target)
                loss = loss_dict['total_loss']

                # Backward pass
                model_engine.backward(loss)
                model_engine.step()

                # Update progress bar
                total_loss += loss.item()
                pbar.set_postfix({
                    'loss': total_loss / (pbar.n + 1),
                    'mask_loss': loss_dict['loss_mask'].item(),
                    'dice_loss': loss_dict['loss_dice'].item()
                })

        # Save checkpoint
        if args.local_rank in [-1, 0]:
            checkpoint_path = os.path.join(args.output_dir, f'checkpoint_epoch_{epoch+1}.pt')
            model_engine.save_checkpoint(args.output_dir, tag=f'epoch_{epoch+1}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser('ReferSAM training script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args) 