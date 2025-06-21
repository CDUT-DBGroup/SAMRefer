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
from model.builder import refersam
from model.models.refersam import ReferSAM
from model.segment_anything.build_sam import sam_model_registry
from model.criterion import SegMaskLoss
from evaluation import validate
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
from get_args import get_args
# os.environ["CUDA_VISIBLE_DEVICES"]="1"

import random
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed()

def visualize_prediction(image_tensor, pred_mask, gt_mask, idx, save_dir='visual_results'):
    """
    保存预测图、真实掩码、原图组合可视化。
    """
    os.makedirs(save_dir, exist_ok=True)

    # 去归一化图像
    image = image_tensor.cpu().clone()
    image = TF.normalize(image, mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                         std=[1 / 0.229, 1 / 0.224, 1 / 0.225])
    image = image[0].permute(1, 2, 0).numpy().clip(0, 1)

    pred = pred_mask[0].squeeze().cpu().numpy()
    gt = gt_mask[0].squeeze().cpu().numpy()

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.title('Image')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(pred, cmap='jet')
    plt.title('Predicted Mask')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(gt, cmap='gray')
    plt.title('Ground Truth Mask')
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'vis_{idx}.png'))
    plt.close()

def visualize_prediction_batch(image_tensor, pred_mask, idx_start=0, save_dir='visual_results'):
    """
    批量保存预测掩码和原图的可视化图像（不包含 Ground Truth）。
    
    参数:
    - image_tensor: torch.Tensor, 形状为 (B, 3, H, W)
    - pred_mask: torch.Tensor, 形状为 (B, H, W)
    - idx_start: int, 用于命名起始编号
    - save_dir: str, 保存路径
    """
    os.makedirs(save_dir, exist_ok=True)

    image_tensor = image_tensor.cpu()
    pred_mask = pred_mask.cpu()

    for i in range(image_tensor.shape[0]):
        # 去归一化图像
        img = TF.normalize(
            image_tensor[i],
            mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
            std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
        ).permute(1, 2, 0).numpy().clip(0, 1)

        pred = pred_mask[i].numpy()

        plt.figure(figsize=(8, 4))
        plt.subplot(1, 2, 1)
        plt.imshow(img)
        plt.title('Image')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(pred, cmap='jet')
        plt.title('Predicted Mask')
        plt.axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'vis_{idx_start + i}.png'))
        plt.close()


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
    args = get_args()
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Set device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize models and criterion
    print("Initializing models...")
    model = refersam(args=args, pretrained=True)
    # Load trained model weights
    print(f"Loading model weights from {args.pre_train_path}")
    model.eval()  # Set model to evaluation mode

    # Print model parameters
    total_params, trainable_params = count_parameters(model)
    print(f"\nModel Parameters:")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {total_params - trainable_params:,}")

    # Define image transforms

    # Create validation dataset
    print("Creating validation dataset...")
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

    # Create validation data loader
    print("Creating validation data loader...")
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    # Run validation
    print("\nStarting validation...")
    metrics = validate(model, val_loader, device)
    
    # Print validation results
    print("\nValidation Results:")
    print(f"mIoU: {metrics['mIoU']:.4f}")
    print(f"IoU: {metrics['IoU']:.4f}")
    print(f"pointM: {metrics['pointM']:.4f}")
    print(f"best_cIoU: {metrics['best_cIoU']:.4f}")
    print(f"best_gIoU: {metrics['best_gIoU']:.4f}")
    # 可视化前两张预测结果
    print("\nGenerating visualization for first 2 samples...")
    model.eval()
    with torch.no_grad():
        vis_count = 0
        for batch_idx, batch in enumerate(val_loader):
            images = batch[batch_idx]['img'].to(device)
            word_ids = batch[batch_idx]['word_ids'].to(device, non_blocking=True)
            word_masks = batch[batch_idx]['word_masks'].to(device)
            texts = batch[batch_idx]['text']

            preds = model(images, word_ids, word_masks)
            pred_masks = (preds > 0.5).float()
            visualize_prediction_batch(images, pred_masks)

if __name__ == '__main__':
    main() 