import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import numpy as np
from dataset.Dataset_referit import ReferitDataset
from dataset.ReferDataset import ReferDataset
from model.builder import refersam
from model.segment_anything.build_sam import sam_model_registry
from validation.evaluation import validate
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
from get_args import get_args
# os.environ["CUDA_VISIBLE_DEVICES"]="1"

import random
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

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



def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def log_sample_info(dataset, name, num_samples=2):
    logger.info(f"===== Inspecting {name} dataset, total {len(dataset)} samples =====")
    for i in range(min(num_samples, len(dataset))):
        sample, target = dataset[i]
        logger.info(f"[{name}][{i}] img shape: {getattr(sample['img'], 'shape', type(sample['img']))}")
        logger.info(f"[{name}][{i}] orig_size: {sample.get('orig_size', None)}")
        logger.info(f"[{name}][{i}] text: {sample.get('text', None)}")
        logger.info(f"[{name}][{i}] word_ids: {sample.get('word_ids', None)}")
        logger.info(f"[{name}][{i}] word_masks: {sample.get('word_masks', None)}")
        logger.info(f"[{name}][{i}] mask shape: {getattr(target['mask'], 'shape', type(target['mask']))}")
        logger.info(f"[{name}][{i}] img_path: {target.get('img_path', None)}")
        logger.info(f"[{name}][{i}] sentences: {target.get('sentences', None)}")
        logger.info(f"[{name}][{i}] boxes: {target.get('boxes', None)}")
        logger.info(f"[{name}][{i}] orig_size: {target.get('orig_size', None)}")
        logger.info(f"[{name}][{i}] img_full_path: {target.get('img_full_path', None)}")
        logger.info("-")

def create_datasets(args):
    """
    创建并返回需要验证的数据集和其名称组成的列表
    """
    dataset_configs = [
        {
            'name': 'refcoco',
            'class': ReferDataset,
            'kwargs': {
                'refer_data_root': args.data_root,
                'dataset': 'refcoco',
                'splitBy': 'unc',
                'bert_tokenizer': args.tokenizer_type,
                'max_tokens': getattr(args, 'max_tokens', 30),
                'split': 'val',
                'eval_mode': True,
                'size': getattr(args, 'img_size', 320),
                'precision': args.precision
            }
        },
        {
            'name': 'refcoco',
            'class': ReferDataset,
            'kwargs': {
                'refer_data_root': args.data_root,
                'dataset': 'refcoco',
                'splitBy': 'unc',
                'bert_tokenizer': args.tokenizer_type,
                'max_tokens': getattr(args, 'max_tokens', 30),
                'split': 'testA',
                'eval_mode': True,
                'size': getattr(args, 'img_size', 320),
                'precision': args.precision
            }
        },
         {
            'name': 'refcoco',
            'class': ReferDataset,
            'kwargs': {
                'refer_data_root': args.data_root,
                'dataset': 'refcoco',
                'splitBy': 'unc',
                'bert_tokenizer': args.tokenizer_type,
                'max_tokens': getattr(args, 'max_tokens', 30),
                'split': 'testB',
                'eval_mode': True,
                'size': getattr(args, 'img_size', 320),
                'precision': args.precision
            }
        },
        {
            'name': 'refcoco+',
            'class': ReferDataset,
            'kwargs': {
                'refer_data_root': args.data_root,
                'dataset': 'refcoco+',
                'splitBy': 'unc',
                'bert_tokenizer': args.tokenizer_type,
                'max_tokens': getattr(args, 'max_tokens', 30),
                'split': 'val',
                'eval_mode': True,
                'size': getattr(args, 'img_size', 320),
                'precision': args.precision
            }
        },
        {
            'name': 'refcoco+',
            'class': ReferDataset,
            'kwargs': {
                'refer_data_root': args.data_root,
                'dataset': 'refcoco+',
                'splitBy': 'unc',
                'bert_tokenizer': args.tokenizer_type,
                'max_tokens': getattr(args, 'max_tokens', 30),
                'split': 'testA',
                'eval_mode': True,
                'size': getattr(args, 'img_size', 320),
                'precision': args.precision
            }
        },
        {
            'name': 'refcoco+',
            'class': ReferDataset,
            'kwargs': {
                'refer_data_root': args.data_root,
                'dataset': 'refcoco+',
                'splitBy': 'unc',
                'bert_tokenizer': args.tokenizer_type,
                'max_tokens': getattr(args, 'max_tokens', 30),
                'split': 'testB',
                'eval_mode': True,
                'size': getattr(args, 'img_size', 320),
                'precision': args.precision
            }
        },
        {
            'name': 'refcocog',
            'class': ReferDataset,
            'kwargs': {
                'refer_data_root': args.data_root,
                'dataset': 'refcocog',
                'splitBy': 'umd',
                'bert_tokenizer': args.tokenizer_type,
                'max_tokens': getattr(args, 'max_tokens', 30),
                'split': 'val',
                'eval_mode': True,
                'size': getattr(args, 'img_size', 320),
                'precision': args.precision
            }
        },
        {
            'name': 'refcocog',
            'class': ReferDataset,
            'kwargs': {
                'refer_data_root': args.data_root,
                'dataset': 'refcocog',
                'splitBy': 'umd',
                'bert_tokenizer': args.tokenizer_type,
                'max_tokens': getattr(args, 'max_tokens', 30),
                'split': 'test',
                'eval_mode': True,
                'size': getattr(args, 'img_size', 320),
                'precision': args.precision
            }
        },
        {
            'name': 'referit',
            'class': ReferitDataset,
            'kwargs': {
                'root': args.data_referit_root,
                'split': 'val',
                'max_tokens': getattr(args, 'max_tokens', 30),
                'size': getattr(args, 'img_size', 320)
            }
        },
        {
            'name': 'grefcoco',
            'class': GRefDataset,
            'kwargs': {
                'refer_data_root': args.data_root,
                'dataset': 'grefcoco',
                'splitBy': 'unc',
                'bert_tokenizer': args.tokenizer_type,
                'max_tokens': getattr(args, 'max_tokens', 30),
                'split': 'val',
                'eval_mode': True,
                'size': getattr(args, 'img_size', 320),
                'precision': args.precision
            }
        },
        {
            'name': 'ref-zom',
            'class': ReferzomDataset,
            'kwargs': {
                'refer_data_root': args.data_root,
                'dataset': 'ref-zom',
                'splitBy': 'final',
                'bert_tokenizer': args.tokenizer_type,
                'max_tokens': getattr(args, 'max_tokens', 30),
                'split': 'val',
                'eval_mode': True,
                'size': getattr(args, 'img_size', 320),
                'precision': args.precision
            }
        }
    ]

    datasets = []
    for cfg in dataset_configs:
        dataset = cfg['class'](**cfg['kwargs'])
        datasets.append((dataset, cfg['name']))
    return datasets



def evaluate_four_datasets():
    # Fixed arguments for BERT configuration
    args = get_args()
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Set device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Initialize models and criterion
    logger.info("Initializing models...")
    model = refersam(args=args, pretrained=True)
    # Load trained model weights
    logger.info(f"Loading model weights from {args.pre_train_path}")
    model.eval()  # Set model to evaluation mode

    # Print model parameters
    total_params, trainable_params = count_parameters(model)
    logger.info(f"\nModel Parameters:")
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    logger.info(f"Non-trainable parameters: {total_params - trainable_params:,}")

    # Create validation datasets
    logger.info("Creating validation datasets...")
    # 创建所有数据集
    datasets = create_datasets(args)

    # 打印每个数据集前2个样本
    for dataset, name in datasets:
        log_sample_info(dataset, name)

    # 验证每个数据集
    for dataset, name in datasets:
        logger.info(f"\nStarting validation for {name}...")
        val_loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )
        # 支持通过args传递use_negative_masks参数
        use_negative_masks = getattr(args, 'use_negative_masks', False)
        metrics = validate(model, val_loader, device, use_negative_masks=use_negative_masks)
        logger.info(f"\nValidation Results for {name} (use_negative_masks={use_negative_masks}):")
        logger.info(f"mIoU: {metrics['mIoU']:.4f}")
        logger.info(f"oIoU: {metrics['oIoU']:.4f}")
        logger.info(f"gIoU: {metrics['gIoU']:.4f}")
        logger.info(f"Acc: {metrics['Acc']:.4f}")
        logger.info(f"pointM: {metrics['pointM']:.4f}")
        logger.info(f"best_IoU: {metrics['best_IoU']:.4f}")

if __name__ == '__main__':
    evaluate_four_datasets()