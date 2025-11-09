import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import numpy as np
from dataset.Dataset_referit import ReferitDataset
from dataset.GRefDataset import GRefDataset
from dataset.ReferDataset import ReferDataset
from dataset.RefzomDataset import ReferzomDataset
from model.enhanced_builder import refersam
from model.segment_anything.build_sam import sam_model_registry
from validation.evaluation import validate
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
from get_args import get_args
# os.environ["CUDA_VISIBLE_DEVICES"]="1"
import glob
import random
import logging
import json
import deepspeed

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
                'split': 'test',
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


def is_deepspeed_checkpoint(checkpoint_path):
    """
    检测检查点路径是否是 DeepSpeed 格式
    DeepSpeed 检查点通常包含 global_step 子目录或 mp_rank_00 等目录
    """
    if not os.path.exists(checkpoint_path):
        return False
    
    if os.path.isdir(checkpoint_path):
        # 检查是否有 global_step 子目录
        subdirs = glob.glob(os.path.join(checkpoint_path, "global_step*"))
        if subdirs:
            # 检查 global_step 子目录中是否有 DeepSpeed 检查点文件
            latest_subdir = max(subdirs, key=os.path.getmtime)
            # DeepSpeed 检查点通常包含 mp_rank_00 目录或 model_states.pt 等文件
            if os.path.exists(os.path.join(latest_subdir, "mp_rank_00")) or \
               glob.glob(os.path.join(latest_subdir, "*.pt")):
                return True
        # 检查根目录是否有 mp_rank_00 目录
        if os.path.exists(os.path.join(checkpoint_path, "mp_rank_00")):
            return True
    return False


def load_deepspeed_checkpoint(checkpoint_path, model_engine):
    """
    加载 DeepSpeed 检查点
    DeepSpeed 的 load_checkpoint 会自动查找最新的 global_step 子目录
    """
    # 记录检查点路径信息
    if os.path.isdir(checkpoint_path):
        subdirs = glob.glob(os.path.join(checkpoint_path, "global_step*"))
        if subdirs:
            latest_subdir = max(subdirs, key=os.path.getmtime)
            logger.info(f"Found DeepSpeed checkpoint directory with global_step subdirectories")
            logger.info(f"Loading from parent directory: {checkpoint_path}")
            logger.info(f"Latest global_step: {os.path.basename(latest_subdir)}")
        else:
            logger.info(f"Loading DeepSpeed checkpoint from: {checkpoint_path}")
    else:
        logger.info(f"Loading DeepSpeed checkpoint from: {checkpoint_path}")
    
    # DeepSpeed 的 load_checkpoint 会自动查找最新的 global_step，所以直接传父目录
    _, client_state = model_engine.load_checkpoint(checkpoint_path)
    logger.info(f"Successfully loaded DeepSpeed checkpoint: {checkpoint_path}")
    if client_state:
        logger.info(f"Client state: {client_state}")
    return client_state


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
    model = refersam(args=args, pretrained=False)
    model = model.to(device)
    
    # 检测是否是 DeepSpeed 检查点
    use_deepspeed = False
    model_engine = None
    use_fp16 = False
    use_bf16 = False
    
    if hasattr(args, 'pre_train_path') and args.pre_train_path:
        if is_deepspeed_checkpoint(args.pre_train_path):
            use_deepspeed = True
            logger.info("Detected DeepSpeed checkpoint format, initializing DeepSpeed engine...")
            
            # 读取 DeepSpeed 配置
            ds_config = getattr(args, 'deepspeed_config', 'configs/ds_config.json')
            if not os.path.exists(ds_config):
                # 尝试其他可能的路径
                ds_config = 'ds_config.json'
            if os.path.exists(ds_config):
                with open(ds_config) as f:
                    ds_conf = json.load(f)
                use_fp16 = ds_conf.get("fp16", {}).get("enabled", False)
                use_bf16 = ds_conf.get("bf16", {}).get("enabled", False)
                logger.info(f"DeepSpeed config: fp16={use_fp16}, bf16={use_bf16}")
            else:
                logger.warning(f"DeepSpeed config file not found at {ds_config}, using default settings")
            
            # 初始化 DeepSpeed 引擎
            if hasattr(model, 'params_to_optimize'):
                param_groups = model.params_to_optimize()
            else:
                param_groups = model.parameters()
            
            model_engine, optimizer, _, _ = deepspeed.initialize(
                model=model,
                model_parameters=param_groups,
                config=ds_config if os.path.exists(ds_config) else None
            )
            
            # 加载 DeepSpeed 检查点
            logger.info(f"Loading DeepSpeed checkpoint from {args.pre_train_path}")
            load_deepspeed_checkpoint(args.pre_train_path, model_engine)
            model_engine.eval()
        else:
            # 使用普通 PyTorch 检查点加载方式
            logger.info("Using standard PyTorch checkpoint loading...")
            if os.path.isdir(args.pre_train_path):
                # 查找所有 global_step 子目录
                subdirs = glob.glob(os.path.join(args.pre_train_path, "global_step*"))
                if subdirs:
                    # 取最新的 global_step 子目录
                    latest_subdir = max(subdirs, key=os.path.getmtime)
                    logger.info(f"Resuming from checkpoint: {latest_subdir}")
                    checkpoint = torch.load(latest_subdir, map_location=device)
                    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                        model.load_state_dict(checkpoint['model_state_dict'])
                    else:
                        model.load_state_dict(checkpoint)
                else:
                    logger.info(f"Resuming from checkpoint: {args.pre_train_path}")
                    checkpoint = torch.load(args.pre_train_path, map_location=device)
                    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                        model.load_state_dict(checkpoint['model_state_dict'])
                    else:
                        model.load_state_dict(checkpoint)
            else:
                logger.info(f"Loading model weights from {args.pre_train_path}")
                checkpoint = torch.load(args.pre_train_path, map_location=device)
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    model.load_state_dict(checkpoint)
            model.eval()  # Set model to evaluation mode
    else:
        logger.warning("No checkpoint path specified, using initialized model")
        model.eval()
    
    # 获取实际用于验证的模型
    eval_model = model_engine.module if use_deepspeed else model
    
    # Print model parameters
    total_params, trainable_params = count_parameters(eval_model)
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
        metrics = validate(eval_model, val_loader, device, use_fp16=use_fp16, use_bf16=use_bf16, use_negative_masks=use_negative_masks)
        logger.info(f"\nValidation Results for {name} (use_negative_masks={use_negative_masks}):")
        logger.info(f"mIoU: {metrics['mIoU']:.4f}")
        logger.info(f"oIoU: {metrics['oIoU']:.4f}")
        logger.info(f"gIoU: {metrics['gIoU']:.4f}")
        logger.info(f"Acc: {metrics['Acc']:.4f}")
        logger.info(f"pointM: {metrics['pointM']:.4f}")
        logger.info(f"best_IoU: {metrics['best_IoU']:.4f}")

if __name__ == '__main__':
    evaluate_four_datasets()