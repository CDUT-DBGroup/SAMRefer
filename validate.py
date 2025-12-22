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
from model.enhanced_builder import refersam, load_model_with_checkpoint
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

def set_seed(seed=123456):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

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


def custom_collate_fn(batch):
    """
    自定义collate函数，用于处理包含列表的样本（如all_word_ids, all_word_masks）
    """
    samples_list = [item[0] for item in batch]
    targets_list = [item[1] for item in batch]
    
    # 检查是否有all_word_ids（表示需要特殊处理）
    has_all_sentences = 'all_word_ids' in samples_list[0]
    
    if has_all_sentences:
        # 对于包含所有描述的情况，我们需要特殊处理
        # 将列表字段单独处理，其他字段正常collate
        collated_samples = {}
        collated_targets = {}
        
        # 正常collate的字段
        # 处理img
        if 'img' in samples_list[0]:
            collated_samples['img'] = torch.stack([s['img'] for s in samples_list])
        
        # 处理orig_size（可能是numpy数组）
        if 'orig_size' in samples_list[0]:
            orig_sizes = [s['orig_size'] for s in samples_list]
            if isinstance(orig_sizes[0], np.ndarray):
                collated_samples['orig_size'] = torch.stack([torch.from_numpy(os) for os in orig_sizes])
            elif isinstance(orig_sizes[0], torch.Tensor):
                collated_samples['orig_size'] = torch.stack(orig_sizes)
            else:
                collated_samples['orig_size'] = torch.stack([torch.tensor(os) for os in orig_sizes])
        
        # 处理word_ids和word_masks
        if 'word_ids' in samples_list[0]:
            collated_samples['word_ids'] = torch.stack([s['word_ids'] for s in samples_list])
        if 'word_masks' in samples_list[0]:
            collated_samples['word_masks'] = torch.stack([s['word_masks'] for s in samples_list])
        
        # 处理text（保持为列表）
        if 'text' in samples_list[0]:
            collated_samples['text'] = [s['text'] for s in samples_list]
        
        # 列表字段保持为列表（不collate）
        list_keys = ['all_word_ids', 'all_word_masks', 'all_sentences']
        for key in list_keys:
            if key in samples_list[0]:
                collated_samples[key] = [s[key] for s in samples_list]
        
        # targets正常collate
        for key in targets_list[0].keys():
            if key == 'mask':
                collated_targets[key] = torch.stack([t[key] for t in targets_list])
            elif key == 'orig_size':
                orig_sizes = [t[key] for t in targets_list]
                if isinstance(orig_sizes[0], np.ndarray):
                    collated_targets[key] = torch.stack([torch.from_numpy(os) for os in orig_sizes])
                elif isinstance(orig_sizes[0], torch.Tensor):
                    collated_targets[key] = torch.stack(orig_sizes)
                else:
                    collated_targets[key] = torch.stack([torch.tensor(os) for os in orig_sizes])
            else:
                collated_targets[key] = [t[key] for t in targets_list]
        
        return collated_samples, collated_targets
    else:
        # 使用默认的collate方式
        from torch.utils.data._utils.collate import default_collate
        return default_collate(batch)

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
        # {
        #     'name': 'refcoco',
        #     'class': ReferDataset,
        #     'kwargs': {
        #         'refer_data_root': args.data_root,
        #         'dataset': 'refcoco',
        #         'splitBy': 'unc',
        #         'bert_tokenizer': args.tokenizer_type,
        #         'max_tokens': getattr(args, 'max_tokens', 30),
        #         'split': 'val',
        #         'eval_mode': False,
        #         'size': getattr(args, 'img_size', 320),
        #         'precision': args.precision
        #     }
        # },
        # {
        #     'name': 'refcoco',
        #     'class': ReferDataset,
        #     'kwargs': {
        #         'refer_data_root': args.data_root,
        #         'dataset': 'refcoco',
        #         'splitBy': 'unc',
        #         'bert_tokenizer': args.tokenizer_type,
        #         'max_tokens': getattr(args, 'max_tokens', 30),
        #         'split': 'testA',
        #         'eval_mode': False,
        #         'size': getattr(args, 'img_size', 320),
        #         'precision': args.precision
        #     }
        # },
        #  {
        #     'name': 'refcoco',
        #     'class': ReferDataset,
        #     'kwargs': {
        #         'refer_data_root': args.data_root,
        #         'dataset': 'refcoco',
        #         'splitBy': 'unc',
        #         'bert_tokenizer': args.tokenizer_type,
        #         'max_tokens': getattr(args, 'max_tokens', 30),
        #         'split': 'testB',
        #         'eval_mode': False,
        #         'size': getattr(args, 'img_size', 320),
        #         'precision': args.precision
        #     }
        # },
        # {
        #     'name': 'refcoco+',
        #     'class': ReferDataset,
        #     'kwargs': {
        #         'refer_data_root': args.data_root,
        #         'dataset': 'refcoco+',
        #         'splitBy': 'unc',
        #         'bert_tokenizer': args.tokenizer_type,
        #         'max_tokens': getattr(args, 'max_tokens', 30),
        #         'split': 'val',
        #         'eval_mode': False,
        #         'size': getattr(args, 'img_size', 320),
        #         'precision': args.precision
        #     }
        # },
        # {
        #     'name': 'refcoco+',
        #     'class': ReferDataset,
        #     'kwargs': {
        #         'refer_data_root': args.data_root,
        #         'dataset': 'refcoco+',
        #         'splitBy': 'unc',
        #         'bert_tokenizer': args.tokenizer_type,
        #         'max_tokens': getattr(args, 'max_tokens', 30),
        #         'split': 'testA',
        #         'eval_mode': False,
        #         'size': getattr(args, 'img_size', 320),
        #         'precision': args.precision
        #     }
        # },
        # {
        #     'name': 'refcoco+',
        #     'class': ReferDataset,
        #     'kwargs': {
        #         'refer_data_root': args.data_root,
        #         'dataset': 'refcoco+',
        #         'splitBy': 'unc',
        #         'bert_tokenizer': args.tokenizer_type,
        #         'max_tokens': getattr(args, 'max_tokens', 30),
        #         'split': 'testB',
        #         'eval_mode': False,
        #         'size': getattr(args, 'img_size', 320),
        #         'precision': args.precision
        #     }
        # },
        # {
        #     'name': 'refcocog',
        #     'class': ReferDataset,
        #     'kwargs': {
        #         'refer_data_root': args.data_root,
        #         'dataset': 'refcocog',
        #         'splitBy': 'umd',
        #         'bert_tokenizer': args.tokenizer_type,
        #         'max_tokens': getattr(args, 'max_tokens', 30),
        #         'split': 'val',
        #         'eval_mode': False,
        #         'size': getattr(args, 'img_size', 320),
        #         'precision': args.precision
        #     }
        # },
        # {
        #     'name': 'refcocog',
        #     'class': ReferDataset,
        #     'kwargs': {
        #         'refer_data_root': args.data_root,
        #         'dataset': 'refcocog',
        #         'splitBy': 'umd',
        #         'bert_tokenizer': args.tokenizer_type,
        #         'max_tokens': getattr(args, 'max_tokens', 30),
        #         'split': 'test',
        #         'eval_mode': False,
        #         'size': getattr(args, 'img_size', 320),
        #         'precision': args.precision
        #     }
        # },
        # {
        #     'name': 'referit',
        #     'class': ReferitDataset,
        #     'kwargs': {
        #         'root': args.data_referit_root,
        #         'split': 'val',
        #         'max_tokens': getattr(args, 'max_tokens', 30),
        #         'size': getattr(args, 'img_size', 320)
        #     }
        # },
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
                'eval_mode': False,
                'size': getattr(args, 'img_size', 320),
                'precision': args.precision
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
                'split': 'testA',
                'eval_mode': False,
                'size': getattr(args, 'img_size', 320),
                'precision': args.precision
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
                'split': 'testB',
                'eval_mode': False,
                'size': getattr(args, 'img_size', 320),
                'precision': args.precision
            }
        },
        # {
        #     'name': 'ref-zom',
        #     'class': ReferzomDataset,
        #     'kwargs': {
        #         'refer_data_root': args.data_root,
        #         'dataset': 'ref-zom',
        #         'splitBy': 'final',
        #         'bert_tokenizer': args.tokenizer_type,
        #         'max_tokens': getattr(args, 'max_tokens', 30),
        #         'split': 'test',
        #         'eval_mode': False,
        #         'size': getattr(args, 'img_size', 320),
        #         'precision': args.precision
        #     }
        # }
    ]

    datasets = []
    use_best_sentence = getattr(args, 'use_best_sentence', False)
    for cfg in dataset_configs:
        kwargs = cfg['kwargs'].copy()
        # 如果使用最优描述，为支持的数据集添加return_all_sentences参数
        if use_best_sentence:
            # ReferDataset, GRefDataset, ReferzomDataset都支持
            if cfg['class'] in [ReferDataset, GRefDataset, ReferzomDataset]:
                kwargs['return_all_sentences'] = True
                kwargs['eval_mode'] = True  # 使用最优描述时，必须启用eval_mode
        dataset = cfg['class'](**kwargs)
        # 构建包含split信息的完整名称
        split = kwargs.get('split', 'unknown')
        full_name = f"{cfg['name']}_{split}"
        datasets.append((dataset, full_name))
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
    
    # 使用统一的模型加载函数
    pretrained = hasattr(args, 'pre_train_path') and args.pre_train_path is not None
    loss_config_path = getattr(args, 'loss_config_path', None) if hasattr(args, 'use_enhanced_loss') and args.use_enhanced_loss else None
    
    eval_model, use_fp16, use_bf16, model_engine = load_model_with_checkpoint(
        model_func=refersam,
        pretrained=pretrained,
        args=args,
        loss_config_path=loss_config_path,
        device=device
    )
    
    if model_engine is not None:
        logger.info("Successfully loaded DeepSpeed checkpoint")
    elif pretrained:
        logger.info("Successfully loaded PyTorch checkpoint")
    else:
        logger.warning("No checkpoint path specified, using initialized model")
    
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
    use_best_sentence = getattr(args, 'use_best_sentence', False)
    for dataset, name in datasets:
        logger.info(f"\nStarting validation for {name}...")
        # 如果使用最优描述，需要使用自定义collate函数
        collate_fn = custom_collate_fn if use_best_sentence else None
        val_loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            collate_fn=collate_fn
        )
        # 支持通过args传递use_negative_masks、use_best_sentence和sentence_aggregation参数
        use_negative_masks = getattr(args, 'use_negative_masks', False)
        sentence_aggregation = getattr(args, 'sentence_aggregation', 'mean')  # 默认使用平均方式（更公平）
        metrics = validate(eval_model, val_loader, device, use_fp16=use_fp16, use_bf16=use_bf16, 
                          use_negative_masks=use_negative_masks, use_best_sentence=use_best_sentence,
                          sentence_aggregation=sentence_aggregation)
        logger.info(f"\nValidation Results for {name} (use_negative_masks={use_negative_masks}, use_best_sentence={use_best_sentence}, aggregation={sentence_aggregation}):")
        logger.info(f"mIoU: {metrics['mIoU']:.4f}")
        logger.info(f"oIoU: {metrics['oIoU']:.4f}")
        logger.info(f"gIoU: {metrics['gIoU']:.4f}")
        logger.info(f"cIoU: {metrics['cIoU']:.4f}")
        logger.info(f"Acc: {metrics['Acc']:.4f}")
        logger.info(f"pointM: {metrics['pointM']:.4f}")
        logger.info(f"best_IoU: {metrics['best_IoU']:.4f}")

if __name__ == '__main__':
    evaluate_four_datasets()