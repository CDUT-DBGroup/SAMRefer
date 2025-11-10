import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from dataset.RefzomDataset import ReferzomDataset
from dataset.GRefDataset import GRefDataset
from model.enhanced_builder import refersam, load_model_with_checkpoint
from validation.evaluation import validate
from get_args import get_args
import random
import logging
import json

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

def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def log_sample_info(dataset, name, num_samples=2):
    logger.info(f"===== 检查{name}数据集, 共{len(dataset)}条 =====")
    for i in range(min(num_samples, len(dataset))):
        sample, target = dataset[i]
        logger.info(f"[{name}][{i}] img shape: {getattr(sample['img'], 'shape', type(sample['img']))}")
        logger.info(f"[{name}][{i}] orig_size: {sample.get('orig_size', None)}")
        logger.info(f"[{name}][{i}] word_ids: {sample.get('word_ids', None)}")
        logger.info(f"[{name}][{i}] word_masks: {sample.get('word_masks', None)}")
        logger.info(f"[{name}][{i}] mask shape: {getattr(target['mask'], 'shape', type(target['mask']))}")
        logger.info(f"[{name}][{i}] img_path: {target.get('img_path', None)}")
        logger.info(f"[{name}][{i}] sentences: {target.get('sentences', None)}")
        logger.info(f"[{name}][{i}] boxes: {target.get('boxes', None)}")
        logger.info(f"[{name}][{i}] orig_size: {target.get('orig_size', None)}")
        logger.info(f"[{name}][{i}] img_full_path: {target.get('img_full_path', None)}")
        logger.info("-")

def create_refzom_dataset(args):
    dataset = ReferzomDataset(
        refer_data_root=args.data_root,
        dataset='ref-zom',
        splitBy='final',
        bert_tokenizer=args.tokenizer_type,
        max_tokens=getattr(args, 'max_tokens', 30),
        split='test',
        eval_mode=True,
        size=getattr(args, 'img_size', 320),
        precision=args.precision
    )
    return dataset

def create_gref_dataset(args):
    dataset = GRefDataset(
        refer_data_root=args.data_root,
        dataset='grefcoco',
        splitBy='unc',
        bert_tokenizer=args.tokenizer_type,
        max_tokens=getattr(args, 'max_tokens', 30),
        split='val',
        eval_mode=False,
        size=getattr(args, 'img_size', 320),
        precision=args.precision
    )
    return dataset

def evaluate_refzom():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")

    # 初始化模型
    logger.info("初始化模型...")
    
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
    
    total_params, trainable_params = count_parameters(eval_model)
    logger.info(f"模型参数总数: {total_params:,}, 可训练: {trainable_params:,}")
    
    if model_engine is not None:
        logger.info("成功加载 DeepSpeed checkpoint")
    elif pretrained:
        logger.info("成功加载 PyTorch checkpoint")
    else:
        logger.warning("未指定resume断点，将使用初始化模型进行验证。")

    # 创建验证数据集
    logger.info("创建Refzom验证集...")
    # dataset = create_refzom_dataset(args)
    dataset = create_gref_dataset(args)
    log_sample_info(dataset, 'refzom')
    val_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    # 验证
    logger.info("开始验证...")
    use_negative_masks = getattr(args, 'use_negative_masks', False)
    metrics = validate(eval_model, val_loader, device, use_fp16, use_bf16, use_negative_masks=use_negative_masks)
    logger.info("\n验证结果:")
    logger.info(f"mIoU: {metrics['mIoU']:.4f}")
    logger.info(f"oIoU: {metrics['oIoU']:.4f}")
    logger.info(f"gIoU: {metrics['gIoU']:.4f}")
    logger.info(f"Acc: {metrics['Acc']:.4f}")
    logger.info(f"pointM: {metrics['pointM']:.4f}")
    logger.info(f"best_IoU: {metrics['best_IoU']:.4f}")
    # 保存结果
    results_file = os.path.join(args.output_dir, 'refzom_validation_results.txt')
    with open(results_file, 'w') as f:
        f.write("Refzom验证集结果:\n")
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")
    logger.info(f"结果已保存到: {results_file}")

if __name__ == '__main__':
    evaluate_refzom() 