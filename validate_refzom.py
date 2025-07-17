import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from dataset.RefzomDataset import ReferzomDataset
from dataset.GRefDataset import GRefDataset
from model.builder import refersam
from validation.evaluation import validate
from get_args import get_args
import random
import logging
import json
import glob
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

def load_deepspeed_checkpoint(checkpoint_path, model_engine):
    # 支持目录resume，自动找最新global_step子目录
    if os.path.isdir(checkpoint_path):
        subdirs = glob.glob(os.path.join(checkpoint_path, "global_step*"))
        if subdirs:
            latest_subdir = max(subdirs, key=os.path.getmtime)
            logger.info(f"从断点目录加载: {latest_subdir}")
            checkpoint_path = latest_subdir
        else:
            logger.info(f"从断点目录加载: {checkpoint_path}")
    else:
        logger.info(f"从断点文件加载: {checkpoint_path}")
    _, client_state = model_engine.load_checkpoint(checkpoint_path)
    logger.info(f"成功加载断点: {checkpoint_path}")
    logger.info(f"Client state: {client_state}")
    return client_state

def evaluate_refzom():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")

    # 初始化模型
    logger.info("初始化模型...")
    model = refersam(args=args)  # 不加载预训练权重
    model = model.to(device)
    total_params, trainable_params = count_parameters(model)
    logger.info(f"模型参数总数: {total_params:,}, 可训练: {trainable_params:,}")

    # 读取DeepSpeed配置
    ds_config = 'configs/ds_config.json'
    with open(ds_config) as f:
        ds_conf = json.load(f)
    use_fp16 = ds_conf.get("fp16", {}).get("enabled", False)
    use_bf16 = ds_conf.get("bf16", {}).get("enabled", False)

    # 初始化DeepSpeed引擎
    logger.info("初始化DeepSpeed引擎...")
    if hasattr(model, 'params_to_optimize'):
        param_groups = model.params_to_optimize()
    else:
        param_groups = model.parameters()
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=param_groups,
        config=ds_config
    )

    # 从断点恢复
    if hasattr(args, 'pre_train_path') and args.pre_train_path:
        logger.info(f"从预训练权重加载: {args.pre_train_path}")
        load_deepspeed_checkpoint(args.pre_train_path, model_engine)
    else:
        logger.warning("未指定resume断点，将使用初始化模型进行验证。")
    model_engine.eval()

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
    metrics = validate(model_engine.module, val_loader, device, use_fp16, use_bf16)
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