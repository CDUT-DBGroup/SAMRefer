"""
快速测试负样本掩码功能的脚本
使用方法: python test_negative_masks.py [--use_negative_masks] [--compare] [--num_samples N] [--pre_train_path PATH]
"""
import torch
import argparse
import os
from torch.utils.data import DataLoader
from get_args import get_args
from model.enhanced_builder import refersam, load_model_with_checkpoint
from validation.evaluation import validate
from dataset.ReferDataset import ReferDataset
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_model(args, device):
    """
    加载模型，支持 DeepSpeed 和普通 PyTorch checkpoint
    """
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
    
    if model_engine is not None:
        logger.info("成功加载 DeepSpeed checkpoint")
    elif pretrained:
        logger.info("成功加载 PyTorch checkpoint")
    else:
        logger.info("使用初始化的模型（未加载 checkpoint）")
    
    return eval_model, use_fp16, use_bf16


def test_negative_masks(use_negative_masks=True, num_samples=50, checkpoint_path=None, args=None):
    """
    快速测试负样本掩码功能
    
    Args:
        use_negative_masks: 是否使用负样本掩码
        num_samples: 测试样本数量（0表示使用全部）
        checkpoint_path: 模型 checkpoint 路径
        args: 参数对象，如果为 None 则调用 get_args() 获取
    """
    logger.info("=" * 60)
    logger.info(f"测试负样本掩码功能 (use_negative_masks={use_negative_masks})")
    logger.info("=" * 60)
    
    # 获取参数
    if args is None:
        args = get_args()
    
    # 如果提供了 checkpoint 路径，覆盖 args 中的路径
    if checkpoint_path:
        args.pre_train_path = checkpoint_path
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")
    
    # 加载模型
    eval_model, use_fp16, use_bf16 = load_model(args, device)
    
    # 创建测试数据集（使用refcoco验证集，eval_mode=True）
    logger.info("创建测试数据集...")
    test_dataset = ReferDataset(
        refer_data_root=args.data_root,
        dataset='refcoco',
        splitBy='unc',
        bert_tokenizer=args.tokenizer_type,
        max_tokens=getattr(args, 'max_tokens', 30),
        split='val',
        eval_mode=True,  # 验证模式
        size=getattr(args, 'img_size', 320),
        precision=args.precision,
    )
    
    # 限制测试样本数量
    if num_samples > 0 and num_samples < len(test_dataset):
        indices = torch.randperm(len(test_dataset))[:num_samples].tolist()
        test_dataset = torch.utils.data.Subset(test_dataset, indices)
        logger.info(f"使用 {num_samples} 个样本进行测试")
    else:
        logger.info(f"使用全部 {len(test_dataset)} 个样本进行测试")
    
    # 创建数据加载器
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    # 运行验证
    logger.info("开始验证...")
    metrics = validate(
        eval_model, 
        test_loader, 
        device, 
        use_fp16=use_fp16, 
        use_bf16=use_bf16,
        use_negative_masks=use_negative_masks
    )
    
    # 打印结果
    logger.info("\n" + "=" * 60)
    logger.info("验证结果:")
    logger.info("=" * 60)
    logger.info(f"mIoU:      {metrics['mIoU']:.4f}")
    logger.info(f"oIoU:      {metrics['oIoU']:.4f}")
    logger.info(f"gIoU:      {metrics['gIoU']:.4f}")
    logger.info(f"Acc:       {metrics['Acc']:.4f}")
    logger.info(f"pointM:    {metrics['pointM']:.4f}")
    logger.info(f"best_IoU:  {metrics['best_IoU']:.4f}")
    logger.info("=" * 60)
    
    return metrics


def compare_with_without_negative_masks(num_samples=50, checkpoint_path=None, args=None):
    """
    对比使用和不使用负样本掩码的性能差异
    """
    logger.info("\n" + "=" * 60)
    logger.info("对比测试：使用 vs 不使用负样本掩码")
    logger.info("=" * 60)
    
    # 不使用负样本掩码
    logger.info("\n1. 测试不使用负样本掩码...")
    metrics_without = test_negative_masks(
        use_negative_masks=False, 
        num_samples=num_samples,
        checkpoint_path=checkpoint_path,
        args=args
    )
    
    # 使用负样本掩码
    logger.info("\n2. 测试使用负样本掩码...")
    metrics_with = test_negative_masks(
        use_negative_masks=True, 
        num_samples=num_samples,
        checkpoint_path=checkpoint_path,
        args=args
    )
    
    # 对比结果
    logger.info("\n" + "=" * 60)
    logger.info("性能对比:")
    logger.info("=" * 60)
    logger.info(f"{'指标':<12} {'不使用负掩码':<15} {'使用负掩码':<15} {'提升':<10}")
    logger.info("-" * 60)
    
    metrics_names = ['mIoU', 'oIoU', 'gIoU', 'pointM']
    for name in metrics_names:
        without = metrics_without[name]
        with_val = metrics_with[name]
        improvement = (with_val - without) * 100 / without if without > 0 else 0
        logger.info(f"{name:<12} {without:<15.4f} {with_val:<15.4f} {improvement:>+6.2f}%")
    
    logger.info("=" * 60)
    
    return metrics_without, metrics_with


if __name__ == '__main__':
    # 先解析脚本特有的参数，使用 parse_known_args 避免与 get_args() 冲突
    parser = argparse.ArgumentParser(description='测试负样本掩码功能', add_help=False)
    parser.add_argument('--use_negative_masks', action='store_true', 
                       help='启用负样本掩码功能')
    parser.add_argument('--compare', action='store_true',
                       help='对比使用和不使用负样本掩码的性能')
    parser.add_argument('--num_samples', type=int, default=50,
                       help='测试样本数量（0表示使用全部）')
    parser.add_argument('--pre_train_path', type=str, default=None,
                       help='模型 checkpoint 路径（可选，会覆盖配置文件中的路径）')
    
    # 解析已知参数，剩余参数会传递给 get_args()
    script_args, remaining_argv = parser.parse_known_args()
    
    # 将剩余参数设置回 sys.argv，让 get_args() 可以解析它们
    import sys
    original_argv = sys.argv
    sys.argv = [sys.argv[0]] + remaining_argv
    
    # 获取模型配置参数
    try:
        model_args = get_args()
    finally:
        # 恢复原始 argv
        sys.argv = original_argv
    
    # 如果提供了 checkpoint 路径，覆盖 model_args 中的路径
    if script_args.pre_train_path:
        model_args.pre_train_path = script_args.pre_train_path
    
    if script_args.compare:
        # 对比测试
        compare_with_without_negative_masks(
            num_samples=script_args.num_samples,
            checkpoint_path=script_args.pre_train_path,
            args=model_args
        )
    else:
        # 单次测试
        test_negative_masks(
            use_negative_masks=script_args.use_negative_masks,
            num_samples=script_args.num_samples,
            checkpoint_path=script_args.pre_train_path,
            args=model_args
        )
