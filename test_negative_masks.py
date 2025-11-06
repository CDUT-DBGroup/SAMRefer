"""
快速测试负样本掩码功能的脚本
使用方法: python test_negative_masks.py [--use_negative_masks]
"""
import torch
import argparse
from torch.utils.data import DataLoader
from get_args import get_args
from model.builder import refersam
from validation.evaluation import validate
from dataset.ReferDataset import ReferDataset
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_negative_masks(use_negative_masks=True, num_samples=50):
    """
    快速测试负样本掩码功能
    
    Args:
        use_negative_masks: 是否使用负样本掩码
        num_samples: 测试样本数量
    """
    logger.info("=" * 60)
    logger.info(f"测试负样本掩码功能 (use_negative_masks={use_negative_masks})")
    logger.info("=" * 60)
    
    # 获取参数
    args = get_args()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")
    
    # 加载模型
    logger.info("加载模型...")
    model = refersam(args=args, pretrained=True)
    model = model.to(device)
    model.eval()
    
    # 创建测试数据集（使用refcoco验证集）
    logger.info("创建测试数据集...")
    test_dataset = ReferDataset(
        refer_data_root=args.data_root,
        dataset='refcoco',
        splitBy='unc',
        bert_tokenizer=args.tokenizer_type,
        max_tokens=30,
        split='val',
        eval_mode=False,
        size=320,
        precision=args.precision,
    )
    
    # 限制测试样本数量
    if num_samples > 0 and num_samples < len(test_dataset):
        # 创建一个子集
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
        model, 
        test_loader, 
        device, 
        use_fp16=False, 
        use_bf16=False,
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


def compare_with_without_negative_masks(num_samples=50):
    """
    对比使用和不使用负样本掩码的性能差异
    """
    logger.info("\n" + "=" * 60)
    logger.info("对比测试：使用 vs 不使用负样本掩码")
    logger.info("=" * 60)
    
    # 不使用负样本掩码
    logger.info("\n1. 测试不使用负样本掩码...")
    metrics_without = test_negative_masks(use_negative_masks=False, num_samples=num_samples)
    
    # 使用负样本掩码
    logger.info("\n2. 测试使用负样本掩码...")
    metrics_with = test_negative_masks(use_negative_masks=True, num_samples=num_samples)
    
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
    parser = argparse.ArgumentParser(description='测试负样本掩码功能')
    parser.add_argument('--use_negative_masks', action='store_true', 
                       help='启用负样本掩码功能')
    parser.add_argument('--compare', action='store_true',
                       help='对比使用和不使用负样本掩码的性能')
    parser.add_argument('--num_samples', type=int, default=50,
                       help='测试样本数量（0表示使用全部）')
    
    args = parser.parse_args()
    
    if args.compare:
        # 对比测试
        compare_with_without_negative_masks(num_samples=args.num_samples)
    else:
        # 单次测试
        test_negative_masks(
            use_negative_masks=args.use_negative_masks,
            num_samples=args.num_samples
        )

