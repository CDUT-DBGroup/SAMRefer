import torch
import numpy as np
from validation.evaluation import validate
from dataset.ReferDataset import ReferDataset
from model.builder import refersam
from get_args import get_args
from torch.utils.data import DataLoader

def test_validation():
    """测试验证函数是否正常工作"""
    print("Testing validation function...")
    
    # 设置参数
    args = get_args()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # 创建模型
    print("Creating model...")
    model = refersam(args=args, pretrained=True)
    model = model.to(device)
    model.eval()
    
    # 创建小数据集进行测试
    print("Creating test dataset...")
    test_dataset = ReferDataset(
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
    
    # 只取前几个样本进行测试
    test_loader = DataLoader(
        test_dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )
    
    print("Running validation test...")
    metrics = validate(model, test_loader, device)
    
    print("Validation results:")
    print(f"mIoU: {metrics['mIoU']:.4f}")
    print(f"IoU: {metrics['IoU']:.4f}")
    print(f"pointM: {metrics['pointM']:.4f}")
    print(f"best_IoU: {metrics['best_IoU']:.4f}")
    
    # 检查结果是否合理
    assert 0 <= metrics['mIoU'] <= 1, f"mIoU should be between 0 and 1, got {metrics['mIoU']}"
    assert 0 <= metrics['IoU'] <= 1, f"IoU should be between 0 and 1, got {metrics['IoU']}"
    assert 0 <= metrics['pointM'] <= 1, f"pointM should be between 0 and 1, got {metrics['pointM']}"
    assert 0 <= metrics['best_IoU'] <= 1, f"best_IoU should be between 0 and 1, got {metrics['best_IoU']}"
    
    print("✅ Validation test passed!")

if __name__ == '__main__':
    test_validation() 