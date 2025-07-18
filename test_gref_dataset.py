#!/usr/bin/env python3
import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataset.GRefDataset import GRefDataset
from get_args import get_args

def test_dataset():
    print("Testing GRefDataset...")
    
    try:
        args = get_args()
        
        # 创建数据集
        dataset = GRefDataset(
            refer_data_root=args.data_root,
            dataset='grefcoco',
            splitBy='unc',
            bert_tokenizer=args.tokenizer_type,
            max_tokens=30,
            split='train',
        )
        
        print(f"Dataset length: {len(dataset)}")
        
        # 测试获取第一个样本
        print("Testing first sample...")
        samples, targets = dataset[0]
        
        print("Samples keys:", samples.keys())
        print("Targets keys:", targets.keys())
        
        # 检查数据类型
        print(f"word_ids type: {type(samples['word_ids'])}")
        print(f"word_ids dtype: {samples['word_ids'].dtype}")
        print(f"word_masks type: {type(samples['word_masks'])}")
        print(f"word_masks dtype: {samples['word_masks'].dtype}")
        print(f"orig_size type: {type(samples['orig_size'])}")
        print(f"orig_size dtype: {samples['orig_size'].dtype}")
        print(f"img type: {type(samples['img'])}")
        print(f"img shape: {samples['img'].shape}")
        print(f"mask type: {type(targets['mask'])}")
        print(f"mask shape: {targets['mask'].shape}")
        
        # 测试DataLoader
        print("\nTesting DataLoader...")
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=2,
            shuffle=False,
            num_workers=0,  # 使用0避免多进程问题
            pin_memory=False
        )
        
        for i, (batch_samples, batch_targets) in enumerate(dataloader):
            print(f"Batch {i}:")
            print(f"  word_ids shape: {batch_samples['word_ids'].shape}")
            print(f"  word_masks shape: {batch_samples['word_masks'].shape}")
            print(f"  img shape: {batch_samples['img'].shape}")
            print(f"  mask shape: {batch_targets['mask'].shape}")
            break
            
        print("Dataset test completed successfully!")
        
    except Exception as e:
        print(f"Error testing dataset: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_dataset() 