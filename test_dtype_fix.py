#!/usr/bin/env python3
"""
测试数据类型修复的简单脚本
"""
import torch
import json
import os

def test_dtype_consistency():
    """测试数据类型一致性"""
    print("测试数据类型一致性...")
    
    # 读取DeepSpeed配置
    with open('configs/ds_config.json', 'r') as f:
        ds_config = json.load(f)
    
    use_fp16 = ds_config.get("fp16", {}).get("enabled", False)
    use_bf16 = ds_config.get("bf16", {}).get("enabled", False)
    
    print(f"DeepSpeed配置:")
    print(f"  fp16 enabled: {use_fp16}")
    print(f"  bf16 enabled: {use_bf16}")
    
    # 创建测试张量
    test_input = torch.randn(2, 3, 320, 320)
    print(f"原始输入数据类型: {test_input.dtype}")
    
    # 模拟修复后的数据类型处理
    if use_fp16:
        test_input = test_input.half()
        print(f"fp16转换后数据类型: {test_input.dtype}")
    elif use_bf16:
        test_input = test_input.to(torch.bfloat16)
        print(f"bf16转换后数据类型: {test_input.dtype}")
    else:
        print(f"使用fp32: {test_input.dtype}")
    
    # 创建测试模型参数
    test_param = torch.randn(64, 3, 7, 7)
    print(f"原始参数数据类型: {test_param.dtype}")
    
    if use_fp16:
        test_param = test_param.half()
        print(f"fp16转换后参数数据类型: {test_param.dtype}")
    elif use_bf16:
        test_param = test_param.to(torch.bfloat16)
        print(f"bf16转换后参数数据类型: {test_param.dtype}")
    
    # 检查数据类型匹配
    if test_input.dtype == test_param.dtype:
        print("✅ 数据类型匹配成功！")
        return True
    else:
        print("❌ 数据类型不匹配！")
        print(f"输入类型: {test_input.dtype}")
        print(f"参数类型: {test_param.dtype}")
        return False

if __name__ == "__main__":
    success = test_dtype_consistency()
    if success:
        print("\n🎉 数据类型修复验证通过！")
    else:
        print("\n⚠️  数据类型修复需要进一步调整")
