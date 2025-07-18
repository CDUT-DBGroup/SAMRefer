#!/bin/bash

# DeepSpeed 启动脚本 - 所有操作在 GPU 上运行
# 使用方法: bash run_deepspeed.sh

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0  # 根据你的GPU数量调整
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=2

# GPU 内存优化设置
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export CUDA_LAUNCH_BLOCKING=0  # 设置为0以提高性能

# DeepSpeed 参数
NUM_GPUS=1  # 根据你的GPU数量调整

# 启动 DeepSpeed 训练
deepspeed --num_gpus $NUM_GPUS \
    train_bert_multiGpu.py \
    --deepspeed_config configs/ds_config.json

/usr/bin/shutdown -h now