#!/bin/bash

# DeepSpeed 启动脚本 - 所有操作在 GPU 上运行
# 使用方法: bash run_deepspeed.sh

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0  # 根据你的GPU数量调整
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1

# GPU 内存优化设置
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export CUDA_LAUNCH_BLOCKING=0  # 设置为0以提高性能

# DeepSpeed 参数
NUM_GPUS=1  # 根据你的GPU数量调整
BATCH_SIZE=16  # 每个GPU的batch size
GRADIENT_ACCUMULATION=4  # 梯度累积步数

# 启动 DeepSpeed 训练
deepspeed --num_gpus $NUM_GPUS \
    train_bert_multiGpu.py \
    --deepspeed_config ds_config.json \
    --batch_size $BATCH_SIZE \
    --epochs 15 \
    --lr 1e-4 \
    --output_dir ./outputs/deepspeed_training \
    --tokenizer_type bert-base-uncased \