#!/bin/bash

# 优化后的训练脚本
# 使用改进的模型架构、训练策略和数据增强

echo "Starting optimized ReferSAM training..."

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0,1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# 激活conda环境
source ~/anaconda3/etc/profile.d/conda.sh
conda activate ovseg_lower_pytorch

# 设置随机种子
export PYTHONHASHSEED=123456
NUM_GPUS=2
export FILELOCK_DEFAULT_CLASS=SoftFileLock

 #启动训练
export TRANSFORMER_AUTOTUNE_CACHE=/tmp/deepspeed_autotune_cache

nohup deepspeed --num_gpus $NUM_GPUS train_optimized_fixed.py \
    --deepspeed_config configs/ds_config.json \
    --config configs/student.yaml \
    > train_optimized_$(date +%m%d_%H%M).log 2>&1 &

echo "Training started in background. Check the log file for progress."
echo "To monitor training: tail -f train_optimized_*.log"
echo "To stop training: pkill -f train_optimized_fixed.py"
