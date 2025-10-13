#!/bin/bash

# 多数据集训练脚本
# 使用改进的模型架构、训练策略和数据增强，支持多个数据集联合训练

echo "Starting multi-dataset ReferSAM training..."

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0,1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# 激活conda环境
source ~/anaconda3/etc/profile.d/conda.sh
# source ~/miniconda3/etc/profile.d/conda.sh
conda activate ovseg_lower_pytorch
# conda activate SAM

# 设置随机种子
export PYTHONHASHSEED=123456
NUM_GPUS=2
export FILELOCK_DEFAULT_CLASS=SoftFileLock

# 启动多数据集训练
export TRANSFORMER_AUTOTUNE_CACHE=/tmp/deepspeed_autotune_cache

nohup deepspeed --num_gpus $NUM_GPUS train_enhanced_multi_dataset.py \
# nohup deepspeed --num_gpus $NUM_GPUS train_enhanced_loss.py \
    --deepspeed_config configs/ds_config.json \
    --config configs/main_refersam_bert.yaml \
    --use_enhanced_loss \
    --loss_config_path configs/enhanced_loss_config.yaml \
     > train_multi_dataset_$(date +%m%d_%H%M).log 2>&1 &

echo "Multi-dataset training started in background. Check the log file for progress."
echo "To monitor training: tail -f train_multi_dataset_*.log"
echo "To stop training: pkill -f train_enhanced_multi_dataset.py"

# 可选：训练完成后自动关机（取消注释以启用）
# wait
# echo "Training completed successfully. Shutting down..."
# /usr/bin/shutdown -h now
