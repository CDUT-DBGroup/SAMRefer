#!/bin/bash

# 多数据集训练脚本
# 使用改进的模型架构、训练策略和数据增强，支持多个数据集联合训练

echo "Starting multi-dataset ReferSAM training..."

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# 激活conda环境
# source ~/anaconda3/etc/profile.d/conda.sh
source ~/miniconda3/etc/profile.d/conda.sh
# conda activate ovseg_lower_pytorch
conda activate SAM

# 设置随机种子
export PYTHONHASHSEED=123456
NUM_GPUS=1
export FILELOCK_DEFAULT_CLASS=SoftFileLock

# 启动多数据集训练
export TRANSFORMER_AUTOTUNE_CACHE=/tmp/deepspeed_autotune_cache

# 记录日志文件名
LOG_FILE="train_multi_dataset_$(date +%m%d_%H%M).log"


echo "Starting training... Log file: $LOG_FILE"
echo "To monitor training: tail -f $LOG_FILE"
echo "To stop training: pkill -f train_enhanced_multi_dataset.py"

# 运行训练命令（后台运行，会阻塞直到完成）
# 无论训练成功或失败，完成后都会自动关机
nohup deepspeed --num_gpus $NUM_GPUS train_enhanced_multi_dataset.py \
    --deepspeed_config configs/ds_config.json \
    --config configs/main_refersam_bert.yaml \
    --use_enhanced_loss \
    --loss_config_path configs/enhanced_loss_config.yaml \
     > "$LOG_FILE" 2>&1 &

# 保存后台进程的 PID
TRAIN_PID=$!
echo "Training process started with PID: $TRAIN_PID"

# 等待训练进程完成（包括所有子进程）
wait $TRAIN_PID
TRAIN_EXIT_CODE=$?

# 检查训练是否成功完成
if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo "Training completed successfully. Exit code: $TRAIN_EXIT_CODE"
else
    echo "Training completed with errors. Exit code: $TRAIN_EXIT_CODE"
fi

echo "Shutting down..."
# 使用 sudo shutdown 或直接 shutdown，根据系统权限配置
/usr/bin/shutdown -h now
