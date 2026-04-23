#!/bin/bash
echo "Starting optimized ReferSAM training..."

# 不再依赖 CUDA_VISIBLE_DEVICES（避免被 DeepSpeed 覆盖）
unset CUDA_VISIBLE_DEVICES

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export DS_ACCELERATOR=cuda
export PYTHONHASHSEED=123456
export FILELOCK_DEFAULT_CLASS=SoftFileLock
export TRANSFORMER_AUTOTUNE_CACHE=/tmp/deepspeed_autotune_cache

# 端口（避免 29500 冲突）
MASTER_PORT=29501

# 激活环境
source ~/conda.env
conda activate ovseg_lower_pytorch

# 检查 CUDA
python - <<EOF
import torch
print("CUDA available:", torch.cuda.is_available())
EOF

# ======== 启动 DeepSpeed（关键部分） ========
deepspeed \
  --master_port ${MASTER_PORT} \
  --include localhost:0,1 \
  train_enhanced_multi_dataset.py \
  --deepspeed_config configs/ds_config.json \
  --config configs/student.yaml \
  --use_enhanced_loss \
  --loss_config_path configs/enhanced_loss_config.yaml \
  --output_dir output/text/no_fusion_weight \
  > 文本注意力测试-没有融合权重.log 2>&1

echo "Training started."
echo "Monitor: tail -f train_loss_*.log"
echo "Stop: pkill -f train_enhanced_multi_dataset.py"
