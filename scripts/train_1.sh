#!/bin/bash

# 正确初始化 conda（确保路径正确，视你使用 miniconda 或 anaconda 而定）
source ~/miniconda3/etc/profile.d/conda.sh
conda activate SAM

# 出错时立即退出并自动关机
set -e
trap 'echo "Training failed or interrupted. Shutting down..."; /usr/bin/shutdown -h now' ERR

# 启动训练
torchrun --nproc_per_node=2 train_bert_multiGpu.py --resume /root/autodl-tmp/vision_paper/ReferSAM/output/refersam_bert/checkpoint_epoch_14.pt

# 成功则关机
/usr/bin/shutdown -h now
