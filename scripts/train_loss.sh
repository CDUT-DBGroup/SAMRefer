#!/bin/bash

# 优化后的训练脚本
# 使用改进的模型架构、训练策略和数据增强

echo "Starting optimized ReferSAM training..."

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

 #启动训练
export TRANSFORMER_AUTOTUNE_CACHE=/tmp/deepspeed_autotune_cache

nohup deepspeed --num_gpus $NUM_GPUS train_enhanced_loss.py \
# nohup deepspeed --num_gpus $NUM_GPUS train_enhanced_multi_dataset.py \
    --deepspeed_config configs/ds_config.json \
    --config configs/student.yaml \
    --use_enhanced_loss \
    --loss_config_path configs/enhanced_loss_config.yaml \
     > train_loss_$(date +%m%d_%H%M).log 2>&1 &

echo "Training started in background. Check the log file for progress."
echo "To monitor training: tail -f train_loss_*.log"
echo "To stop training: pkill -f train_enhanced_loss.py"


# #!/bin/bash

# # 正确初始化 conda（确保路径正确，视你使用 miniconda 或 anaconda 而定）
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate SAM

# # 出错时立即退出并自动关机
# trap '' SIGHUP

# # 启动训练
# torchrun --nproc_per_node=2 train_bert_multiGpu.py #--resume /root/autodl-tmp/vision_paper/ReferSAM/output/refersam_bert/checkpoint_epoch_2.pt
# # torchrun --nproc_per_node=1 train_bert_multiGpu.py 
# # 成功则关机
# /usr/bin/shutdown -h now

# scp -P 10961 train_loss_1003_1904.log root@connect.cqa1.seetacloud.com:/root/autodl-tmp/vision_paper/ReferSAM/
