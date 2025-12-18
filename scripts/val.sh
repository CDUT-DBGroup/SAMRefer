echo "Starting multi-dataset ReferSAM training..."

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# 激活conda环境
source ~/anaconda3/etc/profile.d/conda.sh
# source ~/miniconda3/etc/profile.d/conda.sh
conda activate ovseg_lower_pytorch
# conda activate SAM

# 设置随机种子
export PYTHONHASHSEED=123456
NUM_GPUS=1
deepspeed --num_gpus $NUM_GPUS validate_bert.py --deepspeed_config /root/autodl-tmp/vision_paper/ReferSAM/configs/ds_config.json --config configs/main_refersam_bert.yaml --use_enhanced_loss --loss_config_path configs/enhanced_loss_config.yaml > 我的论文模型的验证集-1208-不缺少position_ids.log 2>&1 &