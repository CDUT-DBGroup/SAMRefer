echo "Starting multi-dataset ReferSAM training with model_origin..."

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# 可选：手动指定 MASTER_PORT（如果不设置，代码会自动查找可用端口）
# export MASTER_PORT=29501  # 取消注释并修改为你想要的端口号

# 激活conda环境
# source ~/anaconda3/etc/profile.d/conda.sh   
source ~/miniconda3/etc/profile.d/conda.sh
# conda activate ovseg_lower_pytorch
conda activate SAM

# 设置随机种子
export PYTHONHASHSEED=123456
NUM_GPUS=1
python validate_origin.py --config configs/main_origin.yaml --use_best_sentence --sentence_aggregation mean_iou > 我的论文模型的验证集-model_origin-1222-使用meanIoU.log 2>&1 &

