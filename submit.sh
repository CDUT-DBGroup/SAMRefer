#! /bin/bash

#JSUB -q ai_share
#JSUB -m gpu07
#JSUB -n 32
#JSUB -gpgpu 2
#JSUB -J 文本注意力-没有融合权重
bash scripts/train_test_loss.sh
#bash scripts/val.sh
# bash scripts/visual.sh
