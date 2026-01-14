#! /bin/bash

#JSUB -q ai_share
#JSUB -m gpu01
#JSUB -n 24
#JSUB -gpgpu 2
#JSUB -J 没有边界损失函数和iou
bash scripts/train_test_loss.sh
#bash scripts/val.sh
# bash scripts/visual.sh
