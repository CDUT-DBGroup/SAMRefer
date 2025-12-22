#! /bin/bash

#JSUB -q ai_share
#JSUB -m gpu03
#JSUB -n 8
#JSUB -gpgpu 2
#JSUB -J refersam_train
bash scripts/train_test_loss.sh
# bash scripts/val.sh
