#! /bin/bash

#JSUB -q ai_share
#JSUB -gpgpu 2
#JSUB -J refersam_train
#JSUB -o output.%J
#JSUB -e error.%J
# bash scripts/train_test_loss.sh
bash scripts/val.sh
