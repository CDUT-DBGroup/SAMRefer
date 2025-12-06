#! /bin/bash

#JUB -q nonmig
#JSUB -gpgpu 2               # 申请 2 张 GPU 卡
#JSUB -J refersam_train
#JSUB -o output.%J
#JSUB -e error.%J
bash scripts/train_test_loss.sh
