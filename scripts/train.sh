screen -S refersam
conda activate SAM
bash scripts/train_1.sh > train_0707_main_32_multi.log 2>&1
Ctrl + A，然后松手，按 D

nohup ./scripts/train_1.sh > train_0628.log 2>&1 &