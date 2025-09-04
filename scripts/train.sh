screen -S refersam
conda activate SAM
bash scripts/train_1.sh > train_0711_referit.log 2>&1
Ctrl + A，然后松手，按 D

nohup ./scripts/train_1.sh > train_0628.log 2>&1 &

nohup ./scripts/run_deepspeed.sh > train_0717.log 2>&1 &

