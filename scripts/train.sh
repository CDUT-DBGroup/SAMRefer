screen -S refersam
bash scripts/train_1.sh > train_0703.log 2>&1
Ctrl + A，然后松手，按 D

nohup ./scripts/train_1.sh > train_0628.log 2>&1 &