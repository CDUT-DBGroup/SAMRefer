CUDA_VISIBLE_DEVICES=0,1  nohup torchrun --nproc_per_node=2 train_bert_multiGpu.py --resume # 分布式多GPU训练

nohup torchrun --nproc_per_node=2 train_bert_multiGpu.py>train_0624.log 2>&1 &

nohup bash -c 'torchrun --nproc_per_node=2 train_bert_multiGpu.py --resume /root/autodl-tmp/vision_paper/ReferSAM/output/refersam_bert/best_iou_miou_model.pt > train_0624.log 2>&1; /usr/bin/shutdown -h now' &
