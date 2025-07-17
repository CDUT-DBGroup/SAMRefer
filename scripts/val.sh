 NUM_GPUS=1
 deepspeed --num_gpus $NUM_GPUS \
    validate_refzom.py \
    --deepspeed_config configs/ds_config.json