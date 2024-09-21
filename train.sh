#!/usr/bin/env bash

CONFIG=$1
# export RANK=0
# export WORLD_SIZE=2     # the number of the thread to be called
# export MASTER_ADDR=localhost
# export MASTER_PORT=5678
# export LOCAL_RANK=0
# export CUDA_VISIBLE_DEVICES=0

# echo $CONFIG


python -m torch.distributed.launch --nproc_per_node=1 --master_port=4321 basicsr/train.py --local_rank 0 -opt $CONFIG --launcher pytorch
