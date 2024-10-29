#!/bin/bash
cd ./rtdetrv2_pytorch

# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port=9909 --nproc_per_node=4 tools/train.py -c configs/rtdetrv2/rtdetrv2_stdc2vd_6x_coco.yml --use-amp --seed=0 &> log.txt 2>&1 &
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.run --nproc_per_node=4 tools/train.py -c configs/rtdetrv2/rtdetrv2_stdc2vd_6x_coco.yml --use-amp --seed=0 -u batch_size=64 --writer-type wandb &> log.txt 2>&1 &
# python3 tools/train.py -c configs/rtdetrv2/rtdetrv2_stdc2vd_6x_coco.yml --use-amp --seed=0 --writer-type wandb &> log.txt 2>&1 &
