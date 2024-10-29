#!/bin/bash
cd ./rtdetrv2_pytorch

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.run \
                                    --nproc_per_node=4 tools/train.py \
                                    -c configs/rtdetrv2/rtdetrv2_stdc2vd_6x_coco.yml \
                                    --use-amp --seed=0 --writer-type wandb \
                                    --project-path test \
                                    -u train_dataloader.total_batch_size=64 train_dataloader.num_workers=8 val_dataloader.total_batch_size=128 val_dataloader.num_workers=8 &> log.txt 2>&1 &

# python3 tools/train.py \
# -c configs/rtdetrv2/rtdetrv2_stdc2vd_6x_coco.yml \
# --use-amp --seed=0 --writer-type wandb \
# -u train_dataloader.total_batch_size=6 val_dataloader.total_batch_size=12 # &> log.txt 2>&1 &
