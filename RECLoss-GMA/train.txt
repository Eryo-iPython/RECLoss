#!/usr/bin/env bash
python train.py --stage chairs --validation chairs --num_steps 120000 --lr 0.00025 --image_size 368 496 --wdecay 0.0001 --gpus 0 1 --batch_size 8 --mixed_precision
python train.py --stage things --validation sintel --num_steps 120000 --lr 0.000125 --image_size 400 720 --wdecay 0.0001 --gpus 0 1 --batch_size 6 --mixed_precision
python train.py --stage sintel --validation sintel --num_steps 120000 --lr 0.000125 --image_size 368 768 --wdecay 0.00001 --gamma 0.85 --gpus 0 1 --batch_size 6 --mixed_precision
python train.py --stage kitti --validation kitti --num_steps 50000 --lr 0.000125 --image_size 288 960 --wdecay 0.00001 --gamma 0.85 --gpus 0 1 --batch_size 6 --mixed_precision
