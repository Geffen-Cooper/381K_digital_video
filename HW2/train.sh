#!/bin/sh

python train.py --loss=L1 --val-loss=PSNR --opt=SGD --log_name=L1_PSNR_SGD --batch-size=4 --epochs=10 --lr=0.01 --ese=3