#!/bin/sh

python train.py --loss=SSIM --val-metric=SSIM --opt=SGD --log_name=SSIM_SSIM_SGD_res2 --batch-size=4 --epochs=10 --lr=0.02 --ese=3
# python train.py --checkpoint=models/SSIM_SSIM_SGD.pth --loss=SSIM --val-metric=SSIM --opt=SGD --log_name=SSIM_SSIM_SGD --batch-size=4 --epochs=10 --lr=0.001 --ese=3

# # vary training loss for val loss PSNR and opt SGD
# python train.py --loss=L1 --val-metric=PSNR --opt=SGD --log_name=L1_PSNR_SGD --batch-size=4 --epochs=10 --lr=0.02 --ese=3
# python train.py --loss=L2 --val-metric=PSNR --opt=SGD --log_name=L2_PSNR_SGD --batch-size=4 --epochs=10 --lr=0.02 --ese=3
# python train.py --loss=Huber --val-metric=PSNR --opt=SGD --log_name=Huber_PSNR_SGD --batch-size=4 --epochs=10 --lr=0.02 --ese=3
# python train.py --loss=SSIM --val-metric=PSNR --opt=SGD --log_name=SSIM_PSNR_SGD --batch-size=4 --epochs=10 --lr=0.02 --ese=3

# # vary training loss for val loss SSIM and opt SGD
# python train.py --loss=L1 --val-metric=SSIM --opt=SGD --log_name=L1_SSIM_SGD --batch-size=4 --epochs=10 --lr=0.02 --ese=3
# python train.py --loss=L2 --val-metric=SSIM --opt=SGD --log_name=L2_SSIM_SGD --batch-size=4 --epochs=10 --lr=0.02 --ese=3
# python train.py --loss=Huber --val-metric=SSIM --opt=SGD --log_name=Huber_SSIM_SGD --batch-size=4 --epochs=10 --lr=0.02 --ese=3
# python train.py --loss=SSIM --val-metric=SSIM --opt=SGD --log_name=SSIM_SSIM_SGD --batch-size=4 --epochs=10 --lr=0.02 --ese=3

# # vary training loss for val loss PSNR and opt Adam
# python train.py --loss=L1 --val-metric=PSNR --opt=Adam --log_name=L1_PSNR_Adam --batch-size=4 --epochs=10 --lr=0.02 --ese=3
# python train.py --loss=L2 --val-metric=PSNR --opt=Adam --log_name=L2_PSNR_Adam --batch-size=4 --epochs=10 --lr=0.02 --ese=3
# python train.py --loss=Huber --val-metric=PSNR --opt=Adam --log_name=Huber_PSNR_Adam --batch-size=4 --epochs=10 --lr=0.02 --ese=3
# python train.py --loss=SSIM --val-metric=PSNR --opt=Adam --log_name=SSIM_PSNR_Adam --batch-size=4 --epochs=10 --lr=0.02 --ese=3

# # vary training loss for val loss SSIM and opt Adam
# python train.py --loss=L1 --val-metric=SSIM --opt=Adam --log_name=L1_SSIM_Adam --batch-size=4 --epochs=10 --lr=0.02 --ese=3
# python train.py --loss=L2 --val-metric=SSIM --opt=Adam --log_name=L2_SSIM_Adam --batch-size=4 --epochs=10 --lr=0.02 --ese=3
# python train.py --loss=Huber --val-metric=SSIM --opt=Adam --log_name=Huber_SSIM_Adam --batch-size=4 --epochs=10 --lr=0.02 --ese=3
# python train.py --loss=SSIM --val-metric=SSIM --opt=Adam --log_name=SSIM_SSIM_Adam --batch-size=4 --epochs=10 --lr=0.02 --ese=3