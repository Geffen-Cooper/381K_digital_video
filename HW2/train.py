'''
    This file takes in command line arguments for the training parameters
    and runs a training/test function. In general, this code tries to be agnostic
    to the model and dataset but assumes a standard supervised training setup.
'''

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import time
from datasets import *
from models import *
import math

# the input will be a video frame (C,W,H)
def PSNR(comp, gt):
    mse = F.mse_loss(comp,gt)
    max_val = 255 # videos are uint8 
    return 10*torch.log10(max_val**2/(mse+1e-6))

# expects the video volumes as inputs (N,C,D,W,H)
class PSNR_loss(nn.Module):
    def __init__(self) -> None:
        super(PSNR_loss,self).__init__()

    def forward(self,output,target):
        # first reshape into (D,C,W,H)
        output = torch.permute(output[0],(1,0,2,3))
        target = torch.permute(target[0],(1,0,2,3))

        # average PSNR over all frames
        loss = 0
        for comp,gt in zip(output,target):
            loss += PSNR(comp,gt)
        return loss / len(output)


# this SSIM code is taken from the following source
# https://medium.com/srm-mic/all-about-structural-similarity-index-ssim-theory-code-in-pytorch-6551b455541e

def get_gaussian_window(window_size,sigma):
    # get densities values of 2D gaussian distribution
        # define region
    x = np.arange(0,window_size)
    y = np.arange(0,window_size)
    mu = window_size // 2
    xx,yy = np.meshgrid(x,y)

    g_window = np.zeros((window_size,window_size))

    # fill the window
    for row,y in enumerate(yy):
        for col,x in enumerate(xx):
            g_window[row,col] = 1/(np.sqrt(2*np.pi)*sigma)*np.exp(-np.square(x[col]-mu)/(2*sigma**2))*\
                                1/(np.sqrt(2*np.pi)*sigma)*np.exp(-np.square(y[row]-mu)/(2*sigma**2))


    # normalize
    return torch.tensor(g_window/np.sum(g_window)).unsqueeze(0).expand(3,1,window_size,window_size).float()

# the input will be a video frame (C,W,H)
def SSIM(comp, gt, window):
    L = 1 # we normalized the image to [0,1]
    pad = window.shape[-1] // 2
    window = window.to(comp.device)
    
    # convolve input image with gaussian window, treat each RGB channel independently
    # this gets the localized means
    mu1 = F.conv2d(comp,window,padding=pad,groups=3)
    mu2 = F.conv2d(gt,window,padding=pad,groups=3)
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu12 = mu1*mu2

    # get the variance parameters
    sigma1_sq = F.conv2d(comp * comp,window,padding=pad,groups=3) - mu1_sq
    sigma2_sq = F.conv2d(gt * gt,window,padding=pad,groups=3) - mu2_sq
    sigma12 =  F.conv2d(comp * gt, window,padding=pad,groups=3) - mu12

    # constants 
    C1 = (0.01 ) ** 2
    C2 = (0.03 ) ** 2 

    contrast_metric = (2.0 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)
    contrast_metric = torch.mean(contrast_metric)

    numerator1 = 2 * mu12 + C1  
    numerator2 = 2 * sigma12 + C2
    denominator1 = mu1_sq + mu2_sq + C1 
    denominator2 = sigma1_sq + sigma2_sq + C2

    ssim_score = (numerator1 * numerator2) / (denominator1 * denominator2)

    return ssim_score.mean()

# expects the video volumes as inputs (N,C,D,W,H)
class SSIM_loss(nn.Module):
    def __init__(self,window_size=11,sigma=1.5) -> None:
        super(SSIM_loss,self).__init__()
        self.window = get_gaussian_window(window_size,sigma)

    def forward(self,output,target):
        # first reshape into (D,C,W,H)
        output = torch.permute(output[0],(1,0,2,3))
        target = torch.permute(target[0],(1,0,2,3))

        # average SSIM over all frames
        loss = 0
        for comp,gt in zip(output,target):
            loss += SSIM(comp,gt,self.window)
        return loss / len(output)




def train(model,train_loader,val_loader,device,loss_fn,optimizer,args):
    
    # init tensorboard
    writer = SummaryWriter()

    # create the optimizer
    #optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)#,weight_decay=0.00004,momentum=0.9)
    # # scheduler1 = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5,10,20], gamma=0.2)
    # scheduler2 = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.5)
    # scheduler1 = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    best_val_loss = 1e6

    model.train()
    model = model.to(device)
    batch_iter = 0

    for e in range(args.epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target)
            writer.add_scalar("Loss/train", loss, batch_iter)
            loss.backward()
            optimizer.step()

            print('Train Epoch: {} [{}/{} ({:.0f}%)] train loss: {:.3f}'.format(
                e, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss))#, scheduler1.get_last_lr()[0]))
            # if batch_idx > 0:
            #     scheduler1.step()
            batch_iter+=1
        # scheduler2.step()
            
        # evaluate on all the validation sets
        val_loss = validate(model,val_loader,device,loss_fn)
        writer.add_scalar("Loss/val", val_loss, batch_iter)
        print('Train Epoch: {} [{}/{} ({:.0f}%)] train loss: {:.3f}, val loss: {:.3f}'.format(
            e, batch_idx * len(data), len(train_loader.dataset),
            100. * batch_idx / len(train_loader), loss, val_loss))
        # scheduler1.step()
        if best_val_loss > val_loss:
            print("==================== best validation loss ====================")
            print("epoch: {}, val loss: {}".format(e,val_loss))
            best_val_loss = val_loss
            torch.save({
                'epoch': e+1,
                'model_state_dict': model.state_dict(),
                'val_loss': val_loss,
                }, 'models/best_batch_i'+str(batch_iter)+args.log_name+str(time.time())+'.pth')
        # scheduler2.step()

    
def validate(model, val_loader, device, loss_fn):
    model.eval()
    model = model.to(device)
    val_loss = 0

    with torch.no_grad():
        i=0
        for data, target in tqdm(val_loader):
            data, target = data.to(device), target.to(device)

            # Forward
            output = model(data)
            val_loss += loss_fn(output, target).item()  # sum up batch loss
            total_loss += val_loss
            batch_size = target.size(0)
            i+=1

        # Compute loss and accuracy
        val_loss /= i
        return val_loss


# ===================================== Command Line Arguments =====================================
def parse_args():
    parser = argparse.ArgumentParser(description="Training and Evaluation")

    # logging details
    parser.add_argument('--loss', type=str, default = 'L1', help='loss function')
    parser.add_argument('--opt', type=str, default = 'SGD', help='optimizer')
    parser.add_argument('--log_name', type=str, default = 'default', help='checkpoint file name')
    parser.add_argument('--batch-size', type=int, default=1, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                        help='how many batches to wait before logging training status')


    args = parser.parse_args()
    print(args)
    
    return args
    
# ===================================== Main =====================================
if __name__ == "__main__":
    # vd = VideoPairs("/home/gc28692/Projects/data/video_pairs",True,training=False,testing=True)
    # psnr = PSNR_loss()
    # ssim = SSIM_loss()
    # comp,gt = vd.__getitem__(20)
    # print(psnr(comp,gt))
    # print(ssim(comp,gt))
    # exit()
    vd = VideoPairs("/home/gc28692/Projects/data/video_pairs",True,training=False,testing=True)
    exit()
    print("=================")
    # get arguments
    args = parse_args()

    # setup device
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    if use_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    torch.manual_seed(args.seed)
    args.device = device

    # load datasets
    train_loader, val_loader, test_loader = load_video_pairs(args.batch_size,args.seed)

    # load the model
    model = ArtifactReduction()

    # set loss function
    if args.loss == "L1":
        loss = torch.nn.L1Loss()
    elif args.loss == "L2":
        loss = torch.nn.MSELoss()
    elif args.loss == "Huber":
        loss = torch.nn.HuberLoss()
    elif args.loss == "SSIM":
        loss = SSIM_loss()
    elif args.loss == "PSNR":
        loss = PSNR_loss()

    # set optimizer
    if args.opt == "SGD":
        opt = torch.optim.SGD(params=model.parameters(),lr=args.lr)
    elif args.opt == "Adam":
        opt = torch.optim.Adam(params=model.parameters(),lr=args.lr)

    train(model,train_loader,val_loader,device,loss,opt,args)