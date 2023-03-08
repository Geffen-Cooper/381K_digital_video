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
import time

# the input will be a video frame (C,W,H)
def PSNR(comp, gt):
    mse = F.mse_loss(comp,gt)
    max_val = 1 # videos are uint8 
    return 10*torch.log10(max_val**2/(mse+1e-12))

# expects frames as inputs (N,C,1,W,H)
class PSNR_metric(nn.Module):
    def __init__(self) -> None:
        super(PSNR_metric,self).__init__()

    def forward(self,output,target):
        # average PSNR over all frames in the batch
        loss = 0
        for comp_f,gt_f in zip(output,target):
            # print(comp_f.shape,gt_f.shape)
            loss += PSNR(comp_f,gt_f)
        return loss / (output.shape[0])


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
    mu1 = F.conv2d(comp.unsqueeze(0),window,padding=pad,groups=3)
    mu2 = F.conv2d(gt.unsqueeze(0),window,padding=pad,groups=3)
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu12 = mu1*mu2

    # get the variance parameters
    sigma1_sq = F.conv2d((comp * comp).unsqueeze(0),window,padding=pad,groups=3) - mu1_sq
    sigma2_sq = F.conv2d((gt * gt).unsqueeze(0),window,padding=pad,groups=3) - mu2_sq
    sigma12 =  F.conv2d((comp * gt).unsqueeze(0), window,padding=pad,groups=3) - mu12

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
class SSIM_metric(nn.Module):
    def __init__(self,window_size=11,sigma=1.5,loss=False) -> None:
        super(SSIM_metric,self).__init__()
        self.window = get_gaussian_window(window_size,sigma)
        self.loss = loss

    def forward(self,output,target):
        score = 0
        # average SSIM over all frames in batch
        for comp_f,gt_f in zip(output,target):
            # print(comp_f.shape,gt_f.shape)
            score += SSIM(comp_f,gt_f,self.window)
        avg = score / (output.shape[0])

        # we want to maximize SSIM so use negative
        # of it to use as a loss that we minimize
        if self.loss == True:
            return -avg
        else:
            return avg


def train(model,train_loader,val_loader,test_loader,device,loss_fn,optimizer,args):
    if args.checkpoint != "None":
        # init tensorboard
        args.log_name = args.log_name+"resume"
        writer = SummaryWriter("runs/"+args.log_name)
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("resume from:",checkpoint['val_metric'])
        best_val_metric = checkpoint['val_metric']
    else:
        writer = SummaryWriter("runs/"+args.log_name)
        best_val_metric = 0

    # create the optimizer
    #optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)#,weight_decay=0.00004,momentum=0.9)
    # # scheduler1 = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5,10,20], gamma=0.2)
    # scheduler2 = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.5)
    # scheduler1 = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    

    model.train()
    model = model.to(device)
    batch_iter = 0

    # how many epochs the validation loss did not decrease, 
    # used for early stopping
    num_epochs_worse = 0
    for e in range(args.epochs):
        if num_epochs_worse == args.ese:
                break
        for batch_idx, (data, target) in enumerate(train_loader):
            # stop training, run on the test set
            if num_epochs_worse == args.ese:
                break

            # we get 5 frames
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()

            # the model outputs the prediction for a single frame [t-2,t-1,t,t+1,t+2]
            output = model(data)
            loss = loss_fn(output, target)
            writer.add_scalar("Metric/train_"+args.loss, loss, batch_iter)
            loss.backward()
            optimizer.step()

            if (batch_iter % 80) == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)] train loss: {:.3f}'.format(
                    e, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss))#, scheduler1.get_last_lr()[0]))
            # if batch_idx > 0:
            #     scheduler1.step()
            

            if (batch_iter % args.log_interval) == 0:
                print("\nStarting Validation")
                # evaluate on all the validation sets
                val_metrics = validate(model,val_loader,device,[PSNR_metric(),SSIM_metric()])
                val_metric_PSNR = val_metrics[0]
                val_metric_SSIM = val_metrics[1]
                
                writer.add_scalar("Metric/val_PSNR", val_metric_PSNR, batch_iter)
                writer.add_scalar("Metric/val_SSIM", val_metric_SSIM, batch_iter)
                
                print('********Validation, Epoch: {} [{}/{} ({:.0f}%)] train loss: {:.3f}, val metric PSNR: {:.3f}, val metric SSIM: {:.3f}'.format(
                    e, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss, val_metric_PSNR,val_metric_SSIM))
                # scheduler1.step()

                if args.val_metric == "PSNR":
                    val_metric = val_metric_PSNR
                elif args.val_metric == "SSIM":
                    val_metric = val_metric_SSIM

                if best_val_metric < val_metric:
                    print("==================== best validation metric ====================")
                    print("Validation, epoch: {}, val loss: {}".format(e,val_metric))
                    best_val_metric = val_metric
                    torch.save({
                        'epoch': e+1,
                        'model_state_dict': model.state_dict(),
                        'val_metric': val_metric,
                        }, 'models/'+args.log_name+'.pth')
                    num_epochs_worse = 0
                else:
                    print("WARNING: validation metric decreased", num_epochs_worse+1, "times")
                    num_epochs_worse += 1
            batch_iter+=args.batch_size
        # scheduler2.step()
            
        if num_epochs_worse == args.ese:
            break
        # evaluate on all the validation sets
        val_metrics = validate(model,val_loader,device,[PSNR_metric(),SSIM_metric()])
        val_metric_PSNR = val_metrics[0]
        val_metric_SSIM = val_metrics[1]
        
        writer.add_scalar("Metric/val_PSNR", val_metric_PSNR, batch_iter)
        writer.add_scalar("Metric/val_SSIM", val_metric_SSIM, batch_iter)
        
        print('Train Epoch: {} [{}/{} ({:.0f}%)] train loss: {:.3f}, val matric PSNR: {:.3f}, val metric SSIM: {:.3f}'.format(
            e, batch_idx * len(data), len(train_loader.dataset),
            100. * batch_idx / len(train_loader), loss, val_metric_PSNR,val_metric_SSIM))
        # scheduler1.step()

        if args.val_metric == "PSNR":
            val_metric = val_metric_PSNR
        elif args.val_metric == "SSIM":
            val_metric = val_metric_SSIM

        if best_val_metric < val_metric:
            print("==================== best validation metric ====================")
            print("epoch: {}, val loss: {}".format(e,val_metric))
            best_val_metric = val_metric
            torch.save({
                'epoch': e+1,
                'model_state_dict': model.state_dict(),
                'val_metric': val_metric,
                }, 'models/'+args.log_name+'.pth')
            num_epochs_worse = 0
        # scheduler2.step()
        else:
            num_epochs_worse += 1
    
    # evaluate on test set
    print("\n\n\n==================== TRAINING FINISHED ====================")
    print("Evaluating on test set")

    # load the best model
    model = ArtifactReduction()
    model.load_state_dict(torch.load('models/'+args.log_name+'.pth')['model_state_dict'])
    model.eval()
    test_metrics = validate(model,test_loader,device,[PSNR_metric(),SSIM_metric()])
    
    print('test metric PSNR: {:.3f}, test metric SSIM: {:.3f}'.format(
          test_metrics[0],test_metrics[1]))

    writer.add_scalar("Metric/val_PSNR", val_metric_PSNR, batch_iter)
    writer.add_scalar("Metric/val_SSIM", val_metric_SSIM, batch_iter)
    
def validate(model, val_loader, device, loss_fns):
    model.eval()
    model = model.to(device)
    val_metrics = torch.zeros(len(loss_fns))
    # the loss functions take care of reshaping the input
    with torch.no_grad():
        i=0
        # batch items are is (1,3,5,720,1280) and (1,3,720,1280)
        start = time.time()
        for comp, gt in tqdm(val_loader):
            comp, gt = comp.to(device), gt.to(device)
            print("load:",time.time()-start)

            # Forward
            start = time.time()
            output = model(comp)
            print("forward:",time.time()-start)

            start = time.time()
            # get value for each metric
            for k,loss_fn in enumerate(loss_fns):
                val_metrics[k] += loss_fn(output, gt).item() 
            i+=1
            print("metrics:",time.time()-start)

            start = time.time()

        # Compute loss and accuracy
        print(i)
        val_metrics /= i
        exit()
        return val_metrics


# ===================================== Command Line Arguments =====================================
def parse_args():
    parser = argparse.ArgumentParser(description="Training and Evaluation")

    # logging details
    parser.add_argument('--loss', type=str, default = 'L1', help='loss function')
    parser.add_argument('--checkpoint', type=str, default = 'None', help='checkpoint to resume from')
    parser.add_argument('--val-metric', type=str, default = 'PSNR', help='validation metric')
    parser.add_argument('--opt', type=str, default = 'SGD', help='optimizer')
    parser.add_argument('--log_name', type=str, default = 'default', help='checkpoint file name')
    parser.add_argument('--batch-size', type=int, default=1, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train')
    parser.add_argument('--ese', type=int, default=3, metavar='N',
                        help='early stopping epochs')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                        help='how many batches to wait before logging training status')


    args = parser.parse_args()
    print(args)
    
    return args
    
# ===================================== Main =====================================
if __name__ == "__main__":
    # vd = VideoPairs("/home/gc28692/Projects/data/video_pairs",True,training=False,validation=True,patch_size=(1280,720),overlap_ratio=0)
    # psnr = PSNR_loss()
    # ssim = SSIM_loss()
    # comp,gt = vd.__getitem__(1)
    # print(psnr(comp,gt))
    # print(ssim(comp,gt))
    # vd.visualize_sample()
    # exit()

    # vd = VideoPairs("/home/gc28692/Projects/data/video_pairs",True,training=False,validation=True,patch_size=(1280,720),overlap_ratio=0)
    # model = ArtifactReduction()
    # model.load_state_dict(torch.load("models/SSIM_SSIM_SGD_res.pth")['model_state_dict'])
    # vd.visualize_sample()
    # exit()
    print("=================")
    # get arguments
    args = parse_args()

    # setup device
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    if use_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # torch.manual_seed(args.seed)
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
        loss = SSIM_metric(loss=True)
    elif args.loss == "PSNR":
        loss = PSNR_metric()

    # set optimizer
    if args.opt == "SGD":
        opt = torch.optim.SGD(params=model.parameters(),lr=args.lr,momentum=0.0)
    elif args.opt == "Adam":
        opt = torch.optim.Adam(params=model.parameters(),lr=args.lr)

    train(model,train_loader,val_loader,test_loader,device,loss,opt,args)