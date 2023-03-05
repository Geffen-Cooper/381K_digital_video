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
        pass

    # set optimizer
    if args.opt == "SGD":
        opt = torch.optim.SGD(params=model.parameters(),lr=args.lr)
    elif args.opt == "Adam":
        opt = torch.optim.Adam(params=model.parameters(),lr=args.lr)

    train(model,train_loader,val_loader,device,loss,opt,args)