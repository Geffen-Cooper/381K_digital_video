import argparse
import glob
import math
import random
import re
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import tensorboard as tb
from torchvision import models
from tqdm import tqdm
import time
from datasets import *
from models import *
import time


np.set_printoptions(linewidth=np.nan)



def IOU(bb_pred,bb_label):
    
    x_tlc_out = bb_pred[:,0].data
    y_tlc_out = bb_pred[:,1].data
    x_tlc_gt = bb_label[:,0].data
    y_tlc_gt = bb_label[:,1].data
    
    x_brc_out = bb_pred[:,0].data + bb_pred[:,2]
    y_brc_out = bb_pred[:,1].data + bb_pred[:,3]
    x_brc_gt = bb_label[:,0].data + bb_label[:,2]
    y_brc_gt = bb_label[:,1].data + bb_label[:,3]

    # find the max
    x_tlc = torch.max(x_tlc_out, x_tlc_gt)
    y_tlc = torch.max(y_tlc_out, y_tlc_gt)
    x_brc = torch.min(x_brc_out, x_brc_gt)
    y_brc = torch.min(y_brc_out, y_brc_gt)
    
    inter_area = torch.max(torch.zeros_like(x_brc), x_brc-x_tlc)*torch.max(torch.zeros_like(y_brc), y_brc-y_tlc)
    out_area = (x_brc_out-x_tlc_out)*(y_brc_out-y_tlc_out)
    label_area = (x_brc_gt-x_tlc_gt)*(y_brc_gt-y_tlc_gt)
    iou = inter_area / (label_area+out_area-inter_area)
    
    return iou.mean()


def train(loss,from_checkpoint,optimizer,log_name,root_dir,batch_size,epochs,ese,lr,use_cuda,seed,save_model_ckpt):

    writer = SummaryWriter("runs/" + log_name+"_"+str(time.time()))
    # log training parameters
    print("===========================================")
    for k,v in zip(locals().keys(),locals().values()):
        writer.add_text(f"locals/{k}", f"{v}")
        print(f"locals/{k}", f"{v}")
    print("===========================================")

    # ================== parse the arguments ==================
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Checkpoint path
    checkpoint_path = 'models/' + log_name + '.pth'

    # setup device
    use_cuda = use_cuda and torch.cuda.is_available()
    if use_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    # load datasets
    train_loader, val_loader, test_loader = load_palms(batch_size, seed, root_dir=root_dir)

    # load the model
    model = models.shufflenet_v2_x0_5(weights='DEFAULT')
    model.fc = torch.nn.Linear(1024,4)

    # set loss function
    if loss == "MSE":
        loss_fn = torch.nn.MSELoss()
    elif loss == "L1":
        loss_fn = torch.nn.L1Loss()
    elif loss == "L1smooth":
        loss_fn = torch.nn.SmoothL1Loss()
    else:
        raise NotImplementedError()

    # set optimizer
    if optimizer == "SGD":
        opt = torch.optim.SGD(params=model.parameters(), lr=lr)
    elif optimizer == "Adam":
        opt = torch.optim.Adam(params=model.parameters(), lr=lr)
    elif optimizer == "AdamW":
        opt = torch.optim.AdamW(params=model.parameters(), lr=lr)
    else:
        raise NotImplementedError()

    # continue training a model
    if from_checkpoint != None:
        checkpoint = torch.load(from_checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("resume from:", checkpoint['val_acc'])
        best_val_acc = checkpoint['val_acc']

    else:
        best_val_iou = 0
        lowest_loss = 1e6



    # ================== training loop ==================
    model.train()
    model = model.to(device)
    batch_iter = 0

    # how many epochs the validation loss did not decrease, 
    # used for early stopping
    num_epochs_worse = 0
    for e in range(epochs):
        if num_epochs_worse == ese:
            break
        for batch_idx, (data, target) in enumerate(train_loader):
            # stop training, run on the test set
            if num_epochs_worse == ese:
                break

            data, target = data.to(device), target.to(device)

            opt.zero_grad()
            model.train()

            output = model(data.float())

            train_loss = loss_fn(output, target)
            writer.add_scalar("Metric/train_" + loss, train_loss, batch_iter)
            train_loss.backward()
            opt.step()

            print('Train Epoch: {} [{}/{} ({:.0f}%)] train loss: {:.3f}'.format(
                e, batch_idx * train_loader.batch_size + len(data), len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader), train_loss))  # , scheduler1.get_last_lr()[0]))
            # if batch_idx > 0:
            #     scheduler1.step()
            batch_iter += 1
        # scheduler2.step()

        if num_epochs_worse == ese:
            print(f"Stopping training because accuracy did not improve after {num_epochs_worse} epochs")
            break

        # evaluate on the validation set
        val_iou, val_loss = validate(model, val_loader, device, loss_fn)

        writer.add_scalar("Metric/val_iou", val_iou, e)
        writer.add_scalar("Metric/val_loss", val_loss, e)

        print('Train Epoch: {} [{}/{} ({:.0f}%)] train loss: {:.3f}, val iou: {:.3f}, val loss: {:.3f}'.format(
            e, batch_idx * train_loader.batch_size + len(data), len(train_loader.dataset),
               100. * batch_idx / len(train_loader), train_loss, val_iou, val_loss))
        # scheduler1.step()

        if best_val_iou < val_iou:
        # if lowest_loss > val_loss:
            print("==================== best validation metric ====================")
            print("epoch: {}, val iou: {}, val loss: {}".format(e, val_iou, val_loss))
            best_val_iou = val_iou
            lowest_loss = val_loss
            torch.save({
                'epoch': e + 1,
                'model_state_dict': model.state_dict(),
                'val_iou': val_iou,
                'val_loss': val_loss,
            }, checkpoint_path)
            num_epochs_worse = 0
        # scheduler2.step()
        else:
            print(f"WARNING: {num_epochs_worse} num epochs without improving")
            num_epochs_worse += 1

    # evaluate on test set
    print("\n\n\n==================== TRAINING FINISHED ====================")
    print("Evaluating on test set")

    # load the best model
    model.load_state_dict(torch.load(checkpoint_path)['model_state_dict'])
    model.eval()
    test_iou, test_loss = validate(model, test_loader, device, loss_fn)

    print('test iou: {:.3f}, test loss: {:.3f}'.format(test_iou, test_loss))
    writer.add_scalar("Metric/test_iou", test_iou, e)
    writer.add_text("test_iou",f"{test_iou}")

    if not save_model_ckpt:
        os.remove(checkpoint_path)
    
    test_loader.dataset.visualize_batch(model)


def validate(model, val_loader, device, loss_fn):
    model.eval()
    model = model.to(device)

    val_loss = 0
    val_iou = 0

    with torch.no_grad():
        i = 0
        for idx, (data, target) in enumerate(val_loader):
            data, target = data.to(device), target.to(device)
            out = model(data.float())
            val_loss += loss_fn(out, target)
            val_iou += IOU(out,target)
            i += 1

        # Compute loss and accuracy
        val_loss /= i
        val_iou /= i

        return val_iou, val_loss

# ===================================== Main =====================================
if __name__ == "__main__":

    train_params = {'loss': "L1smooth", 'from_checkpoint': None, 'optimizer': "Adam", 'log_name': "baseline", 'root_dir': "palm_imgs",
                    'batch_size': 64, 'epochs': 50, 'ese': 5, 'lr': 0.001, 'use_cuda': True, 'seed': 42,'save_model_ckpt': True}

    train(**train_params)