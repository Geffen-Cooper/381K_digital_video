import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class Palms(Dataset):
    def __init__(self, root_dir,transform=None,train=True,test=False):
        """
        Args:
            root_dir (string): directory where the imgs are
            transform (callable, optional): transform to be applied on a sample
        """
        self.root_dir = root_dir
        self.transform = transform

        if train == True:
            subdir = "train"
        elif test == True:
            subdir = "test"

        # store imgs and labels
        self.img_paths = [] 
        self.labels = []

        img_files = os.listdir(os.path.join(root_dir, subdir))
        for file in img_files:
            params = file.split("_")
            x,y,w,h = int(params[0]),int(params[1]),int(params[2]),int(params[3].split("-")[0])
            self.img_paths.append(os.path.join(root_dir,subdir,file))
            self.labels.append(np.array([x,y,w,h]))


    def __getitem__(self, idx):
        # read the image
        img = Image.open(self.img_paths[idx])

        # get the label
        label = self.labels[idx]
        img_dim = img.size[0]
        label = label / img_dim

        # apply transform
        if self.transform:
            img = self.transform(img)
            
        # return the sample (img (tensor)), coords (x,y,w,h)
        return img, torch.tensor(label)

    def __len__(self):
        return len(self.img_paths)

    def visualize_batch(self,model=None):
        batch_size = 64
        data_loader = DataLoader(self,batch_size)

        # get the first batch
        (imgs, labels) = next(iter(data_loader))
        img_dim = imgs[0].shape[1]

        if model != None:
            with torch.no_grad():
                preds = model(imgs)
        
        # display the batch in a grid with the img, label, idx
        rows = 8
        cols = 8
        
        fig,ax_array = plt.subplots(rows,cols,figsize=(20,20))
        fig.subplots_adjust(hspace=0.2)
        for i in range(rows):
            for j in range(cols):
                idx = i*rows+j
                if idx == len(labels):
                    break

                ax_array[i,j].imshow((imgs[idx].permute(1, 2, 0)))
                
                rect = patches.Rectangle((labels[idx][0]*img_dim, labels[idx][1]*img_dim), labels[idx][2]*img_dim, labels[idx][3]*img_dim,\
                                          linewidth=1, edgecolor='g', facecolor='none')
                if model != None:
                    pred_rect = patches.Rectangle((preds[idx][0]*img_dim, preds[idx][1]*img_dim), preds[idx][2]*img_dim, preds[idx][3]*img_dim,\
                                          linewidth=1, edgecolor='r', facecolor='none')
                    ax_array[i,j].add_patch(pred_rect)
                
                # Add the patch to the Axes
                ax_array[i,j].add_patch(rect)

                ax_array[i,j].set_xticks([])
                ax_array[i,j].set_yticks([])
            if idx == len(labels):
                    break
        plt.show()


# used to apply augmentation to training only
class SubsetWrapper(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, label = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, label

    def __len__(self):
        return len(self.subset)


def load_palms(batch_size, rand_seed, root_dir):
    # create transforms
    train_tsfms = transforms.Compose([
        transforms.ColorJitter(brightness=.25,contrast=.25,saturation=.25,hue=.25),
        transforms.ToTensor()
        ]
    )

    test_tsfms = transforms.Compose([
        transforms.ToTensor()
        ]
    )

    dataset = Palms(root_dir, None, train=True)
    train_subset, val_subset = torch.utils.data.random_split(dataset, [int(len(dataset) * 0.9), len(dataset) - int(len(dataset) * .9)])
    
    train_set = SubsetWrapper(train_subset, transform=train_tsfms)
    val_set = SubsetWrapper(val_subset, transform=test_tsfms)
    test_set = Palms(root_dir, test_tsfms, train=False, test=True)

    # create the data loaders
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=64)
    test_loader = DataLoader(test_set, batch_size=64)

    # return test_loader
    return (train_loader, val_loader, test_loader)

if __name__ == "__main__":
    # create transforms
    train_tsfms = transforms.Compose([
        transforms.ColorJitter(brightness=.25,contrast=.25,saturation=.25,hue=.25),
        transforms.ToTensor()
        ]
    )

    # create dataset
    dataset = Palms(root_dir="palm_imgs",transform=train_tsfms,train=True)
    dataset.visualize_batch()
