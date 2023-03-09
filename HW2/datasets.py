import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import os
import copy
import numpy as np
import cv2
import time

from train import *

"""
Helper function to read video frames

video_path (string): full path to video file
is_img_frames (bool): video is set of image files rather than video file
patch_size (tuple): (w,h) of square patch to extract from video 
patch_coord (tuple): (x,y) top left coordinate of patch to extract
num_frames (int): number of frames to get from the video

by default the function will return the full first frame of the video
"""
def get_video_frames(video_path,is_img_frames,patch_size=None,patch_coord=None,num_frames=None):
    if is_img_frames == False:
        # create video capture object
        cap = cv2.VideoCapture(video_path)
            
        # Check if video opened successfully
        if (cap.isOpened()== False): 
            print("Error opening video file")
            exit()

        # read the first frame
        ret, frame = cap.read()

        if ret:
            # by defaut return the first frame
            if patch_size == None and patch_coord == None and num_frames == None:
                # release the video capture object
                cap.release()
                return frame

            else:
                if num_frames == None:
                    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

                # create frame buffer
                frames = np.empty((num_frames, patch_size[1], patch_size[0], 3), np.dtype('uint8'))
                frame_count = 0

                # store the first frame
                start_x,start_y = patch_coord[0], patch_coord[1]
                end_x, end_y = start_x + patch_size[0], start_y + patch_size[1]

                frames[frame_count] = frame[start_y:end_y,start_x:end_x,:]
                frame_count += 1

                # store the remaning frames
                while (frame_count < num_frames and ret):
                    start_x,start_y = patch_coord[0], patch_coord[1]
                    end_x, end_y = start_x + patch_size[0], start_y + patch_size[1]
                    try:
                        frames[frame_count] = frame[start_y:end_y,start_x:end_x,:]
                    except:
                        print(frame_count)
                        print(frame,ret)
                    frame_count += 1
                    ret, frame = cap.read()
                return frames
        else:
            print("can't get first frame")
            exit()
    else:
        files = video_path
        num_frames = len(files)

        # create frame buffer
        frames = np.empty((num_frames, patch_size[1], patch_size[0], 3), np.dtype('uint8'))
        frame_count = 0

        for file in files:
            frame = cv2.imread(file)
            # store the first frame
            start_x,start_y = patch_coord[0], patch_coord[1]
            end_x, end_y = start_x + patch_size[0], start_y + patch_size[1]

            frames[frame_count] = frame[start_y:end_y,start_x:end_x,:]
            frame_count += 1
        return frames



class VideoPairs(Dataset):
    """Video Pairs Dataset.
    """

    def __init__(self, root_dir,img_frames=True,patch_size=224,overlap_ratio=0.5,frame_depth=10,transform=None,training=True,validation=False,testing=False):
        """
        Args:
            root_dir (string): directory where the training/validation/test splits are
            img_frames (bool): videos are stored as individual frames rather than a video file
            patch_size (tuple): (w,h) the crop dimension to use
            overlap_ratio (float): the amount of overlap between crops
            transform (callable, optional): transform to be applied on a sample
            training (bool): specify if to load the training data
            validation (bool): specify if to load the validation data
            testing (bool): specify if to load the testing data
        """
        self.root_dir = root_dir
        self.img_frames = img_frames
        self.transform = transform
        self.patch_size = patch_size
        self.overlap_ratio = overlap_ratio

        if training:
            self.sub_dir = "training"
        elif validation:
            self.sub_dir = "validation"
        elif testing:
            self.sub_dir = "test"

        self.video_path = os.path.join(self.root_dir,self.sub_dir)

        if img_frames == False:
            self.comp_video_files = []
            self.gt_video_files = []
            for f in os.listdir(os.path.join(self.video_path,"compressed")):
                self.comp_video_files.append(os.path.join(self.video_path,"compressed",f))
            for f in os.listdir(os.path.join(self.video_path,"GT")):
                self.gt_video_files.append(os.path.join(self.video_path,"GT",f))

            self.num_videos = len(self.comp_video_files)
            first_frame = get_video_frames(self.comp_video_files[0],False) 
        else:
            # store files for each subfolder
            subfolders = os.listdir(os.path.join(self.video_path,"compressed"))

            self.comp_video_files = [[] for i in range(len(subfolders))]
            self.gt_video_files = [[] for i in range(len(subfolders))]

            for i,folder in enumerate(subfolders):
                files = os.listdir(os.path.join(self.video_path,"compressed",folder))
                files = sorted(files,key=lambda x: int(os.path.splitext(x)[0]))
                
                for file in files:
                    self.comp_video_files[i].append(os.path.join(self.video_path,"compressed",folder,file))
                    self.gt_video_files[i].append(os.path.join(self.video_path,"GT",folder,file))

            self.num_videos = len(self.comp_video_files)
            first_frame = cv2.imread(self.comp_video_files[0][0])

        # We need to generate a mapping from the sample idx to a video crop
        # For each video we will identify how many patch_x by patch_y regions fit in the video WxH,
        # using an overlap of overlap_ratio. If the WxH is not divisible by overlap_px, then for the last
        # crop just fit the remaining portion.
        
        # video dimensions
        self.vid_h, self.vid_w = first_frame.shape[:2]

        # now get the coordinates of each patch using the given overlap ratio
        x_overlap_px = int(self.patch_size[0]*overlap_ratio)
        num_patches_horizontal = (self.vid_w - patch_size[0]) // (patch_size[0]-x_overlap_px)
        y_overlap_px = int(self.patch_size[1]*overlap_ratio)
        num_patches_vertical = (self.vid_h - patch_size[1]) // (patch_size[1]-y_overlap_px)
        

        # patch doesn't fit evenly
        if (self.vid_w - patch_size[0]) % (patch_size[0]-x_overlap_px) != 0:
            num_patches_horizontal += 1
            x_coords = np.concatenate([np.arange(0,self.vid_w-patch_size[0],(patch_size[0]-x_overlap_px)),np.array([self.vid_w-patch_size[0]])])
        else:
            x_coords = np.arange(0,self.vid_w,(patch_size[0]-x_overlap_px))
        if (self.vid_h - patch_size[1]) % (patch_size[1]-y_overlap_px) != 0:
            num_patches_vertical += 1
            y_coords = np.concatenate([np.arange(0,self.vid_h-patch_size[1],(patch_size[1]-y_overlap_px)),np.array([self.vid_h-patch_size[1]])])
        else:
            y_coords = np.arange(0,self.vid_h,(patch_size[1]-y_overlap_px))
        
        # we use these patch coordinates to get crops as samples
        self.patch_coords_x, self.patch_coords_y = np.meshgrid(x_coords, y_coords)

    def __getitem__(self, idx):
        # map the idx into a 3D coordinate (x,y,video number)
        w,h = self.patch_coords_x.shape[1], self.patch_coords_x.shape[0]
        video_number = idx // (w*h)
    
        x_coord = (idx - video_number*(w*h)) % w
        y_coord = (idx - video_number*(w*h)) // w
        # print("video num:",video_number)
        # print(self.comp_video_files[video_number][0])

        # scale to pixel coordinates
        x_coord = self.patch_coords_x[0,x_coord]
        y_coord = self.patch_coords_y[y_coord,0]

        # read video patch, need to convert to tensor, current shape is (D,W,H,C)
        compressed = get_video_frames(self.comp_video_files[video_number],self.img_frames,self.patch_size,(x_coord,y_coord))
        ground_truth = get_video_frames(self.gt_video_files[video_number],self.img_frames,self.patch_size,(x_coord,y_coord)) 
        
        # now apply transforms, convert to tensor and permute dimensions to get (C,D,W,H)
        compressed = torch.tensor(compressed.transpose(3,0,1,2)).float().div(255.0)
        ground_truth = torch.tensor(ground_truth.transpose(3,0,1,2)).float().div(255.0)

        return compressed, ground_truth

    def __len__(self):
        return self.patch_coords_x.shape[0]*self.patch_coords_x.shape[1]*self.num_videos

    def visualize_sample(self,model=None):
        # to visualize from the trained model just pass it through the model before conversion
        # also print out the PSNR and SSIM before and after passing through the model
        comp,gt = self.__getitem__(10)#torch.randperm(len(self))[0])
        comp = comp.unsqueeze(0).to('cuda')
        gt = gt.unsqueeze(0).to('cuda')
        aft = torch.clone(comp)
        with torch.no_grad():
            print("before")
            print(PSNR_metric()(comp,gt))
            print(SSIM_metric()(comp,gt))
            if model != None: 
                model.to('cuda')
                frame_depth = 5
                patch_size = self.patch_size
                # go over each frame
                for depth_coord in range(100):
                    # if at the beginning add padding
                    if depth_coord < frame_depth//2:
                        frames = torch.cat([torch.zeros((1,3,frame_depth//2-depth_coord,patch_size[1],patch_size[0])).to('cuda'),comp[:,:,:depth_coord+frame_depth//2+1,:,:]],dim=2)
                    # if at the end add padding
                    elif (99-depth_coord) < frame_depth//2:
                        frames = torch.cat([comp[:,:,depth_coord-frame_depth//2:,:,:],torch.zeros((1,3,depth_coord-99+frame_depth//2,patch_size[1],patch_size[0])).to('cuda')],dim=2)
                    # otherwise get frames [t-2,t-1,t,t+1,t+2] where t is ground truth
                    else:
                        # get frame_depth frames for training, and the ground truth is the middle frame
                        frames = comp[:,:,depth_coord-frame_depth//2:depth_coord+frame_depth//2+1,:,:]
                
                    # forward, output is a single frame
                    output = model(frames)
                    aft[:,:,depth_coord,:,:] = (output-output.min())/(output.max()-output.min())
                print("after")
                print(PSNR_metric()(aft,gt))
                print(SSIM_metric()(aft,gt))
            else:
                aft = gt
        # inputs are (N,C,D,W,H), need to convert back to (D,W,H,C) and uint8
        
        comp = (torch.permute(comp[0].to('cpu'),(1,2,3,0)).numpy()*255).astype(np.uint8)
        aft = (torch.permute(aft[0].to('cpu'),(1,2,3,0)).numpy()*255).astype(np.uint8)
        # gt = (torch.permute(gt[0],(1,2,3,0)).numpy()*255).astype(np.uint8)

        print(comp.shape,aft.shape)
        out_comp = cv2.VideoWriter('out_comp.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 30, self.patch_size)
        out_aft = cv2.VideoWriter('out_aft.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 30, self.patch_size)
        # out_gt = cv2.VideoWriter('out_gt.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 30, self.patch_size)

        i = 0
        # for frame_comp, frame_gt in zip(comp,gt):
        #     out_comp.write(frame_comp)
        #     out_gt.write(frame_gt)
        for frame_comp, frame_aft in zip(comp,aft):
            out_comp.write(frame_comp)
            out_aft.write(frame_aft)

        out_comp.release()
        out_aft.release()
        # out_gt.release()

        
        cap_comp = cv2.VideoCapture("out_comp.avi")
        cap_aft = cv2.VideoCapture("out_aft.avi")
        # cap_gt = cv2.VideoCapture("out_gt.avi")

        while(True):
            ret_comp, frame_comp = cap_comp.read()
            ret_aft, frame_aft = cap_aft.read()
            # ret_gt, frame_gt = cap_gt.read()

            if ret_comp == True and ret_aft == True:#ret_gt == True: 
                
                # Display the resulting frame    
                cv2.imshow('comp',frame_comp)
                cv2.imshow('aft',frame_aft)
                # cv2.imshow('gt',frame_gt)
                
                if cv2.waitKey(40) & 0xFF == ord('q'):
                    break
            
            # Break the loop
            else:
                cap_comp.set(cv2.CAP_PROP_POS_FRAMES, 0)
                cap_aft.set(cv2.CAP_PROP_POS_FRAMES, 0)
                # cap_gt.set(cv2.CAP_PROP_POS_FRAMES, 0)
                # break 


def load_video_pairs(batch_size,rand_seed):
    root_dir = "/home/gc28692/Projects/data/video_pairs"

    vd_train = VideoPairs(root_dir,True,(224,224),0.0,None,training=True,validation=False,testing=False)
    vd_val = VideoPairs(root_dir,True,(1280,720),0.0,None,training=False,validation=True,testing=False)
    vd_test = VideoPairs(root_dir,True,(1280,720),0.0,None,training=False,validation=False,testing=True)

    # create the data loaders
    train_loader = torch.utils.data.DataLoader(vd_train, batch_size=batch_size, shuffle=True,num_workers=4)
    val_loader = torch.utils.data.DataLoader(vd_val, batch_size=1)
    test_loader = torch.utils.data.DataLoader(vd_test, batch_size=1)

    
    # return test_loader
    return (train_loader, val_loader, test_loader)

# vd = VideoPairs("/home/gc28692/Projects/data/video_pairs",True,training=True,testing=False,overlap_ratio=0.0,patch_size=(1280,720))
# print("num crop samples",len(vd))
# vd.visualize_sample()