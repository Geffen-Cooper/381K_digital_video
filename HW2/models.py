import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

# model for reducing compression artifacts in videos
class ArtifactReduction(nn.Module):
    def __init__(self):
        super(ArtifactReduction, self).__init__()

        # input volume is (N x C X D x H x W)
        # N (batch size) = 1
        # C (channel) = 3 RGB
        # D (depth) = input dependent, number of frames
        # H,W (height, width) = input dependent

        # we want to keep the dimensionality to avoid loosing spatial and temporal
        # information since the output is also a video volume

        # the kernel is 3D now where the 3rd dimension corresponds to how many frames we
        # want to convolve over since the kernel also slides in the time dimension

        # in the same way that the H,W are implicit in 2D, the H,W,D are implicit in 3D
        # so the filter will slide in all three dimensions depending on stride and padding,
        # use stride 1 and padding 1 to keep the volume the same dimensions

        # the number of output channels corresponds to the number of filters since each filter
        # will collapse a volume into a single channel as it did in 2D 

        # (1 x 3 x 100 x 224 x 224) --> (1 x 16 x 100 x 224 x 224)
        self.conv1 = nn.Conv3d(in_channels=3, out_channels=8, kernel_size=3, padding=1) 
        
        # (1 x 16 x 100 x 224 x 224) --> (1 x 32 x 100 x 224 x 224)
        self.conv2 = nn.Conv3d(in_channels=8, out_channels=16, kernel_size=3, padding=1) 

        # (1 x 32 x 100 x 224 x 224) --> (1 x 3 x 100 x 224 x 224)
        self.conv3 = nn.Conv3d(in_channels=16, out_channels=3, kernel_size=3, padding=1) 

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)

        x = self.conv3(x)

        return x