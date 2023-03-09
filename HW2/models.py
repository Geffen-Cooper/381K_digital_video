import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR


def conv_block(num_filters, *args, **kwargs):
    return nn.Sequential(
        nn.Conv2d(num_filters, num_filters, *args, **kwargs),
        nn.BatchNorm2d(num_filters),
        nn.ReLU(),
        nn.Conv2d(num_filters, num_filters, *args, **kwargs),
        nn.BatchNorm2d(num_filters)
    )

# model for reducing compression artifacts in videos
class ArtifactReduction(nn.Module):
    def __init__(self):
        super(ArtifactReduction, self).__init__()

        # input volume is (N x C X D x H x W)
        # N (batch size) = 1
        # C (channel) = 3 RGB
        # D (depth) = input dependent, number of frames
        # H,W (height, width) = input dependent

        # we want to keep the dimensionality to avoid loosing spatial and temporal information

        # the kernel is 3D now where the 3rd dimension corresponds to how many frames we
        # want to convolve over since the kernel also slides in the time dimension

        # in the same way that the H,W are implicit in 2D, the H,W,D are implicit in 3D
        # so the filter will slide in all three dimensions depending on stride and padding,
        # use stride 1 and padding 1 to keep the volume the same dimensions (assuming kernel is 3x3x3)

        # the number of output channels corresponds to the number of filters since each filter
        # will collapse a volume into a single channel as it did in 2D 

        # (1 x 3 x 5 x 224 x 224) --> (1 x 8 x 5 x 224 x 224)
        self.conv1 = nn.Conv3d(in_channels=3, out_channels=8, kernel_size=3, padding=1)
        
        # (1 x 8 x 5 x 224 x 224) --> (1 x 16 x 5 x 224 x 224) --> (1 x 80 x 224 x 224)
        self.conv2 = nn.Conv3d(in_channels=8, out_channels=16, kernel_size=3, padding=1) 

        # (1 x 80 x 224 x 224) --> (1 x 64 x 224 x 224)
        self.conv3 = nn.Conv2d(in_channels=80, out_channels=64,kernel_size=3,padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        self.conv_block1 = conv_block(64,kernel_size=3, padding=1)
        self.conv_block2 = conv_block(64,kernel_size=3, padding=1)
        self.conv_block3 = conv_block(64,kernel_size=3, padding=1)
        self.conv_block4 = conv_block(64,kernel_size=3, padding=1)
        self.conv_block5 = conv_block(64,kernel_size=3, padding=1)
        self.conv_block6 = conv_block(64,kernel_size=3, padding=1)
        self.conv_block7 = conv_block(64,kernel_size=3, padding=1)
        self.conv_block8 = conv_block(64,kernel_size=3, padding=1)

        # (1 x 64 x 224 x 224) --> (1 x 3 x 224 x 224)
        self.conv_end = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, padding=1)

    def forward(self, x):
        # print("model input:",x.shape)
        x1 = self.conv1(x)
        x1 = F.relu(x1)

        x2 = self.conv2(x1)
        x2 = F.relu(x2)
        x2 = x2.view(x2.shape[0],-1,x2.shape[3],x2.shape[4])

        x3 = self.conv3(x2)
        x3 = self.bn3(x3)

        x = self.conv_block1(x3) + x3
        x = self.conv_block2(x) + x
        x = self.conv_block3(x) + x
        x = self.conv_block4(x) + x
        x = self.conv_block5(x) + x
        x = self.conv_block6(x) + x
        x = self.conv_block7(x) + x
        x = self.conv_block8(x) + x + x3
       
        x_end = self.conv_end(x)
        x_end = (x_end-x_end.min())/(x_end.max()-x_end.min())
        return x_end