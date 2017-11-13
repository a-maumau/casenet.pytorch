import torch 
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable

# 3x3 Convolution
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                     stride=stride, padding=1, bias=False)
# 1x1 Convolution
def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                     stride=stride, padding=0, bias=False)

def shared_concat(predic, map1, map2, map3, class_num):
    # concatting
    map1 = map1.squeeze(1)
    map2 = map2.squeeze(1)
    map3 = map3.squeeze(1)
    cat = []
    
    for i in range(class_num):
        cat.append(predic[:, i, :])
        cat.append(map1)
        cat.append(map2)
        cat.append(map3)
    
    return torch.stack(tuple(cat) ,1)
    
# Residual Block
class ResidualBlock(nn.Module):
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv1x1(in_channels, out_channels)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = conv3x3(out_channels, out_channels, stride)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = conv1x1(out_channels, out_channels * self.expansion)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out

# ResNet Base
class Model(nn.Module):
    def __init__(self, block, layers, class_num):
        super(Model, self).__init__()

        self.class_num = class_num # COCO:184, SBD:20

        self.in_channels = 64
        self.conv = nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        # ouput from ReLU 64
        self.fmap1 = nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0, bias=False)
        #self.upsample1 = nn.Upsample(scale_factor=1)
        # it seems like nn.UpsamplingBilinear2d is deprecated.
        #self.upsample1 = nn.UpsamplingBilinear2d(scale_factor=1)
        
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self.make_layer(block, 64, layers[0])
        # output from layer1 is 256
        self.fmap2 =  nn.Conv2d(256, 1, kernel_size=1, stride=1, padding=0, bias=False)
        self.upsample2 = nn.Upsample(scale_factor=2)
        
        # In original Resnet, ouput is 128
        self.layer2 = self.make_layer(block, 128, layers[1], 2)
        # ouput from layer2 is 1024
        self.fmap3 = nn.Conv2d(512, 1, kernel_size=1, stride=1, padding=0, bias=False)
        self.upsample3 = nn.Upsample(scale_factor=4)
        
        """
            in the paper, their figure shows output is 2048.
            but in the caffe code witch is provided is 1024, so it should be the same as the original Resnet101.

        self.layer2 = self.make_layer(block, 512, layers[1], 2)
        # ouput from layer2 is 1024
        self.fmap3 = nn.Conv2d(1024, 1, kernel_size=1, stride=1, padding=0, bias=False)
        self.upsample3 = nn.Upsample(scale_factor=4)
        """
        
        self.layer3 = self.make_layer(block, 256, layers[2], 2)

        self.layer4 = self.make_layer(block, 512, layers[3], 1)
        self.predict = nn.Conv2d(2048, self.class_num, kernel_size=1, stride=1, padding=0, bias=False)
        
        self.fuse_classification = nn.Conv2d(self.class_num*4, self.class_num, kernel_size=1, stride=1, padding=0, bias=False, groups=self.class_num)
        self.upsample4 = nn.Upsample(scale_factor=8)

    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels*block.expansion):
            downsample = nn.Sequential(
                conv1x1(self.in_channels, out_channels*block.expansion, stride=stride),
                nn.BatchNorm2d(out_channels*block.expansion))

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))

        self.in_channels = out_channels*block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        f1 = self.fmap1(out)
        #f1 = self.upsample1(f1) #same size
        
        out = self.maxpool(out)
        out = self.layer1(out)
        f2 = self.fmap2(out)
        f2 = self.upsample2(f2)
        
        out = self.layer2(out)
        f3 = self.fmap3(out)
        f3 = self.upsample3(f3)
        
        out = self.layer3(out)

        out = self.layer4(out)

        out = self.predict(out)
        side_classification = self.upsample4(out)
        
        out = shared_concat(side_classification, f1, f2, f3, self.class_num)
        
        return self.fuse_classification(out) , side_classification