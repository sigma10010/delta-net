import argparse
import os
import shutil
import time, math
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
import torch.utils.model_zoo as model_zoo
from torch.autograd.variable import Variable

from models.kan import KAN
from models.se import SELayer
from models.bam import BAM
from models.cbam import CBAM
from models.sk import SKConv

'''
Pytorch model for the iTracker.

Author: Petr Kellnhofer ( pkel_lnho (at) gmai_l.com // remove underscores and spaces), 2018. 

Website: http://gazecapture.csail.mit.edu/

Cite:

Eye Tracking for Everyone
K.Krafka*, A. Khosla*, P. Kellnhofer, H. Kannan, S. Bhandarkar, W. Matusik and A. Torralba
IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016

@inproceedings{cvpr2016_gazecapture,
Author = {Kyle Krafka and Aditya Khosla and Petr Kellnhofer and Harini Kannan and Suchendra Bhandarkar and Wojciech Matusik and Antonio Torralba},
Title = {Eye Tracking for Everyone},
Year = {2016},
Booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)}
}

'''

class EyeImageModel(nn.Module):
    # Used for both eyes (with shared weights) and the face (with unqiue weights)
    # parameters based on SAGE: On-device Few-shot Personalization for Real-time Gaze Estimation
    def __init__(self):
        super(EyeImageModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=7),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # SELayer(32,4),
            BAM(32,4),
            # CBAM(32,reduction_ratio=4),
            # SKConv(32),
            nn.Conv2d(32, 32, kernel_size=5),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # SELayer(32,4),
            BAM(32,4),
            # CBAM(32,reduction_ratio=4),
            # SKConv(32),
            nn.Conv2d(32, 32, kernel_size=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # SELayer(32,4),
            BAM(32,4),
            # CBAM(32,reduction_ratio=4),
        )
        
        self.fc = nn.Sequential(
            nn.Linear(9*9*32, 128),
            # nn.Linear(13*13*32, 128), # for skconv
            nn.ReLU(inplace=True),
            nn.Linear(128, 16),
            nn.ReLU(inplace=True),
            )
        '''
        self.fc = nn.Sequential(
            KAN(layers_hidden = [9*9*32,128,16]),
            nn.ReLU(inplace=True),
            )
        '''

    def forward(self, x):
        x = self.features(x)
        # print(x.shape)
        x = x.view(x.size(0), -1) # (n,2592)
        x = self.fc(x)
        return x


class ItrackerImageModel(nn.Module):
    # Used for both eyes (with shared weights) and the face (with unqiue weights)
    def __init__(self):
        super(ItrackerImageModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.CrossMapLRN2d(size=5, alpha=0.0001, beta=0.75, k=1.0),
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2, groups=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.CrossMapLRN2d(size=5, alpha=0.0001, beta=0.75, k=1.0),
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 64, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return x
      
class FaceImageModelOrigin(nn.Module):
    
    def __init__(self):
        super(FaceImageModelOrigin, self).__init__()
        self.conv = ItrackerImageModel()
        '''
        self.fc = nn.Sequential(
            nn.Linear(12*12*64, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            )
        '''
        self.fc = KAN(layers_hidden = [12*12*64,128,64])

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x

class FaceImageModel(nn.Module):
    
    def __init__(self):
        super(FaceImageModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=7),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # SELayer(32,4),
            BAM(32,4),
            # CBAM(32,reduction_ratio=4),
            nn.Conv2d(32, 32, kernel_size=5),
            # SKConv(32),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # SELayer(32,4),
            BAM(32,4),
            # CBAM(32,reduction_ratio=4),
            nn.Conv2d(32, 32, kernel_size=3),
            # SKConv(32),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # SELayer(32,4),
            BAM(32,4),
            # CBAM(32,reduction_ratio=4),
            nn.Conv2d(32, 32, kernel_size=3),
            # SKConv(32),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # SELayer(32,4),
            BAM(32,4),
            # CBAM(32,reduction_ratio=4),
        )
        
        self.fc = nn.Sequential(
            nn.Linear(10*10*32, 128),
            # nn.Linear(12*12*32, 128), # for skconv
            nn.ReLU(inplace=True),
            nn.Linear(128, 16),
            nn.ReLU(inplace=True),
            )
        '''
        self.fc = nn.Sequential(
            KAN(layers_hidden = [10*10*32,128,16]),
            nn.ReLU(inplace=True),
            )
        '''

    def forward(self, x):
        x = self.features(x)
        # print(x.shape)
        x = x.view(x.size(0), -1) # (n,3200)
        x = self.fc(x)
        return x

class FaceGridModel(nn.Module):
    # Model for the face grid pathway
    def __init__(self, gridSize = 25):
        super(FaceGridModel, self).__init__()
        
        self.fc = nn.Sequential(
            nn.Linear(gridSize * gridSize, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 16),
            nn.ReLU(inplace=True),
            )
        '''
        self.fc = nn.Sequential(
            KAN(layers_hidden = [gridSize * gridSize,128,16]),
            nn.ReLU(inplace=True),
            )
        '''

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x



class ITrackerModel(nn.Module):

    def __init__(self):
        super(ITrackerModel, self).__init__()
        self.eyeModel = EyeImageModel()
        self.faceModel = FaceImageModel()
        self.gridModel = FaceGridModel()
        
        # Joining everything
        self.fc = nn.Sequential(
          	nn.Linear(16*4, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 2),
            )

    def forward(self, query):
        faces = query[0]
        eyesLeft = query[1]
        eyesRight = query[2]
        faceGrids = query[3]
        
        # Eye nets
        xEyeL = self.eyeModel(eyesLeft)
        xEyeR = self.eyeModel(eyesRight)

        # Face net
        xFace = self.faceModel(faces)
        xGrid = self.gridModel(faceGrids)

        # Cat all
        x = torch.cat((xEyeL, xEyeR, xFace, xGrid), 1)
        x = self.fc(x)
        
        return x