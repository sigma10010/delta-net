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
from models.sk import SKConv, SKUnit
from models.ss import SS

'''
Scale-adaptive and selective-kernel (SASK) network.
'''

class EyeImageModel(nn.Module):
    # parameters based on SAGE and AFF-Net
    # SAGE: On-device Few-shot Personalization for Real-time Gaze Estimation
    # AFF-Net: Adaptive Feature Fusion Network for Gaze Tracking in Mobile Tablets
    def __init__(self):
        super(EyeImageModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # SELayer(32,4),
            # BAM(32,4),
            CBAM(32,reduction_ratio=4),
            SKConv(32,32),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # SELayer(32,4),
            # BAM(32,4),
            CBAM(32,reduction_ratio=4),
            SKConv(32,32),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # SELayer(32,4),
            # BAM(32,4),
            CBAM(32,reduction_ratio=4),
        )
        
        self.fc = nn.Sequential(
            nn.Linear(6*6*32, 128), 
            nn.ReLU(inplace=True),
            nn.Linear(128, 32),
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
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class NewEyeImageModel(nn.Module):
    # parameters based on SAGE and AFF-Net
    # SAGE: On-device Few-shot Personalization for Real-time Gaze Estimation
    # AFF-Net: Adaptive Feature Fusion Network for Gaze Tracking in Mobile Tablets
    def __init__(self, is_att=True, att_mode = 'cbam', network_depth=4, is_scale_adaptive=True, n_scales=2):
        super(NewEyeImageModel, self).__init__()
        self.network_depth = network_depth

        self.convBlocks = nn.ModuleList([])
        for i in range(network_depth):
            if i==0:
                self.convBlocks.append(SKUnit(3,32, is_shortcut=True))
            else:
                self.convBlocks.append(SKUnit(32,32, is_shortcut=True))
                
        self.down_sample = nn.MaxPool2d(kernel_size=2, stride=2)

        self.is_att = is_att
        self.att_layer = CBAM(32,reduction_ratio=4)

        self.is_scale_adaptive = is_scale_adaptive
        self.n_scales = n_scales # [1, network_depth-1]
        self.featureBlocks = nn.ModuleList([])
        size = [4]
        for i in range(self.n_scales):
            size.append(size[-1]*2)
            self.featureBlocks.append(nn.Linear(size[-1-1]*size[-1-1]*32, 128))

        self.ss = SS(M=n_scales, ch_in=128, r=8)
        
        self.fc = nn.Sequential(
            # nn.Linear(7*7*32, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 32),
            nn.ReLU(inplace=True),
            )

    def forward(self, x):
        features = []
        for i in range(self.network_depth):
            x = self.convBlocks[i](x)
            x = self.down_sample(x)
            if self.is_att:
                x = self.att_layer(x)
            # print(x.shape)
            if i >= self.network_depth-self.n_scales:
                # print(i)
                fea = self.featureBlocks[(self.network_depth-1-i)](x.view(x.size(0), -1))
                fea = fea.unsqueeze(dim=-1).unsqueeze(dim=-1)
                features.append(fea)
        if self.is_scale_adaptive:
            x = self.ss(features).squeeze(dim=-1).squeeze(dim=-1)
        else:
            fea_stacked = torch.stack(features, dim=0)
            x = torch.sum(fea_stacked, dim=0).squeeze(dim=-1).squeeze(dim=-1)
            
        # print(x.shape)
        
        x = self.fc(x)
        return x

class NewFaceImageModel(nn.Module):
    
    def __init__(self, is_att=True, att_mode = 'cbam', network_depth=6, is_scale_adaptive=True, n_scales=2):
        super(NewFaceImageModel, self).__init__()
        self.network_depth = network_depth

        self.convBlocks = nn.ModuleList([])
        for i in range(network_depth):
            if i==0:
                self.convBlocks.append(SKUnit(3,32, is_shortcut=True))
            else:
                self.convBlocks.append(SKUnit(32,32, is_shortcut=True))
                
        self.down_sample = nn.MaxPool2d(kernel_size=2, stride=2)

        self.is_att = is_att
        self.att_layer = CBAM(32,reduction_ratio=4)

        self.is_scale_adaptive = is_scale_adaptive
        self.n_scales = n_scales # [1, network_depth-1]
        self.featureBlocks = nn.ModuleList([])
        size = [2]
        for i in range(self.n_scales):
            size.append(size[-1]*3)
            self.featureBlocks.append(nn.Linear(size[-1-1]*size[-1-1]*32, 128))
        
        self.fc = nn.Sequential(
            # nn.Linear(6*6*32, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 32),
            nn.ReLU(inplace=True),
            )

    def forward(self, x):
        features = []
        for i in range(self.network_depth):
            x = self.convBlocks[i](x)
            x = self.down_sample(x)
            if self.is_att:
                x = self.att_layer(x)
            print(x.shape)
            if i >= self.network_depth-self.n_scales:
                print(i)
                # features.append(self.featureBlocks[(self.network_depth-1-i)](x.view(x.size(0), -1)))
        # fea_stacked = torch.stack(features, dim=0)
        # x = torch.sum(fea_stacked, dim=0)
        # print(x.shape)
        # x = self.fc(x)
        return 0

class FaceImageModel(nn.Module):
    
    def __init__(self):
        super(FaceImageModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # SELayer(32,4),
            # BAM(32,4),
            CBAM(32,reduction_ratio=4),
            SKConv(32,32),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # SELayer(32,4),
            # BAM(32,4),
            CBAM(32,reduction_ratio=4),
            SKConv(32,32),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # SELayer(32,4),
            # BAM(32,4),
            CBAM(32,reduction_ratio=4),
            SKConv(32,32),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # SELayer(32,4),
            # BAM(32,4),
            CBAM(32,reduction_ratio=4),
            SKConv(32,32),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # SELayer(32,4),
            # BAM(32,4),
            CBAM(32,reduction_ratio=4),
            SKConv(32,32),
            # SELayer(32,4),
            # BAM(32,4),
            CBAM(32,reduction_ratio=4),
        )
        
        self.fc = nn.Sequential(
            nn.Linear(5*5*32, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 32),
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
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class FaceGridModel(nn.Module):
    # Model for the face grid pathway
    def __init__(self, gridSize = 25):
        super(FaceGridModel, self).__init__()
        
        self.fc = nn.Sequential(
            nn.Linear(gridSize * gridSize, 128),
            # nn.LeakyReLU(inplace=True),
            # nn.Linear(64, 96),
            # nn.LeakyReLU(inplace=True),
            # nn.Linear(96, 128),
            nn.LeakyReLU(inplace=True),
            nn.Linear(128, 32),
            nn.LeakyReLU(inplace=True),
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

class SASKModel(nn.Module):

    def __init__(self):
        super(SASKModel, self).__init__()
        self.eyeModel = NewEyeImageModel(is_att=True, att_mode = 'cbam', network_depth=4, is_scale_adaptive=True, n_scales=2)
        self.faceModel = NewEyeImageModel(is_att=True, att_mode = 'cbam', network_depth=6, is_scale_adaptive=True, n_scales=2)
        self.gridModel = FaceGridModel()
        
        # Joining everything
        self.fc = nn.Sequential(
          	nn.Linear(32*4, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 2),
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