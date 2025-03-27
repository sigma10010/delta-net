import argparse
import os
import shutil
import time, math
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
import torch.utils.model_zoo as model_zoo
from torch.autograd.variable import Variable

from models.itracker import EyeImageModel, FaceGridModel
from models.itracker import FaceImageModel

'''
delta gaze model based on CNN

'''

class DeltaGazeModel(nn.Module):

    def __init__(self):
        super(DeltaGazeModel, self).__init__()
        self.eyeModel = EyeImageModel()
        self.faceModel = FaceImageModel()
        self.gridModel = FaceGridModel()
        
        # Joining everything
        self.featureFC = nn.Sequential(
            #nn.Linear(128+64+128, 128),
          	nn.Linear(16*4, 16),
            nn.ReLU(inplace=True),
            )
        # Joining query and calib
        self.regressor = nn.Sequential(
            nn.Linear(16+16, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 2),
            )
        self.classifier = nn.Sequential(
            nn.Linear(16+16, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 4),
            )

    def forward(self, query, calib):
        # query feature
        faces = query[0]
        eyesLeft = query[1]
        eyesRight = query[2]
        faceGrids = query[3]
        xEyeL = self.eyeModel(eyesLeft)
        xEyeR = self.eyeModel(eyesRight)
        # Face net
        xFace = self.faceModel(faces)
        xGrid = self.gridModel(faceGrids)

        # Cat all
        x = torch.cat((xEyeL, xEyeR, xFace, xGrid), 1)
        x_query = self.featureFC(x)
        
        # calib feature
        faces = calib[0]
        eyesLeft = calib[1]
        eyesRight = calib[2]
        faceGrids = calib[3]
        xEyeL = self.eyeModel(eyesLeft)
        xEyeR = self.eyeModel(eyesRight)
        # Face net
        xFace = self.faceModel(faces)
        xGrid = self.gridModel(faceGrids)

        # Cat all
        x = torch.cat((xEyeL, xEyeR, xFace, xGrid), 1)
        x_calib = self.featureFC(x)
        
        x = torch.cat((x_query, x_calib), 1)
        delta = self.regressor(x)
        ori = self.classifier(x)
        
        return {'delta':delta, 'ori':ori, 'x_query':x_query, 'x_calib':x_calib}
