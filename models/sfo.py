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

from models.itracker import EyeImageModel, FaceGridModel
from models.itracker import FaceImageModel

'''
sfo model on the basis of SAGE.
refer to: On-device Few-shot Personalization for Real-time Gaze Estimation, google
'''

class SFOModel(nn.Module):

    def __init__(self, num_calib = 5):
        super(SFOModel, self).__init__()
        self.num_calib = num_calib
        self.eyeModel = EyeImageModel()
        self.faceModel = FaceImageModel()
        self.gridModel = FaceGridModel()
        
        # Joining eye and grid
        self.featureFC = nn.Sequential(
            #nn.Linear(128+64+128, 128),
          	nn.Linear(16*4, 16),
            nn.ReLU(inplace=True),
            )
        # Joining query and calib
        self.regressor = nn.Sequential(
            #nn.Linear(128+64+128, 128),
          	nn.Linear(16+(16+2)*self.num_calib, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 2),
            )
            

    def forward(self, query, calibs):
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
        #x = torch.cat((xEyes, xFace, xGrid), 1)
        x = torch.cat((xEyeL, xEyeR, xFace, xGrid), 1)
        x_query = self.featureFC(x)
        
        # calib feature
        x_calibs = []
        for i in range(self.num_calib):
            calib = calibs[i]
            faces = calib[0]
            eyesLeft = calib[1]
            eyesRight = calib[2]
            faceGrids = calib[3]
            gaze_calib = calib[4]
            xEyeL = self.eyeModel(eyesLeft)
            xEyeR = self.eyeModel(eyesRight)

            # Face net
            xFace = self.faceModel(faces)
            xGrid = self.gridModel(faceGrids)

            # Cat all
            #x = torch.cat((xEyes, xFace, xGrid), 1)
            x = torch.cat((xEyeL, xEyeR, xFace, xGrid), 1)
            x_calib = self.featureFC(x)
            x_calib = torch.cat((x_calib, gaze_calib), 1)
            x_calibs.append(x_calib)
        x = torch.cat((x_query, torch.cat(x_calibs, 1)), 1)
        gaze = self.regressor(x)
        
        return gaze
