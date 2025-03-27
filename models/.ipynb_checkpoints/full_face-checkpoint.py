import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import math
import numpy as np

class FullFace(nn.Module):
    '''It’s written all over your face: Full-face appearance-based gaze estimation
    '''
    
    def __init__(self):
        super(FullFace, self).__init__()
        alexnet = torchvision.models.alexnet(pretrained=False)

        self.convNet = alexnet.features

        self.weightStream = nn.Sequential(
            nn.Conv2d(256, 256, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1, 1),
        )

        self.FC = nn.Sequential(
            nn.Linear(256*6*6, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 2)
        )

    def forward(self, query):
        faces = query[0]
        faceFeature = self.convNet(faces)
        weight = self.weightStream(faceFeature)
        
        faceFeature = weight * faceFeature
        #print('faceFeature.shape', faceFeature.shape)
        faceFeature = torch.flatten(faceFeature, start_dim=1)
        gaze = self.FC(faceFeature)

        return gaze
    
class FullFaceDelta(nn.Module):
    
    def __init__(self):
        super(FullFaceDelta, self).__init__()
        
        alexnet = torchvision.models.alexnet(pretrained=False)

        self.convNet = alexnet.features

        self.weightStream = nn.Sequential(
            nn.Conv2d(256, 256, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1, 1),
        )

        self.FC = nn.Sequential(
            nn.Linear(2*256*6*6, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 2)
        )

    def forward(self, query, calib):
        faces = query[0]
        faceFeature = self.convNet(faces)
        weight = self.weightStream(faceFeature)
        faceFeature = weight * faceFeature
        faceFeatureQ = torch.flatten(faceFeature, start_dim=1)
        
        faces = calib[0]
        faceFeature = self.convNet(faces)
        weight = self.weightStream(faceFeature)
        faceFeature = weight * faceFeature
        faceFeatureC = torch.flatten(faceFeature, start_dim=1)
        
        x = torch.cat((faceFeatureQ, faceFeatureC), 1)
        
        
        delta = self.FC(x)

        return {'delta': delta}