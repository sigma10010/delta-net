import torch.utils.data as data
import scipy.io as sio
from PIL import Image
import os
import os.path
import torchvision.transforms as transforms
import torch
import numpy as np
import re

import random

'''
Data loader for the iTracker.
Use prepareDataset.py to convert the dataset from http://gazecapture.csail.mit.edu/ to proper format.

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

MEAN_PATH = './metadata/'
META_PATH = './metadata/'

def loadMetadata(filename, silent = False):
    try:
        # http://stackoverflow.com/questions/6273634/access-array-contents-from-a-mat-file-loaded-using-scipy-io-loadmat-python
        if not silent:
            print('\tReading metadata from %s...' % filename)
        metadata = sio.loadmat(filename, squeeze_me=True, struct_as_record=False)
    except:
        print('\tFailed to read the meta file "%s"!' % filename)
        return None
    return metadata

class SubtractMean(object):
    """Normalize an tensor image with mean.
    """

    def __init__(self, meanImg, imSize=(224,224)):
        if imSize[0]==224:
            self.meanImg = transforms.ToTensor()(meanImg / 255)
        else:
            self.meanImg = transforms.Resize(imSize)(transforms.ToTensor()(meanImg / 255))
        #print(self.meanImg.shape,'!!!!!!!!!!')

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """       
        return tensor.sub(self.meanImg)


class ITrackerData(data.Dataset):
    def __init__(self, dataPath, split = 'train', imSize=(224,224), gridSize=(25, 25)):

        self.dataPath = dataPath
        self.imSize = imSize
        self.gridSize = gridSize

        print('Loading iTracker dataset...')
        metaFile = os.path.join(dataPath, 'metadata.mat')
        #metaFile = 'metadata.mat'
        if metaFile is None or not os.path.isfile(metaFile):
            raise RuntimeError('There is no such file %s! Provide a valid dataset path.' % metaFile)
        self.metadata = loadMetadata(metaFile)
        if self.metadata is None:
            raise RuntimeError('Could not read metadata file %s! Provide a valid dataset path.' % metaFile)

        self.faceMean = loadMetadata(os.path.join(MEAN_PATH, 'mean_face_224.mat'))['image_mean']
        self.eyeLeftMean = loadMetadata(os.path.join(MEAN_PATH, 'mean_left_224.mat'))['image_mean']
        self.eyeRightMean = loadMetadata(os.path.join(MEAN_PATH, 'mean_right_224.mat'))['image_mean']
        
        self.transformFace = transforms.Compose([
            transforms.Resize(self.imSize),
            transforms.ToTensor(),
            SubtractMean(meanImg=self.faceMean),         
        ])
        self.transformEyeL = transforms.Compose([
            transforms.Resize(self.imSize),
            transforms.ToTensor(),
            SubtractMean(meanImg=self.eyeLeftMean),          
        ])
        self.transformEyeR = transforms.Compose([
            transforms.Resize(self.imSize),
            transforms.ToTensor(),
            SubtractMean(meanImg=self.eyeRightMean),
            #transforms.RandomHorizontalFlip(p=1.0)
        ])


        if split == 'test':
            mask = self.metadata['labelTest']
        elif split == 'val':
            mask = self.metadata['labelVal']
        else:
            mask = self.metadata['labelTrain']

        self.indices = np.argwhere(mask)[:,0]
        print('Loaded iTracker dataset split "%s" with %d records...' % (split, len(self.indices)))

    def loadImage(self, path):
        try:
            im = Image.open(path).convert('RGB')
        except OSError:
            raise RuntimeError('Could not read image: ' + path)
            #im = Image.new("RGB", self.imSize, "white")

        return im


    def makeGrid(self, params):
        gridLen = self.gridSize[0] * self.gridSize[1]
        grid = np.zeros([gridLen,], np.float32)
        
        indsY = np.array([i // self.gridSize[0] for i in range(gridLen)])
        indsX = np.array([i % self.gridSize[0] for i in range(gridLen)])
        condX = np.logical_and(indsX >= params[0], indsX < params[0] + params[2]) 
        condY = np.logical_and(indsY >= params[1], indsY < params[1] + params[3]) 
        cond = np.logical_and(condX, condY)

        grid[cond] = 1
        return grid

    def __getitem__(self, index):
        index = self.indices[index]

        imFacePath = os.path.join(self.dataPath, '%05d/appleFace/%05d.jpg' % (self.metadata['labelRecNum'][index], self.metadata['frameIndex'][index]))
        imEyeLPath = os.path.join(self.dataPath, '%05d/appleLeftEye/%05d.jpg' % (self.metadata['labelRecNum'][index], self.metadata['frameIndex'][index]))
        imEyeRPath = os.path.join(self.dataPath, '%05d/appleRightEye/%05d.jpg' % (self.metadata['labelRecNum'][index], self.metadata['frameIndex'][index]))

        imFace = self.loadImage(imFacePath)
        imEyeL = self.loadImage(imEyeLPath)
        imEyeR = self.loadImage(imEyeRPath)

        imFace = self.transformFace(imFace)
        imEyeL = self.transformEyeL(imEyeL)
        imEyeR = self.transformEyeR(imEyeR)

        gaze = np.array([self.metadata['labelDotXCam'][index], self.metadata['labelDotYCam'][index]], np.float32)

        faceGrid = self.makeGrid(self.metadata['labelFaceGrid'][index,:])

        # to tensor
        row = torch.LongTensor([int(index)])
        faceGrid = torch.FloatTensor(faceGrid)
        gaze = torch.FloatTensor(gaze)

        return row, imFace, imEyeL, imEyeR, faceGrid, gaze
    
        
    def __len__(self):
        return len(self.indices)
      
class DeltaGazeData(data.Dataset):
    def __init__(self, dataPath, split='train', imSize=(128,128), faceSize = (224,224), gridSize=(25, 25), numCalib=1):

        self.dataPath = dataPath
        self.imSize = imSize
        self.faceSize = faceSize
        self.gridSize = gridSize
        self.numCalib = numCalib
        self.dataAug = True if split=='train' else False

        print('Loading delta dataset...')
        # metaFile = os.path.join(META_PATH, 'metadata_device.mat')
        metaFile = os.path.join(META_PATH, 'metadata9.mat') # for sfo
        #metaFile = 'metadata.mat'
        if metaFile is None or not os.path.isfile(metaFile):
            raise RuntimeError('There is no such file %s! Provide a valid dataset path.' % metaFile)
        self.metadata = loadMetadata(metaFile)
        if self.metadata is None:
            raise RuntimeError('Could not read metadata file %s! Provide a valid dataset path.' % metaFile)

        self.faceMean = loadMetadata(os.path.join(MEAN_PATH, 'mean_face_224.mat'))['image_mean']
        self.eyeLeftMean = loadMetadata(os.path.join(MEAN_PATH, 'mean_left_224.mat'))['image_mean']
        self.eyeRightMean = loadMetadata(os.path.join(MEAN_PATH, 'mean_right_224.mat'))['image_mean']
        
        self.transformFace = transforms.Compose([
            transforms.Resize(self.faceSize),
            transforms.ToTensor(),
            SubtractMean(meanImg=self.faceMean),
        ])
        self.transformEyeL = transforms.Compose([
            transforms.Resize(self.imSize),
            transforms.ToTensor(),
            SubtractMean(meanImg=self.eyeLeftMean, imSize = self.imSize),
        ])
        self.transformEyeR = transforms.Compose([
            transforms.Resize(self.imSize),
            transforms.ToTensor(),
            SubtractMean(meanImg=self.eyeRightMean, imSize = self.imSize),
            transforms.RandomHorizontalFlip(p=1.0)
        ])

        if split == 'test':
            mask = self.metadata['labelTest']
            
            mask_iphone = [self.metadata['device'][i].startswith("iPad") for i in range(len(self.metadata['device']))] # iPhone/iPad
            mask_iphone = np.array(mask_iphone)
            mask = mask * mask_iphone # test on iphone only
            
        elif split == 'val':
            mask = self.metadata['labelVal']
        else:
            mask = self.metadata['labelTrain']
            
            # mask_val = self.metadata['labelVal']
            # mask += mask_val

            mask_iphone = [self.metadata['device'][i].startswith("iPad") for i in range(len(self.metadata['device']))]
            mask_iphone = np.array(mask_iphone)
            mask = mask * mask_iphone # train on iphone only
            
        self.indices = np.argwhere(mask)[:,0]
        print('Loaded delta dataset split "%s" with %d records...' % (split, len(self.indices)))

    def loadImage(self, path, aug):
        try:
            im = Image.open(path).convert('RGB')
        except OSError:
            raise RuntimeError('Could not read image: ' + path)
            #im = Image.new("RGB", self.imSize, "white")
        if aug:
            #bounding boxes aug
            h,w =im.size
            random_number = random.randint(2, 10)
            crop_size = int(h*(1-random_number/100.0))
            centerCrop = transforms.CenterCrop(crop_size)
            im = centerCrop(im)
            
            colorJitter = transforms.ColorJitter(0.2,0.2,0.1,0.0)
            im = colorJitter(im)
        return im


    def makeGrid(self, params):
        gridLen = self.gridSize[0] * self.gridSize[1]
        grid = np.zeros([gridLen,], np.float32)
        
        indsY = np.array([i // self.gridSize[0] for i in range(gridLen)])
        indsX = np.array([i % self.gridSize[0] for i in range(gridLen)])
        condX = np.logical_and(indsX >= params[0], indsX < params[0] + params[2]) 
        condY = np.logical_and(indsY >= params[1], indsY < params[1] + params[3]) 
        cond = np.logical_and(condX, condY)

        grid[cond] = 1
        return grid
    
    def get_similarity(self, gaze_query, gaze_calib, threshold = 1.0):
        dist = gaze_query - gaze_calib
        dist = torch.mul(dist,dist)
        dist = torch.sum(dist)
        dist = torch.mean(torch.sqrt(dist))
        similarity = 1 if dist<threshold else 0
        return similarity
    
    def get_orientation(self, gaze_query, gaze_calib):
        delta_x = gaze_query[0]-gaze_calib[0]
        delta_y = gaze_query[1]-gaze_calib[1]
        if delta_x>0 and delta_y>=0:
            ori = 0
        elif delta_x<=0 and delta_y>0:
            ori = 1
        elif delta_x<0 and delta_y<=0:
            ori = 2
        elif delta_x>=0 and delta_y<0:
            ori = 3
        else:
            ori = 0 #(0,0)
        return ori

    def __getitem__(self, index):
        index = self.indices[index]
        recNum = self.metadata['labelRecNum'][index]

        imFacePath = os.path.join(self.dataPath, '%05d/appleFace/%05d.jpg' % (recNum, self.metadata['frameIndex'][index]))
        imEyeLPath = os.path.join(self.dataPath, '%05d/appleLeftEye/%05d.jpg' % (recNum, self.metadata['frameIndex'][index]))
        imEyeRPath = os.path.join(self.dataPath, '%05d/appleRightEye/%05d.jpg' % (recNum, self.metadata['frameIndex'][index]))

        imFace = self.loadImage(imFacePath, aug = self.dataAug)
        imEyeL = self.loadImage(imEyeLPath, aug = self.dataAug)
        imEyeR = self.loadImage(imEyeRPath, aug = self.dataAug)

        imFace = self.transformFace(imFace)
        imEyeL = self.transformEyeL(imEyeL)
        imEyeR = self.transformEyeR(imEyeR)

        gaze = np.array([self.metadata['labelDotXCam'][index], self.metadata['labelDotYCam'][index]], np.float32)

        faceGrid = self.makeGrid(self.metadata['labelFaceGrid'][index,:])

        # to tensor
        #row = torch.LongTensor([int(index)])
        faceGrid = torch.FloatTensor(faceGrid)
        gaze = torch.FloatTensor(gaze)
        
        query = (imFace, imEyeL, imEyeR, faceGrid, gaze)
        
        # sample reference/calibration frame
        
        calibMask = self.metadata['labelRecNum']==recNum
        calibIndices = np.argwhere(calibMask)[:,0]
        calibIndices = np.delete(calibIndices, np.where(calibIndices == index))
        try:
            calibSelectedIndices = np.random.choice(calibIndices, size = self.numCalib, replace = False)
        except:
            print('number for calib of recNum %d is less than %d'%(recNum, self.numCalib))
            return
        calibs = []
        deltas = []
        oris = []
        similarities = []
        for calibIndex in calibSelectedIndices:
            imFacePath = os.path.join(self.dataPath, '%05d/appleFace/%05d.jpg' % (recNum, self.metadata['frameIndex'][calibIndex]))
            imEyeLPath = os.path.join(self.dataPath, '%05d/appleLeftEye/%05d.jpg' % (recNum, self.metadata['frameIndex'][calibIndex]))
            imEyeRPath = os.path.join(self.dataPath, '%05d/appleRightEye/%05d.jpg' % (recNum, self.metadata['frameIndex'][calibIndex]))

            imFaceCalib = self.loadImage(imFacePath, aug = self.dataAug)
            imEyeLCalib = self.loadImage(imEyeLPath, aug = self.dataAug)
            imEyeRCalib = self.loadImage(imEyeRPath, aug = self.dataAug)

            imFaceCalib = self.transformFace(imFaceCalib)
            imEyeLCalib = self.transformEyeL(imEyeLCalib)
            imEyeRCalib = self.transformEyeR(imEyeRCalib)

            gazeCalib = np.array([self.metadata['labelDotXCam'][calibIndex], self.metadata['labelDotYCam'][calibIndex]], np.float32)

            faceGridCalib = self.makeGrid(self.metadata['labelFaceGrid'][calibIndex,:])

            # to tensor
            faceGridCalib = torch.FloatTensor(faceGridCalib)
            gazeCalib = torch.FloatTensor(gazeCalib)
            
            ori = self.get_orientation(gaze, gazeCalib)
            oris.append(ori)

            similarity = self.get_similarity(gaze, gazeCalib)
            similarities.append(similarity)

            calib = (imFaceCalib, imEyeLCalib, imEyeRCalib, faceGridCalib, gazeCalib)
            calibs.append(calib)

            delta = gaze - gazeCalib
            deltas.append(delta)

        return {'query':query, 'calibs':calibs, 'deltas':deltas, 'oris':oris, 'similarities':similarities, 'recNum': recNum}
    
        
    def __len__(self):
        return len(self.indices)
