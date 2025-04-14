import math, shutil, os, time, argparse
import numpy as np
import scipy.io as sio

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from data import ITrackerData, DeltaGazeData
from models.itracker import ITrackerModel
from models.delta import DeltaGazeModel, KanDeltaModel
from models.gaze_transformer import GazeTR, BaseTR
from models.sfo import SFOModel
from models.full_face import FullFace, FullFaceDelta
from models.aff_net import AFFModel, AFFDelta
from models.sask import SASKModel
from losses import GazeLoss, GazeOriLoss

from collections import OrderedDict

import pandas as pd

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

'''
on the basis of iTracker.
'''

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(description='iTracker-pytorch-Trainer.')
parser.add_argument('--data_path', default='./gc_/', help="Path to processed dataset. It should contain metadata.mat. Use prepareDataset.py.")
parser.add_argument('--sink', type=str2bool, nargs='?', const=True, default=False, help="Just sink and terminate.")
parser.add_argument('--reset', type=str2bool, nargs='?', const=True, default=True, help="Start from scratch (do not load).")
parser.add_argument('--model_type', type=str, nargs='?', const=True, default='delta', help="support model type [base, delta, sfo, tr, ff, kan, aff].")
parser.add_argument('--num_calib', type=int, nargs='?', const=True, default=1, help="number of calibration samples.")
parser.add_argument('--is_best', type=str2bool, nargs='?', const=True, default=True, help="load best or not.")
parser.add_argument('--is_delta', type=str2bool, nargs='?', const=True, default=True, help="delta model or gaze model.")
args = parser.parse_args()

# Change there flags to control what happens.
doLoad = not args.reset # Load checkpoint at the beginning
doTest = args.sink # Only run test, no training
isBaseModel = not args.is_delta

workers = 16
epochs = 25
batch_size = torch.cuda.device_count()*100 # Change if out of cuda memory, sfo 40, tr 50

base_lr = 0.0001 #tr 0.0001
momentum = 0.9
weight_decay = 1e-4
print_freq = 10
prec1 = 0
best_prec1 = 1e20
lr = base_lr

count_test = 0
count = 0

def main():
    global args, best_prec1, weight_decay, momentum

    if args.model_type == 'base':
        # model = ITrackerModel()
        # model = BaseTR()
        # model = AFFModel()
        # model = FullFace()
        model = SASKModel()
    elif args.model_type == 'delta':
        model = DeltaGazeModel()
    elif args.model_type == 'sfo':
        model = SFOModel(num_calib = args.num_calib)
    elif args.model_type == 'tr':
        model = GazeTR()
    elif args.model_type == 'ff':
        model = FullFaceDelta()
    elif args.model_type == 'kan':
        model = KanDeltaModel()
    elif args.model_type == 'aff':
        model = AFFDelta()
    else:
        print('no support model type!')
        return
    model = torch.nn.DataParallel(model)
    model.cuda()
    imSize=(64,64) # eye size, 224 for tr, 112 for aff, 64 for others
    cudnn.benchmark = True   

    epoch = 0
    if doLoad:
        saved = load_checkpoint(is_best = args.is_best, model_type = args.model_type)
        if saved:
            print('Loading checkpoint for epoch %05d with loss %.5f (which is the mean squared error not the actual linear error)...' % (saved['epoch'], saved['best_prec1']))
            state = saved['state_dict']
            '''
            new_state_dict = OrderedDict()
            for k, v in state.items():
                name = k[7:]
                new_state_dict[name] = v
            '''
            try:
                model.module.load_state_dict(state)
            except:
                model.load_state_dict(state)
            epoch = saved['epoch']
            # best_prec1 = saved['best_prec1']
        else:
            print('Warning: Could not read checkpoint!')

    if args.model_type == 'base':
        #dataTrain = ITrackerData(dataPath = args.data_path, split='train', imSize = imSize)
        #dataVal = ITrackerData(dataPath = args.data_path, split='test', imSize = imSize)
        dataTrain = DeltaGazeData(dataPath = args.data_path, split='train', imSize = imSize, numCalib = args.num_calib)
        dataVal = DeltaGazeData(dataPath = args.data_path, split='test', imSize = imSize, numCalib = args.num_calib)
    elif args.model_type in ['delta', 'sfo', 'tr', 'ff', 'kan', 'aff']:
        dataTrain = DeltaGazeData(dataPath = args.data_path, split='train', imSize = imSize, numCalib = args.num_calib)
        dataVal = DeltaGazeData(dataPath = args.data_path, split='test', imSize = imSize, numCalib = args.num_calib)
    else:
        print('no support model type!')
        return
   
    train_loader = torch.utils.data.DataLoader(
        dataTrain,
        batch_size=batch_size, shuffle=True,
        num_workers=workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        dataVal,
        batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=True)


    # criterion = nn.MSELoss().cuda()
    criterion = nn.L1Loss().cuda()
    # criterion = GazeLoss()
    # criterion = GazeOriLoss()

    #optimizer = torch.optim.SGD(model.parameters(), lr,momentum=momentum,weight_decay=weight_decay)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=weight_decay)

    # Quick test
    if doTest and isBaseModel:
        validate(val_loader, model, criterion, epoch)
        return
    elif doTest and args.model_type in ['delta', 'tr', 'ff', 'kan', 'aff']:
        #validateDelta(val_loader, model, criterion, epoch)
        testDelta(val_loader, model, epoch, numCalib=args.num_calib)
        return
    elif doTest and args.model_type == 'sfo':
        validateSFO(val_loader, model, criterion, epoch)
        return

    # training
    for epoch in range(0, epoch):
        adjust_learning_rate(optimizer, epoch)
        
    for epoch in range(epoch, epochs):
        adjust_learning_rate(optimizer, epoch)

        # train/evaluate for one epoch
        if isBaseModel:
            train(train_loader, model, criterion, optimizer, epoch)
            prec1 = validate(val_loader, model, criterion, epoch)
        elif args.model_type in ['delta', 'tr', 'ff', 'kan', 'aff']:
            trainDelta(train_loader, model, criterion, optimizer, epoch)
            prec1 = validateDelta(val_loader, model, criterion, epoch)
        elif args.model_type == 'sfo':
            trainSFO(train_loader, model, criterion, optimizer, epoch)
            prec1 = validateSFO(val_loader, model, criterion, epoch)

        # remember best prec@1 and save checkpoint
        is_best = prec1 < best_prec1
        best_prec1 = min(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
        }, is_best, model_type = args.model_type)


def train(train_loader, model, criterion,optimizer, epoch):
    global count
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()

    for i, data_dict in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        
        query = data_dict['query']
        gaze = query[-1]
        gaze = gaze.cuda()
        gaze = torch.autograd.Variable(gaze, requires_grad = False)

        # compute output
        #output = model(imFace, imEyeL, imEyeR, faceGrid)
        output = model(query)

        loss = criterion(output, gaze)
        
        losses.update(loss.data.item(), query[0].size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        count=count+1

        print('Epoch (train): [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses))
        
def validate(val_loader, model, criterion, epoch):
    global count_test
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    lossesLin = AverageMeter()

    # switch to evaluate mode
    model.eval()
    end = time.time()


    oIndex = 0
    for i, data_dict in enumerate(val_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        
        query = data_dict['query']
        gaze = query[-1]
        gaze = gaze.cuda()
        gaze = torch.autograd.Variable(gaze, requires_grad = False)

        # compute output
        with torch.no_grad():
            #output = model(imFace, imEyeL, imEyeR, faceGrid)
            output = model(query)

        loss = criterion(output, gaze)
        
        lossLin = output - gaze
        lossLin = torch.mul(lossLin,lossLin)
        lossLin = torch.sum(lossLin,1)
        error = torch.sqrt(lossLin)
        lossLin = torch.mean(torch.sqrt(lossLin))

        losses.update(loss.data.item(), query[0].size(0))
        lossesLin.update(lossLin.item(), query[0].size(0))
     
        # compute gradient and do SGD step
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # write output to csv
        '''
        recNum = data_dict['recNum']
        data = {'subject_id': recNum.cpu().numpy().tolist(), 'error': error.cpu().numpy().tolist(), 'pre_x': output[:,0].cpu().numpy().tolist(), 'pre_y': output[:,1].cpu().numpy().tolist(), 'gt_x': gaze[:,0].cpu().numpy().tolist(), 'gt_y': gaze[:,1].cpu().numpy().tolist()}
        df = pd.DataFrame(data)
        output_file = './outputs/output_base.csv'
        if os.path.exists(output_file):
            df.to_csv(output_file, mode='a', header=False, index=False)
        else:
            df.to_csv(output_file, mode='w', header=True, index=False)
        '''

        print('Epoch (val): [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Error L2 {lossLin.val:.4f} ({lossLin.avg:.4f})\t'.format(
                    epoch, i, len(val_loader), batch_time=batch_time,
                   loss=losses,lossLin=lossesLin))

    return lossesLin.avg

def trainSFO(train_loader, model, criterion,optimizer, epoch):
    global count
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()

    for i, data_dict in enumerate(train_loader):
        
        # measure data loading time
        data_time.update(time.time() - end)
        
        query = data_dict['query']
        calibs = data_dict['calibs']
        deltas = data_dict['deltas']
        gaze = query[-1]
        gaze = gaze.cuda()
        gaze = torch.autograd.Variable(gaze, requires_grad = False)

        # compute output
        output = model(query, calibs)

        loss = criterion(output, gaze)
        
        losses.update(loss.data.item(), query[0].size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        count=count+1

        print('Epoch (train): [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses))

def validateSFO(val_loader, model, criterion, epoch):
    global count_test
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    lossesLin = AverageMeter()

    # switch to evaluate mode
    model.eval()
    end = time.time()


    oIndex = 0
    for i, data_dict in enumerate(val_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        
        query = data_dict['query']
        calibs = data_dict['calibs']
        deltas = data_dict['deltas']
        oris = data_dict['oris']
        
        gaze = query[-1]
        gaze = gaze.cuda()
        gaze = torch.autograd.Variable(gaze, requires_grad = False)

        # compute output
        with torch.no_grad():
            output = model(query, calibs)

        loss = criterion(output, gaze)
        
        lossLin = output - gaze
        lossLin = torch.mul(lossLin,lossLin)
        lossLin = torch.sum(lossLin,1)
        lossLin = torch.mean(torch.sqrt(lossLin))

        losses.update(loss.data.item(), query[0].size(0))
        lossesLin.update(lossLin.item(), query[0].size(0))
     
        # compute gradient and do SGD step
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()


        print('Epoch (val): [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Error L2 {lossLin.val:.4f} ({lossLin.avg:.4f})\t'.format(
                    epoch, i, len(val_loader), batch_time=batch_time,
                   loss=losses,lossLin=lossesLin))

    return lossesLin.avg

def trainDelta(train_loader, model, criterion,optimizer, epoch):
    global count
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()

    for i, data_dict in enumerate(train_loader):
        
        # measure data loading time
        data_time.update(time.time() - end)
        
        query = data_dict['query']
        gaze_gt = query[-1].cuda()
        gaze_gt = torch.autograd.Variable(gaze_gt, requires_grad = False)

        calibs = data_dict['calibs']
        calib = calibs[0]

        deltas = data_dict['deltas']
        oris = data_dict['oris']
        delta = deltas[0]
        ori = oris[0]
        delta = delta.cuda()
        delta = torch.autograd.Variable(delta, requires_grad = False)
        ori = ori.cuda()
        ori = torch.autograd.Variable(ori, requires_grad = False)

        similarities = data_dict['similarities']
        similarity = similarities[0]
        similarity = similarity.cuda()
        similarity = torch.autograd.Variable(similarity, requires_grad = False)

        # compute output
        output = model(query, calib)

        loss_dict = criterion(output['gaze'], gaze_gt)
        # loss_dict = criterion(output['delta'], delta)
        loss = loss_dict

        # loss_dict = criterion(output['delta'], delta, output['ori'], ori)
        # loss = loss_dict['overall']
        #loss = loss_dict['reg_loss']

        # loss_dict = criterion(output['delta'], delta, output['x_query'], output['x_calib'], similarity)
        # loss = loss_dict['overall']
        # loss = loss_dict['reg_loss']
        
        losses.update(loss.data.item(), query[0].size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        count=count+1

        print('Epoch (train): [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses))
        # print('reg loss: %.4f, cls loss: %.4f'%(loss_dict['reg_loss'], loss_dict['cls_loss']))
        # print('reg loss: %.4f, cont loss: %.4f'%(loss_dict['reg_loss'], loss_dict['contrast_loss']))

def validateDelta(val_loader, model, criterion, epoch):
    global count_test
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    lossesLin = AverageMeter()

    # switch to evaluate mode
    model.eval()
    end = time.time()


    oIndex = 0
    for i, data_dict in enumerate(val_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        
        query = data_dict['query']
        gaze_gt = query[-1].cuda()
        gaze_gt = torch.autograd.Variable(gaze_gt, requires_grad = False)

        calibs = data_dict['calibs']
        deltas = data_dict['deltas']
        oris = data_dict['oris']
        calib = calibs[0]
        delta = deltas[0]
        ori = oris[0]
        delta = delta.cuda()
        delta = torch.autograd.Variable(delta, requires_grad = False)
        ori = ori.cuda()
        ori = torch.autograd.Variable(ori, requires_grad = False)

        similarities = data_dict['similarities']
        similarity = similarities[0]
        similarity = similarity.cuda()
        similarity = torch.autograd.Variable(similarity, requires_grad = False)

        # compute output
        with torch.no_grad():
            output  = model(query, calib)

        loss_dict = criterion(output['gaze'], gaze_gt)
        # loss_dict = criterion(output['delta'], delta)
        loss = loss_dict

        # loss_dict = criterion(output['delta'], delta, output['ori'], ori)
        # loss = loss_dict['overall']

        # loss_dict = criterion(output['delta'], delta, output['x_query'], output['x_calib'], similarity)
        # loss = loss_dict['overall']
        
        lossLin = output['gaze'] - gaze_gt
        # lossLin = output['delta'] - delta
        lossLin = torch.mul(lossLin,lossLin)
        lossLin = torch.sum(lossLin,1)
        lossLin = torch.mean(torch.sqrt(lossLin))

        losses.update(loss.data.item(), query[0].size(0))
        lossesLin.update(lossLin.item(), query[0].size(0))
     
        # compute gradient and do SGD step
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()


        print('Epoch (val): [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Error L2 {lossLin.val:.4f} ({lossLin.avg:.4f})\t'.format(
                    epoch, i, len(val_loader), batch_time=batch_time,
                   loss=losses,lossLin=lossesLin))
        # print('reg loss: %.4f, cls loss: %.4f'%(loss_dict['reg_loss'], loss_dict['cls_loss']))
        # print('reg loss: %.4f, cont loss: %.4f'%(loss_dict['reg_loss'], loss_dict['contrast_loss']))

    return lossesLin.avg

def testDelta(val_loader, model, epoch, numCalib):
    global count_test
    lossesLin = AverageMeter()
    batch_time = AverageMeter()

    # switch to evaluate mode
    model.eval()
    end = time.time()

    oIndex = 0
    for i, data_dict in enumerate(val_loader):
        query = data_dict['query']
        calibs = data_dict['calibs']
        deltas = data_dict['deltas']
        gaze_gt = query[-1].cuda()
        gaze_pre = torch.zeros(gaze_gt.size(0),2).cuda()
        for j in range(numCalib):
            calib = calibs[j]
            delta = deltas[j]

            delta = delta.cuda()
            delta = torch.autograd.Variable(delta, requires_grad = False)

            # compute output
            with torch.no_grad():
                output = model(query, calib)
                gaze_pre += (output['delta']+calib[-1].cuda())
        gaze_pre = gaze_pre/numCalib
        
        lossLin = gaze_gt - gaze_pre
        lossLin = torch.mul(lossLin,lossLin)
        lossLin = torch.sum(lossLin,1)
        error = torch.sqrt(lossLin)
        lossLin = torch.mean(torch.sqrt(lossLin))

        lossesLin.update(lossLin.item(), query[0].size(0))

        # compute gradient and do SGD step
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # write output to csv
        '''
        recNum = data_dict['recNum']
        data = {'subject_id': recNum.cpu().numpy().tolist(), 'error': error.cpu().numpy().tolist(), 'pre_x': gaze_pre[:,0].cpu().numpy().tolist(), 'pre_y': gaze_pre[:,1].cpu().numpy().tolist(), 'gt_x': gaze_gt[:,0].cpu().numpy().tolist(), 'gt_y': gaze_gt[:,1].cpu().numpy().tolist()}
        df = pd.DataFrame(data)
        output_file = './outputs/output_baseTR.csv'
        if os.path.exists(output_file):
            df.to_csv(output_file, mode='a', header=False, index=False)
        else:
            df.to_csv(output_file, mode='w', header=True, index=False)
        '''
        print('Epoch (val): [{0}][{1}/{2}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Error L2 {lossLin.val:.4f} ({lossLin.avg:.4f})\t'.format(
                    epoch, i, len(val_loader), batch_time=batch_time,
                    lossLin=lossesLin))

    return lossesLin.avg

CHECKPOINTS_PATH = './checkpoints/'

def load_checkpoint(is_best, filename='ckpt.pth.tar', model_type='delta'):
    filename = model_type +'_SK_'+ filename
    # filename = model_type +'_BAM_'+ filename
    # filename = model_type + 'TR_iPad_' + filename
    # filename = model_type +'TR_'+ filename
    # filename = model_type + '_iPad_' + filename
    # filename = model_type + '_sim_' + filename
    # filename = model_type + '_ori_' + filename
    bestFilename = os.path.join(CHECKPOINTS_PATH, 'best_' + filename)
    filename = os.path.join(CHECKPOINTS_PATH, filename)
    if not os.path.isfile(filename):
        return None
    if is_best:
        state = torch.load(bestFilename)
        print(bestFilename)
    else:
        state = torch.load(filename)
        print(filename)
    return state

def save_checkpoint(state, is_best, filename='ckpt.pth.tar', model_type='delta'):
    if not os.path.isdir(CHECKPOINTS_PATH):
        os.makedirs(CHECKPOINTS_PATH, 0o777)
    filename = model_type +'_SK_'+ filename
    # filename = model_type +'_BAM_'+ filename
    # filename = model_type + 'TR_iPad_' + filename
    # filename = model_type +'TR_'+ filename
    # filename = model_type + '_iPad_' + filename
    # filename = model_type + '_sim_' + filename
    # filename = model_type + '_ori_' + filename
    bestFilename = os.path.join(CHECKPOINTS_PATH, 'best_' + filename)
    filename = os.path.join(CHECKPOINTS_PATH, filename)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, bestFilename)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = base_lr * (0.1 ** (epoch // 10))
    for param_group in optimizer.state_dict()['param_groups']:
        param_group['lr'] = lr


if __name__ == "__main__":
    main()
    print('DONE')
