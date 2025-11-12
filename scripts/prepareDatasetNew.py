import math, shutil, os, time, argparse, json, re, sys
import numpy as np
import scipy.io as sio
from PIL import Image

import cv2
import torch
# from moge.model.v1 import MoGeModel
# from moge.model.v2 import MoGeModel # Let's try MoGe-2

from face_det import extract_and_save_face_data

os.environ['GLOG_minloglevel'] = '2'  # 只显示 error

'''
Prepares the GazeCapture dataset for use with the pytorch code. Crops images, compiles JSONs into metadata.mat

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

parser = argparse.ArgumentParser(description='iTracker-pytorch-PrepareDataset.')
parser.add_argument('--dataset_path', default='/home/sigma/gaze/datasets/gc/', help="Path to extracted files. It should have folders called '%%05d' in it.")
parser.add_argument('--output_path', default='/home/sigma/gaze/datasets/gc_mp/', help="Where to write the output. Can be the same as dataset_path if you wish (=default).")
args = parser.parse_args()

device = torch.device("cuda")
# model = MoGeModel.from_pretrained("/home/sigma/moge/moge-2-vitl-normal/model.pt").to(device)

def main():
    if args.output_path is None:
        args.output_path = args.dataset_path
    
    if args.dataset_path is None or not os.path.isdir(args.dataset_path):
        raise RuntimeError('No such dataset folder %s!' % args.dataset_path)

    preparePath(args.output_path)

    # list recordings
    recordings = os.listdir(args.dataset_path)
    recordings = np.array(recordings, object)
    recordings = recordings[[os.path.isdir(os.path.join(args.dataset_path, r)) for r in recordings]]
    recordings.sort()

    # Output structure
    meta = {
        'labelRecNum': [],
        'frameIndex': [],
        'labelDotXCam': [],
        'labelDotYCam': [],
        # 'labelFaceGrid': [],
        'device': [],
        'depth':[],
        # 'landmarks':[],
        'rects': []
    }

    # # --- 1. Load from .mat ---
    # meta = sio.loadmat('/home/sigma/gaze/datasets/gc_mp/metadata_850.mat', struct_as_record=False) 
    # if 'labelFaceGrid' in meta:
    #     del meta['labelFaceGrid']
    # # --- 2. Convert relevant fields for appending (only those you will update) ---
    # for key in ['labelRecNum', 'frameIndex', 'labelDotXCam', 'labelDotYCam',
    #             'labelFaceGrid', 'device', 'depth', 'rects']:
    #     if key in meta and isinstance(meta[key], np.ndarray):
    #         if key == 'labelFaceGrid' or key == 'rects':
    #             # For 2D arrays, use list of arrays for efficient appending
    #             meta[key] = [meta[key][i] for i in range(meta[key].shape[0])]
    #         else:
    #             meta[key] = meta[key].flatten().tolist()

    for i,recording in enumerate(recordings):
        if i<1421:
            continue
        print('[%d/%d] Processing recording %s (%.2f%%)' % (i, len(recordings), recording, i / len(recordings) * 100))
        # if i>1:
            # break
        recDir = os.path.join(args.dataset_path, recording)
        recDirOut = os.path.join(args.output_path, recording)

        #Read JSONs
        # appleFace = readJson(os.path.join(recDir, 'appleFace.json'))
        # if appleFace is None:
        #     continue
        # appleLeftEye = readJson(os.path.join(recDir, 'appleLeftEye.json'))
        # if appleLeftEye is None:
        #     continue
        # appleRightEye = readJson(os.path.join(recDir, 'appleRightEye.json'))
        # if appleRightEye is None:
        #     continue
        dotInfo = readJson(os.path.join(recDir, 'dotInfo.json'))
        if dotInfo is None:
            continue
        # faceGrid = readJson(os.path.join(recDir, 'faceGrid.json'))
        # if faceGrid is None:
        #     continue
        frames = readJson(os.path.join(recDir, 'frames.json'))
        if frames is None:
            continue
        info = readJson(os.path.join(recDir, 'info.json'))
        if info is None:
            continue
        # screen = readJson(os.path.join(recDir, 'screen.json'))
        # if screen is None:
        #     continue

        facePath = preparePath(os.path.join(recDirOut, 'appleFace'))
        leftEyePath = preparePath(os.path.join(recDirOut, 'appleLeftEye'))
        rightEyePath = preparePath(os.path.join(recDirOut, 'appleRightEye'))
        # normalPath = preparePath(os.path.join(recDirOut, 'normal'))
        # depthPath = preparePath(os.path.join(recDirOut, 'depth'))
        # intrinsicsPath = preparePath(os.path.join(recDirOut, 'intrinsics'))

        # Preprocess
        # allValid = np.logical_and(np.logical_and(appleFace['IsValid'], appleLeftEye['IsValid']), np.logical_and(appleRightEye['IsValid'], faceGrid['IsValid']))
        # if not np.any(allValid):
        #     continue

        frames = np.array([int(re.match('(\d{5})\.jpg$', x).group(1)) for x in frames])

        # bboxFromJson = lambda data: np.stack((data['X'], data['Y'], data['W'],data['H']), axis=1).astype(int)
        # faceBbox = bboxFromJson(appleFace) + [-1,-1,1,1] # for compatibility with matlab code
        # leftEyeBbox = bboxFromJson(appleLeftEye) + [0,-1,0,0]
        # rightEyeBbox = bboxFromJson(appleRightEye) + [0,-1,0,0]
        # leftEyeBbox[:,:2] += faceBbox[:,:2] # relative to face
        # rightEyeBbox[:,:2] += faceBbox[:,:2]
        # faceGridBbox = bboxFromJson(faceGrid)


        for j,frame in enumerate(frames):
            # print(j, len(frames))
            # Can we use it?
            # if not allValid[j]:
            #     continue

            # Load image
            imgFile = os.path.join(recDir, 'frames', '%05d.jpg' % frame)
            if not os.path.isfile(imgFile):
                logError('Warning: Could not read image file %s!' % imgFile)
                continue
            img = Image.open(imgFile)        
            if img is None:
                logError('Warning: Could not read image file %s!' % imgFile)
                continue
            img = np.array(img.convert('RGB'))
            h, w = img.shape[:2]

            # # Crop images
            # imFace = cropImage(img, faceBbox[j,:])
            # imEyeL = cropImage(img, leftEyeBbox[j,:])
            # imEyeR = cropImage(img, rightEyeBbox[j,:])

            # # Save images
            # Image.fromarray(imFace).save(os.path.join(facePath, '%05d.jpg' % frame), quality=95)
            # Image.fromarray(imEyeL).save(os.path.join(leftEyePath, '%05d.jpg' % frame), quality=95)
            # Image.fromarray(imEyeR).save(os.path.join(rightEyePath, '%05d.jpg' % frame), quality=95)

            face_data = extract_and_save_face_data(imgFile)
            if face_data is None:
                logError('Warning: Could not extract_and_save_face_data from image file %s!' % imgFile)
                continue

            face_rect = face_data['face_rect']
            x1, y1, x2, y2 = face_rect
            faceBbox = [x1, y1, x2 - x1, y2 - y1]

            # faceGrid = make_face_grid(face_rect, (w, h))  # 注意 img_size 是 (w, h)

            left_eye_rect = face_data['left_eye_rect']
            x1, y1, x2, y2 = left_eye_rect
            leftEyeBbox = [x1, y1, x2 - x1, y2 - y1]
            
            right_eye_rect = face_data['right_eye_rect']
            x1, y1, x2, y2 = right_eye_rect
            rightEyeBbox = [x1, y1, x2 - x1, y2 - y1]

            rects = (
                normalize_rect(face_rect, w, h)
                + normalize_rect(left_eye_rect, w, h)
                + normalize_rect(right_eye_rect, w, h)
            )

            keypoints = face_data['face_keypoints']
            landmarks, mean_depth  = normalize_keypoints(keypoints, w, h)

            '''
            # =====get geometry moge=====
            img_tensor = torch.tensor(img / 255.0, dtype=torch.float32, device=device).permute(2, 0, 1)  # (3, H, W)        
            output = model.infer(img_tensor)

            # Detach to numpy
            points = output["points"].detach().cpu().numpy()     # (H, W, 3)
            depth = output["depth"].detach().cpu().numpy()       # (H, W)
            mask = output["mask"].detach().cpu().numpy()         # (H, W)
            # normal = output["normal"].detach().cpu().numpy()     # (H, W, 3)
            # intrinsics = output["intrinsics"].detach().cpu().numpy() # (3, 3)

            # Crop face region
            depthFace = cropImage2D(depth, faceBbox)
            maskFace = cropImage2D(mask, faceBbox)
            # normalFace = cropImage(normal, faceBbox[j, :])

            # Replace infs for safe processing
            depthFace[np.isinf(depthFace)] = 0

            # Compute mean depth on valid pixels
            valid_mask = (maskFace > 0) & np.isfinite(depthFace)
            mean_depth = depthFace[valid_mask].mean() if valid_mask.any() else 0

            # # Normalize depth for visualization
            # depth_vis = (depthFace - depthFace.min()) / (np.ptp(depthFace) + 1e-8)
            # depth_vis = (depth_vis * 255).astype(np.uint8)
            # Image.fromarray(depth_vis).save(os.path.join(depthPath, '%05d.jpg' % frame), quality=95)

            # # Normalize normal for visualization
            # normal_vis = ((normalFace + 1.0) / 2.0 * 255).clip(0, 255).astype(np.uint8)
            # Image.fromarray(normal_vis).save(os.path.join(normalPath, '%05d.jpg' % frame), quality=95)

            # # save intrinsics as json
            # intrinsics_list = intrinsics.tolist()
            # intrinsics_file = os.path.join(intrinsicsPath, '%05d.json' % frame)
            # with open(intrinsics_file, 'w') as f:
            #     json.dump({'intrinsics': intrinsics_list}, f, indent=2)

            # get landmarks
            landmarks=[]
            # Face center
            x, y = get_center(faceBbox)
            landmarks.extend(get_point(points, x, y))
            # Left eye center
            x, y = get_center(leftEyeBbox)
            landmarks.extend(get_point(points, x, y))
            # Right eye center
            x, y = get_center(rightEyeBbox)
            landmarks.extend(get_point(points, x, y))
            '''

            # Collect metadata
            # meta['labelRecNum'] += [int(recording)]
            # meta['frameIndex'] += [frame]
            # meta['labelDotXCam'] += [dotInfo['XCam'][j]]
            # meta['labelDotYCam'] += [dotInfo['YCam'][j]]
            # meta['labelFaceGrid'] += [faceGrid]

            # meta['device']+= [info['DeviceName']]
            # meta['depth']+=[mean_depth]
            # # meta['landmarks']+=[landmarks]
            # meta['rects']+=[rects]

            meta['labelRecNum'].append(int(recording))
            meta['frameIndex'].append(frame)
            meta['labelDotXCam'].append(dotInfo['XCam'][j])
            meta['labelDotYCam'].append(dotInfo['YCam'][j])
            # meta['labelFaceGrid'].append(faceGrid)
            meta['device'].append(info['DeviceName'])
            meta['depth'].append(mean_depth)
            meta['rects'].append(rects)

        # save meta
        if i % 20 == 0 or i == len(recordings) - 1:
            metaFile = os.path.join(args.output_path, 'metadata.mat')
            # print('Writing out the metadata.mat to %s...' % metaFile)
            sio.savemat(metaFile, meta)

    
    # Integrate
    # meta['labelRecNum'] = np.stack(meta['labelRecNum'], axis = 0).astype(np.int16)
    # meta['frameIndex'] = np.stack(meta['frameIndex'], axis = 0).astype(np.int32)
    # meta['labelDotXCam'] = np.stack(meta['labelDotXCam'], axis = 0)
    # meta['labelDotYCam'] = np.stack(meta['labelDotYCam'], axis = 0)
    # meta['labelFaceGrid'] = np.stack(meta['labelFaceGrid'], axis = 0).astype(np.uint8)
    # # meta['landmarks'] = np.stack(meta['landmarks'], axis = 0).astype(np.float32)
    # meta['rects'] = np.stack(meta['rects'], axis = 0).astype(np.float32)

    # Load reference metadata
    print('Will compare to the reference GitHub dataset metadata.mat...')
    reference = sio.loadmat('/home/sigma/gaze/gaze/metadata/reference_metadata.mat', struct_as_record=False) 
    reference['labelRecNum'] = reference['labelRecNum'].flatten()
    reference['frameIndex'] = reference['frameIndex'].flatten()
    reference['labelDotXCam'] = reference['labelDotXCam'].flatten()
    reference['labelDotYCam'] = reference['labelDotYCam'].flatten()
    reference['labelTrain'] = reference['labelTrain'].flatten()
    reference['labelVal'] = reference['labelVal'].flatten()
    reference['labelTest'] = reference['labelTest'].flatten()

    # Find mapping
    mKey = np.array(['%05d_%05d' % (rec, frame) for rec, frame in zip(meta['labelRecNum'], meta['frameIndex'])], dtype=object)
    rKey = np.array(['%05d_%05d' % (rec, frame) for rec, frame in zip(reference['labelRecNum'], reference['frameIndex'])], dtype=object)

    mIndex = {k: i for i,k in enumerate(mKey)}
    rIndex = {k: i for i,k in enumerate(rKey)}
    mToR = np.zeros((len(mKey,)),int) - 1
    for i,k in enumerate(mKey):
        if k in rIndex:
            mToR[i] = rIndex[k]
        else:
            logError('Did not find rec_frame %s from the new dataset in the reference dataset!' % k)
    rToM = np.zeros((len(rKey,)),int) - 1
    for i,k in enumerate(rKey):
        if k in mIndex:
            rToM[i] = mIndex[k]
        else:
            logError('Did not find rec_frame %s from the reference dataset in the new dataset!' % k, critical = False)
            #break

    # Copy split from reference
    meta['labelTrain'] = np.zeros((len(meta['labelRecNum'],)),np.bool)
    meta['labelVal'] = np.ones((len(meta['labelRecNum'],)),np.bool) # default choice
    meta['labelTest'] = np.zeros((len(meta['labelRecNum'],)),np.bool)

    validMappingMask = mToR >= 0
    meta['labelTrain'][validMappingMask] = reference['labelTrain'][mToR[validMappingMask]]
    meta['labelVal'][validMappingMask] = reference['labelVal'][mToR[validMappingMask]]
    meta['labelTest'][validMappingMask] = reference['labelTest'][mToR[validMappingMask]]

    # Write out metadata
    metaFile = os.path.join(args.output_path, 'metadata.mat')
    print('Writing out the metadata.mat to %s...' % metaFile)
    sio.savemat(metaFile, meta)
    
    # Statistics
    nMissing = np.sum(rToM < 0)
    nExtra = np.sum(mToR < 0)
    totalMatch = len(mKey) == len(rKey) and np.all(np.equal(mKey, rKey))
    print('======================\n\tSummary\n======================')    
    print('Total added %d frames from %d recordings.' % (len(meta['frameIndex']), len(np.unique(meta['labelRecNum']))))
    if nMissing > 0:
        print('There are %d frames missing in the new dataset. This may affect the results. Check the log to see which files are missing.' % nMissing)
    else:
        print('There are no missing files.')
    if nExtra > 0:
        print('There are %d extra frames in the new dataset. This is generally ok as they were marked for validation split only.' % nExtra)
    else:
        print('There are no extra files that were not in the reference dataset.')
    if totalMatch:
        print('The new metadata.mat is an exact match to the reference from GitHub (including ordering)')

    #import pdb; pdb.set_trace()
    input("Press Enter to continue...")

def get_center(bbox):
    x, y, w, h = map(int, bbox)
    return x + w // 2, y + h // 2

# Safely get a point at (x, y) in image bounds
def get_point(points, x, y):
    H, W = points.shape[:2]
    x = np.clip(x, 0, W - 1)
    y = np.clip(y, 0, H - 1)
    return points[y, x]  # note: (row, col)


def readJson(filename):
    if not os.path.isfile(filename):
        logError('Warning: No such file %s!' % filename)
        return None

    with open(filename) as f:
        try:
            data = json.load(f)
        except:
            data = None

    if data is None:
        logError('Warning: Could not read file %s!' % filename)
        return None

    return data

def preparePath(path, clear = False):
    if not os.path.isdir(path):
        os.makedirs(path, mode=0o777, exist_ok=True)
    if clear:
        files = os.listdir(path)
        for f in files:
            fPath = os.path.join(path, f)
            if os.path.isdir(fPath):
                shutil.rmtree(fPath)
            else:
                os.remove(fPath)

    return path

def logError(msg, critical = False):
    print(msg)
    if critical:
        sys.exit(1)


def cropImage(img, bbox):
    bbox = np.array(bbox, int)

    aSrc = np.maximum(bbox[:2], 0)
    bSrc = np.minimum(bbox[:2] + bbox[2:], (img.shape[1], img.shape[0]))

    aDst = aSrc - bbox[:2]
    bDst = aDst + (bSrc - aSrc)

    res = np.zeros((bbox[3], bbox[2], img.shape[2]), img.dtype)    
    res[aDst[1]:bDst[1],aDst[0]:bDst[0],:] = img[aSrc[1]:bSrc[1],aSrc[0]:bSrc[0],:]

    return res

def cropImage2D(img, bbox):
    bbox = np.array(bbox, int)
    aSrc = np.maximum(bbox[:2], 0)
    bSrc = np.minimum(bbox[:2] + bbox[2:], (img.shape[1], img.shape[0]))

    aDst = aSrc - bbox[:2]
    bDst = aDst + (bSrc - aSrc)

    res = np.zeros((bbox[3], bbox[2]), img.dtype)
    res[aDst[1]:bDst[1], aDst[0]:bDst[0]] = img[aSrc[1]:bSrc[1], aSrc[0]:bSrc[0]]
    return res

def make_face_grid(face_rect, img_size, grid_size=(25, 25)):
        """
        face_rect: [x1, y1, x2, y2]，人脸在图像中的像素坐标
        img_size: (width, height) 图像大小
        grid_size: (grid_w, grid_h) 网格大小

        返回：
            numpy array (grid_h, grid_w)，
            人脸区域置1，其余0
        """
        img_w, img_h = img_size
        grid_w, grid_h = grid_size

        x1, y1, x2, y2 = face_rect

        # 计算人脸框宽高
        w = x2 - x1
        h = y2 - y1

        # 映射到网格坐标，注意用float计算，最后用int裁剪索引
        gx = int(x1 / img_w * grid_w)
        gy = int(y1 / img_h * grid_h)
        gw = max(int(w / img_w * grid_w), 1)
        gh = max(int(h / img_h * grid_h), 1)

        # 确保范围合法
        gx = np.clip(gx, 0, grid_w-1)
        gy = np.clip(gy, 0, grid_h-1)
        if gx + gw > grid_w:
            gw = grid_w - gx
        if gy + gh > grid_h:
            gh = grid_h - gy

        grid = np.zeros((grid_h, grid_w), dtype=np.uint8)
        grid[gy:gy+gh, gx:gx+gw] = 1

        return grid.flatten()

def normalize_rect(rect, w, h):
    x1, y1, x2, y2 = rect
    return [
        x1 / w,
        y1 / h,
        x2 / w,
        y2 / h
    ]

def normalize_keypoints(keypoints, image_width, image_height):
    normalized = []
    mean_z = 0
    for pt in keypoints:
        x_norm = pt["x"] / image_width
        y_norm = pt["y"] / image_height
        z = pt["z"]  # z 本身是归一化值，通常不变
        mean_z+=z
        normalized.extend([x_norm, y_norm, z])
    return normalized, mean_z/len(keypoints)



if __name__ == "__main__":
    main()
    print('DONE')
