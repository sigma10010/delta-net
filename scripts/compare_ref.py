import scipy.io as sio
import numpy as np
import os

def logError(msg, critical = False):
    print(msg)
    if critical:
        sys.exit(1)

# --- 1. Load from .mat ---
meta = sio.loadmat('/home/sigma/gaze/datasets/gc_mp/merged_all.mat', struct_as_record=False) 
if 'labelFaceGrid' in meta:
    del meta['labelFaceGrid']
# --- 2. Convert relevant fields for appending (only those you will update) ---
for key in ['labelRecNum', 'frameIndex', 'labelDotXCam', 'labelDotYCam',
            'labelFaceGrid', 'device', 'depth', 'rects']:
    if key in meta and isinstance(meta[key], np.ndarray):
        if key == 'labelFaceGrid' or key == 'rects':
            # For 2D arrays, use list of arrays for efficient appending
            meta[key] = [meta[key][i] for i in range(meta[key].shape[0])]
        else:
            meta[key] = meta[key].flatten().tolist()

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
meta['labelTrain'] = np.zeros((len(meta['labelRecNum'],)),np.bool_)
meta['labelVal'] = np.ones((len(meta['labelRecNum'],)),np.bool_) # default choice
meta['labelTest'] = np.zeros((len(meta['labelRecNum'],)),np.bool_)

validMappingMask = mToR >= 0
meta['labelTrain'][validMappingMask] = reference['labelTrain'][mToR[validMappingMask]]
meta['labelVal'][validMappingMask] = reference['labelVal'][mToR[validMappingMask]]
meta['labelTest'][validMappingMask] = reference['labelTest'][mToR[validMappingMask]]

# Write out metadata
metaFile = os.path.join('/home/sigma/gaze/datasets/gc_mp/', 'metadata.mat')
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