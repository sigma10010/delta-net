import os
import numpy as np
import scipy.io as sio

num_calib=9
filename = '../metadata/metadata_device.mat'
metadata = sio.loadmat(filename, squeeze_me=True, struct_as_record=False)
rec_nums = np.unique(metadata['labelRecNum'])
rec_statistics = {}
for rec_num in rec_nums:
    mask = metadata['labelRecNum']==rec_num
    indices = np.argwhere(mask)[:,0]
    rec_statistics[str(rec_num)] = len(indices)
    if len(indices)<=num_calib:
        metadata['labelTrain'][indices]=0
        print(rec_num, len(indices))
    
#print(rec_statistics)
# Write out metadata
metaFile = os.path.join('../metadata/', 'metadata%d.mat'%num_calib)
print('Writing out the metadata%d.mat to %s...' %(num_calib, metaFile))
sio.savemat(metaFile, metadata)
print('done')
