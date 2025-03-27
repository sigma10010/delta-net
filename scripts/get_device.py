import os
import numpy as np
import scipy.io as sio
import json

num_calib=9
filename = '../gc_/metadata.mat'
metadata = sio.loadmat(filename, squeeze_me=True, struct_as_record=False)
rec_nums = np.unique(metadata['labelRecNum'])
rec_statistics = {}

rec_devices = {}
for rec_num in rec_nums:
    path = '../gc/%05d/info.json'%rec_num
    f = open(path)
    data = json.load(f)
    rec_devices[str(rec_num)] = data['DeviceName']
    
#print(devices)
devices = []
for i in range(len(metadata['labelRecNum'])):
    rec = metadata['labelRecNum'][i]
    devices.append(rec_devices[str(rec)])
metadata['device'] = np.array(devices)
# Write out metadata
metaFile = os.path.join('../metadata', 'metadata_device.mat')
print('Writing out the metadata to %s...' %metaFile)
sio.savemat(metaFile, metadata)
print('done')