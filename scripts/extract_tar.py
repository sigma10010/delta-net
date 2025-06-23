import os
import tarfile

path = "/home/sigma/gaze/datasets/GazeCapture/"
path_des = "/home/sigma/gaze/datasets/gc/"
files = os.listdir(path)
print(len(files))

for i, filename in enumerate(files):
    # if i<1017:
        # continue
    print(i)
    if filename.endswith('tar.gz'):
        # open file 
        file = tarfile.open(path+filename) 
        # extract files 
        file.extractall(path_des) 
        # close file 
        file.close() 
print('done')
