import os
import cv2
import torch
# from moge.model.v1 import MoGeModel
from moge.model.v2 import MoGeModel # Let's try MoGe-2
import numpy as np

# load npz
# data = np.load("output_data.npz")
# points = data["points"]
# depth = data["depth"]
# mask = data["mask"]
# intrinsics = data["intrinsics"]
# normal = data.get("normal")  # Optional


device = torch.device("cuda")
model = MoGeModel.from_pretrained("/home/sigma/moge/moge-2-vitl-normal/model.pt").to(device)

path = "/home/sigma/gaze/datasets/gc"
labelRecNums = os.listdir(path)
total_num = len(labelRecNums)
for i, labelRecNum in enumerate(labelRecNums):
    if i<44:
        continue
    print(labelRecNum)
    print('(%d/%d)'%(i, total_num))
    path_geo = os.path.join(path, labelRecNum, 'geometry')
    os.makedirs(path_geo, exist_ok=True)
    path_frames = os.path.join(path, labelRecNum, 'frames')
    fimgs = os.listdir(path_frames)
    for j, fimg in enumerate(fimgs):
        path_img = os.path.join(path_frames, fimg)

        # Read the input image and convert to tensor (3, H, W) with RGB values normalized to [0, 1]
        input_image = cv2.cvtColor(cv2.imread(path_img), cv2.COLOR_BGR2RGB)                       
        input_image = torch.tensor(input_image / 255, dtype=torch.float32, device=device).permute(2, 0, 1)
        # Infer 
        output = model.infer(input_image)
        # Convert output dictionary from tensors to NumPy arrays
        output_np = {k: v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else v for k, v in output.items()}

        output_file = fimg.split('.')[0]+'.npz'
        np.savez_compressed(os.path.join(path_geo, output_file), **output_np)