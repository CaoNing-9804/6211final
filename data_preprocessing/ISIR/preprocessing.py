import os
import json
import numpy as np
import torch
import random
from tqdm import tqdm


data_dir = "../../../ISIR"
output_dir = "../../data/ISIR"
if not os.path.exists(output_dir):
  os.makedirs(output_dir)

lst = []
files=os.listdir(data_dir)
for f in files: 
    path = os.path.join(data_dir, f)
    if os.path.isdir(path):
        vws = os.listdir(path)
        for v in vws:
            if "090" in v:
                lst.append(os.path.join(path, v))

seqs = [] 
for d in tqdm(lst):
    seq = []
    js_names = os.listdir(d) 
    
    if len(js_names) < 16:
      continue
        
    for j in js_names:
        filename = os.path.join(d, j)
        with open(file = filename, mode='r') as f:
            data = json.load(f)
        points = data["people"][0]["pose_keypoints_2d"]
        rm_points = {"start":14, "end":17}
        idx = [i for i in range(len(points)) if i%3 != 2 and i not in list(range(rm_points["start"]*3,rm_points["end"]*3+3))+[0,1]]
        pv = [points[i] for i in idx] 
        seq.append(pv) 
        
    tpx = random.randint(0,20)  
    seq = seq*8
    if len(seq) < 96:
      print("fuck you!")
    seqs.append(seq[tpx:tpx+96])
    

X_train = np.array(seqs)
#samples*96*26
y_train = np.zeros((X_train.shape[0],1))

#deal with outlier
for i in range(X_train.shape[0]):
  for j in range(X_train.shape[1]):
    for k in range(X_train.shape[2]):
      if X_train[i,j,k] >= 800 or X_train[i,j,k] == 0:
        if j != 0:
          X_train[i,j,k] = X_train[i,j-1,k]
        else:
          X_train[i,j,k] = X_train[i,j+1,k]
        

#normalize
tp = X_train[:,:,0:2]
X_train = X_train - np.tile(tp,(1,1,13))

dat_dict = dict()
dat_dict["samples"] = torch.from_numpy(X_train).transpose(1,2)
print(dat_dict["samples"].shape)
dat_dict["labels"] = torch.from_numpy(y_train)
torch.save(dat_dict, os.path.join(output_dir, "train.pt"))









