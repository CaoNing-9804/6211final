import sys
sys.path.append(r"../..")
import dataloader.augmentations as aug

import os
import torch
import numpy as np
import random
from tqdm import tqdm
from sklearn.model_selection import train_test_split


output_dir = "../../data/ISIR"
if not os.path.exists(output_dir):
  os.makedirs(output_dir)
  
  

val = torch.load("../../data/ISIR/val_no_aug.pt")
X_val = val["samples"].numpy().transpose(0,2,1)
Y_val = val["labels"].numpy()

test = torch.load("../../data/ISIR/test_no_aug.pt")
X_test = test["samples"].numpy().transpose(0,2,1)
y_test = test["labels"].numpy()


X = []
Y = []

for i in range(len(X_val)):
  for j in range(200):
    tp = random.randint(0,95)
    
    tpx = aug.jitter(aug.scaling(X_val[i,tp:tp+96,:].reshape(1,96,26) ),5).reshape(96,26)
    X.append(tpx)
    Y.append(Y_val[i])
    
X = np.array(X)
Y = np.array(Y)    
X_test = X_test[:,-96:,:] 
X_val =  X_val[:,-96:,:]
y_val = Y_val


X_trs, _, y_trs, _ = train_test_split(X,Y,test_size=0.001,random_state=512)
X_val = np.concatenate((X_val, X_test[:X_test.shape[0]//4,:,:]), axis=0)
y_val = np.concatenate((y_val, y_test[:X_test.shape[0]//4]),axis = 0)

print(X_val.shape)
print(X_test.shape)
print(X_trs.shape)
print(y_val.shape)
print(y_test.shape)
print(y_trs.shape)


dat_dict = dict()
dat_dict["samples"] = torch.from_numpy(X_trs).transpose(1,2)
dat_dict["labels"] = torch.from_numpy(y_trs)
torch.save(dat_dict, os.path.join(output_dir, "train_supervised.pt"))  

dat_dict = dict()
dat_dict["samples"] = torch.from_numpy(X_test).transpose(1,2)
dat_dict["labels"] = torch.from_numpy(y_test)
torch.save(dat_dict, os.path.join(output_dir, "test.pt"))

dat_dict = dict()
dat_dict["samples"] = torch.from_numpy(X_val).transpose(1,2)
dat_dict["labels"] = torch.from_numpy(y_val)
torch.save(dat_dict, os.path.join(output_dir, "val.pt"))  
