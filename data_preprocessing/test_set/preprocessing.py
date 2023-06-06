import os
import json
import numpy as np
import torch
from tqdm import tqdm
from sklearn.model_selection import train_test_split

data_dir = "../../../test_data"
output_dir = "../../data/ISIR"
if not os.path.exists(output_dir):
  os.makedirs(output_dir)
  
people_lst_test = []
label_lst_test = []
people_lst_tune = []
label_lst_tune = []

files=os.listdir(data_dir) #PD,NM,KOA

for f in files:
  if "NM" in f:
    peop_path = os.path.join(data_dir, f)#NM
    peop_l = os.listdir(peop_path)
    for p in peop_l:
      if "_01_" in p:
        people_lst_tune.append(os.path.join(peop_path, p))
        label_lst_tune.append(0)
        continue
      else:
        people_lst_test.append(os.path.join(peop_path, p))
        label_lst_test.append(0)
  
  else:
    cl_path = os.path.join(data_dir, f)
    cl_l = os.listdir(cl_path) #PD_SV
    for c in cl_l:
      peop_path = os.path.join(cl_path, c)#NM
      peop_l = os.listdir(peop_path)
      for p in peop_l:
      
        if "_01_" in p:
          people_lst_tune.append(os.path.join(peop_path, p))
          if "PD" in f:
            label_lst_tune.append(1)
          else:
            label_lst_tune.append(2)
        
        else:
          people_lst_test.append(os.path.join(peop_path, p))
          if "PD" in f:
            label_lst_test.append(1)
          else:
            label_lst_test.append(2)


seqs = [] 
for d in tqdm(people_lst_test):
    seq = []
    js_names = os.listdir(d) 
    
    idx = []
    for i in list(range(1,8))+list(range(9,15)):
      idx.append(i*3)
      idx.append(i*3+1)
    
    for j in js_names:
        filename = os.path.join(d, j)
        with open(file = filename, mode='r') as f:
            data = json.load(f)
        
        if data["people"] != []:
          points = data["people"][0]["pose_keypoints_2d"]
        else:
          continue
                
        pv = [points[i] for i in idx] #pv is a single point
        seq.append(pv) # a sequence
        
    seq = seq*4
    seqs.append(seq[-192:])

X_test = np.array(seqs)
Y_test = np.array(label_lst_test).T


seqs = [] 
for d in tqdm(people_lst_tune):
    seq = []
    js_names = os.listdir(d) 
    
    idx = []
    for i in list(range(1,8))+list(range(9,15)):
      idx.append(i*3)
      idx.append(i*3+1)
    
    for j in js_names:
        filename = os.path.join(d, j)
        with open(file = filename, mode='r') as f:
            data = json.load(f)
        
        if data["people"] != []:
          points = data["people"][0]["pose_keypoints_2d"]
        else:
          continue
                
        pv = [points[i] for i in idx] #pv is a single point
        seq.append(pv) # a sequence
        
    seq = seq*4
    seqs.append(seq[-192:])

X_tune = np.array(seqs)
Y_tune = np.array(label_lst_tune).T


#deal with outlier
for i in range(X_test.shape[0]):
  for j in range(X_test.shape[1]):
    for k in range(X_test.shape[2]):
      if  X_test[i,j,k] == 0:
        if j != 0:
          X_test[i,j,k] = X_test[i,j-1,k]
        else:
          X_test[i,j,k] = X_test[i,j+1,k]
          

for i in range(X_tune.shape[0]):
  for j in range(X_tune.shape[1]):
    for k in range(X_tune.shape[2]):
      if  X_tune[i,j,k] == 0:
        if j != 0:
          X_tune[i,j,k] = X_tune[i,j-1,k]
        else:
          X_tune[i,j,k] = X_tune[i,j+1,k]
        

#normalize
tp = X_test[:,:,0:2]
X_test = X_test - np.tile(tp,(1,1,13))

tp = X_tune[:,:,0:2]
X_tune = X_tune - np.tile(tp,(1,1,13))
tp = [i for i in range(26) if i % 2 == 0]
X_tune[:,:,tp] = - X_tune[:,:,tp]

X = np.concatenate((X_test, X_tune))
Y = np.concatenate((Y_test, Y_tune))


X_val, X_test, y_val, y_test = train_test_split(X,Y,test_size=0.9,random_state=525)

print(X_val.shape)
print(X_test.shape)
print(y_val.shape)
print(y_test.shape)


dat_dict = dict()
dat_dict["samples"] = torch.from_numpy(X_val).transpose(1,2)
dat_dict["labels"] = torch.from_numpy(y_val)
torch.save(dat_dict, os.path.join(output_dir, "val_no_aug.pt"))

dat_dict = dict()
dat_dict["samples"] = torch.from_numpy(X_test).transpose(1,2)
dat_dict["labels"] = torch.from_numpy(y_test)
torch.save(dat_dict, os.path.join(output_dir, "test_no_aug.pt"))
