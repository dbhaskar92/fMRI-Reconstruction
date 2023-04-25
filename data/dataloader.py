import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable


class NumpyDataset(Dataset):
    def __init__(self, meg_dir, fmri_dir): 
        
        self.npy_files = [] 
        # self.labels = [] # for subject based labels  
        for i, subdir in enumerate(sorted(os.listdir(meg_dir))):
            for file in sorted(os.listdir(os.path.join(meg_dir, subdir))):
                if file.endswith('.npy'):
                    self.npy_files.append(os.path.join(meg_dir, subdir, file))
                    # self.labels.append(i)

        self.fmri_subs = [x for x in os.listdir(fmri_dir)] 
        self.fmri_dir = fmri_dir  

    def __len__(self):
        return len(self.npy_files)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist() 
        meg_npy_path = self.npy_files[idx]
        meg_sample = np.load(meg_npy_path)
        label = int(meg_npy_path[-7:-4])
        subj = meg_npy_path.split(os.sep)[-2]

        rand_fmri_sub = str(np.random.choice(self.fmri_subs, 1, replace=False).tolist()[0]) 
        fmri_sample = np.load(os.path.join(self.fmri_dir, rand_fmri_sub, (rand_fmri_sub+meg_npy_path[-12:])))  
        return meg_sample, fmri_sample, label, subj

 
 
class NumpyMetaDataset(Dataset):
    def __init__(self, meg_dir, fmri_dir, n_way, batch_size=1, shuffle=True):  
        self.meg_dir = meg_dir  
        self.fmri_dir = fmri_dir  
        self.n_way = n_way 
        self.label = [x for x in range(self.n_way)] 
        self.label = Variable(torch.tensor(self.label)) 
        self.batch_size = batch_size
        self.shuffle = shuffle


    def rand_sub(self):
        self.meg_subs = [x for x in os.listdir(self.meg_dir)] 
        self.fmri_subs = [x for x in os.listdir(self.fmri_dir)] 
 
        rand_meg_sub = str(np.random.choice(self.meg_subs, 1, replace=False).tolist()[0]) 
        rand_fmri_sub = str(np.random.choice(self.fmri_subs, 1, replace=False).tolist()[0]) 
 
        return rand_meg_sub, rand_fmri_sub

    def rand_time(self):
        return np.random.choice(self.class_list, self.n_way, replace=False).tolist()

     
    def __getitem__(self, idx):
        rand_meg_sub, rand_fmri_sub = self.rand_sub()
         
        n_folders_m = self.n_folders(os.path.join(self.meg_dir, rand_meg_sub)+'/')
        n_folders_f = self.n_folders(os.path.join(self.fmri_dir, rand_fmri_sub)+'/') 
        n_time = min(n_folders_m, n_folders_f) 
        self.n_time_pints(n_time= n_time)  

        rand_time_points = self.rand_time()
        
        meg_sample = [np.load(os.path.join(self.meg_dir, rand_meg_sub, (rand_meg_sub+'-min-'+rand_t+'.npy'))) 
                    for rand_t in rand_time_points] 

        fmri_sample = [np.load(os.path.join(self.fmri_dir, rand_fmri_sub, (rand_fmri_sub+'-min-'+rand_t+'.npy'))) 
                    for rand_t in rand_time_points]

        y_batch = list(map(int, rand_time_points))
        y_meta = self.label

        return (meg_sample, fmri_sample, y_meta, y_batch), [rand_time_points, rand_fmri_sub, rand_meg_sub] 


    def n_time_pints(self, n_time):    
        self.class_list = []
        class_couter = 1
        for i in range(n_time):
            if class_couter < 10:
                self.class_list.append('00'+str(class_couter))
            elif class_couter > 99:
                self.class_list.append(str(class_couter))
            else:
                self.class_list.append('0'+str(class_couter)) 
            class_couter += 1

    def n_folders(self, dir_path):  
        return len(os.listdir(dir_path))
  
    def __len__(self):
        return self.batch_size
 
 