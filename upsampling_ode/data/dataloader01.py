from torch.utils.data import Dataset
import os, sys
import numpy as np
import torch
import random
import math
from random import choices
import tqdm
 
class ContrastiveDataLoader():
    
    def __init__(self, device, n_way = 12):  
         
        self.fmri_dir = '/home/aa2793/scratch60/datasets/fmri-meg/fmri/samples_30/' 
        self.meg_dir = '/home/aa2793/scratch60/datasets/fmri-meg/meg/samples_480/'  
        self.meg_sub_list = os.listdir(self.meg_dir)   
        self.fmri_sub_list = os.listdir(self.fmri_dir)
        self.n_way = n_way
        self.device = device
  
    
    def __getitem__(self, i):
        # pick a random subject 
        meg_rand_sub = choices(self.meg_sub_list, k=1)[0]
        fmri_rand_sub = choices(self.fmri_sub_list, k=1)[0]

        file_list_meg = os.listdir(os.path.join(self.meg_dir, meg_rand_sub))
        
        # sampling miniute wihout replacement from inside the subject
        sampled_meg_list = random.sample(file_list_meg, self.n_way) 
        x_meg = [np.load(os.path.join(self.meg_dir, meg_rand_sub, sampled_meg_list[i])) for i in range(self.n_way)]
        
        sampled_fmri_list = [sampled_meg_list[i][6:] for i in range(self.n_way)]
        sampled_fmri_list = [fmri_rand_sub+sampled_fmri_list[i] for i in range(self.n_way)]
        x_fmri = [np.load(os.path.join(self.fmri_dir, fmri_rand_sub, sampled_fmri_list[i])) for i in range(self.n_way)]
        
        x_fmri, x_meg = torch.tensor(x_fmri).float().to(self.device).squeeze(1), torch.tensor(x_meg).float().to(self.device).squeeze(1)
        y = torch.tensor([i for i in range(self.n_way)]) #.to(self.device)
 
        # print(meg_rand_sub,  fmri_rand_sub, '\n', sampled_meg_list )
        # import pdb; pdb.set_trace()
        x_meg = torch.nan_to_num(x_meg, nan=2.0, posinf=0.0)
        return (x_fmri, x_meg, y)

    def __len__(self):
        return 10
    #     self.fmri_sub_list = set([x[:11] for x in self.fmri_dir_list]) 
    #     self.index_randomization()  
        
        
    # def index_randomization(self):
    #     # discarding the last batch  
    #     batch_vector = random.sample(range(len(self.meg_dir_list)), len(self.meg_dir_list))
    #     self.n_batches = int(math.floor(len(self.meg_dir_list)/self.batch_size ))*self.batch_size 
    #     batch_vector = batch_vector[:self.n_batches] 
    #     batch_vector = np.expand_dims(batch_vector, axis=0)
    #     self.batch_vector = np.reshape(batch_vector, (-1, self.batch_size ))
    #     self.batch_conter = 0 
    #     self.fmri_rand_sub = [random.sample(list(self.fmri_sub_list), 1) for i in range(self.batch_size)]  
    #     self.fmri_rand_sub = np.reshape(self.fmri_rand_sub, (-1, self.batch_size))[0]
        
    # def __getitem__(self, k):
    #     """ loads a subject's right and left hippocampus recording.  
    #     """  
    #     self.batch_conter += 1
    #     if self.batch_conter > self.n_batches: 
    #         self.batch_conter = 0
    #         self.index_randomization() 
    #     # meg_sub_index = [self.meg_dir_list[k] for k in self.batch_vector[self.batch_conter]]   

    #     x_meg = [np.load(self.meg_dir+self.meg_dir_list[k]) for k in self.batch_vector[self.batch_conter]]
    #     x_meg = torch.tensor(x_meg).squeeze(1) 
    #     #meg
    #     # import pdb;pdb.set_trace() 
    #     # for i, k in enumerate(self.batch_vector[self.batch_conter]):
    #     #     x_meg_k = torch.from_numpy(np.load(self.meg_dir+self.meg_dir_list[k])) 
    #     #     if i == 0 :
    #     #         x_meg = x_meg_k
    #     #     else:
    #     #         x_meg = torch.cat((x_meg, x_meg_k), dim=0)

    #     y_meg = torch.tensor([int(self.meg_dir_list[k][11:-4]) for k in self.batch_vector[self.batch_conter]]) 

    #     # fmri
    #     meg_dir_list = [self.meg_dir_list[k][11:] for k in self.batch_vector[self.batch_conter]] 
    #     for i in range(len(meg_dir_list)):
    #         x_fmri_i = np.load(self.fmri_dir+self.fmri_rand_sub[i]+meg_dir_list[i])
    #         x_fmri_i = torch.tensor(x_fmri_i)
    #         if i == 0 :
    #             x_fmri = x_fmri_i
    #         else:
    #             x_fmri = torch.cat((x_fmri, x_fmri_i), dim=0)  
    #     # x_meg = x_fmri
    #     # fmri_sub_index = [(self.fmri_rand_sub[i]+meg_dir_list[i]) for i in range(len(meg_dir_list))]  
    #     return (x_meg, x_fmri, y_meg) #, meg_sub_index, fmri_sub_index)  

    # def __len__(self):
    #     return self.n_batches 





# fmri_meg_data = ContrastiveDataLoader(device = 'cuda' ,n_way = 10)

# for (x_fmri, x_meg, y) in fmri_meg_data:
#     a = 1
    







class MiniuteLoader():
    
    def __init__(self, batch_size = 12):  
        
        self.batch_size = batch_size
        self.fmri_dir = '/home/aa2793/scratch60/datasets/fmri-meg/fmri/samples/' 
        self.meg_dir = '/home/aa2793/scratch60/datasets/fmri-meg/meg/samples_150/'  
        self.meg_dir_list = os.listdir(self.meg_dir)   
        self.fmri_dir_list = os.listdir(self.fmri_dir) 
        self.fmri_sub_list = set([x[:11] for x in self.fmri_dir_list]) 
        self.index_randomization()  
        
        
    def index_randomization(self):
        # discarding the last batch  
        batch_vector = random.sample(range(len(self.meg_dir_list)), len(self.meg_dir_list))
        self.n_batches = int(math.floor(len(self.meg_dir_list)/self.batch_size ))*self.batch_size 
        batch_vector = batch_vector[:self.n_batches] 
        batch_vector = np.expand_dims(batch_vector, axis=0)
        self.batch_vector = np.reshape(batch_vector, (-1, self.batch_size ))
        self.batch_conter = 0 
        self.fmri_rand_sub = [random.sample(list(self.fmri_sub_list), 1) for i in range(self.batch_size)]  
        self.fmri_rand_sub = np.reshape(self.fmri_rand_sub, (-1, self.batch_size))[0]
        
    def __getitem__(self, k):
        """ loads a subject's right and left hippocampus recording.  
        """  
        self.batch_conter += 1
        if self.batch_conter > self.n_batches: 
            self.batch_conter = 0
            self.index_randomization() 
        # meg_sub_index = [self.meg_dir_list[k] for k in self.batch_vector[self.batch_conter]]   

        x_meg = [np.load(self.meg_dir+self.meg_dir_list[k]) for k in self.batch_vector[self.batch_conter]]
        x_meg = torch.tensor(x_meg).squeeze(1) 
        #meg
        # import pdb;pdb.set_trace() 
        # for i, k in enumerate(self.batch_vector[self.batch_conter]):
        #     x_meg_k = torch.from_numpy(np.load(self.meg_dir+self.meg_dir_list[k])) 
        #     if i == 0 :
        #         x_meg = x_meg_k
        #     else:
        #         x_meg = torch.cat((x_meg, x_meg_k), dim=0)

        y_meg = torch.tensor([int(self.meg_dir_list[k][11:-4]) for k in self.batch_vector[self.batch_conter]]) 

        # fmri
        meg_dir_list = [self.meg_dir_list[k][11:] for k in self.batch_vector[self.batch_conter]] 
        for i in range(len(meg_dir_list)):
            x_fmri_i = np.load(self.fmri_dir+self.fmri_rand_sub[i]+meg_dir_list[i])
            x_fmri_i = torch.tensor(x_fmri_i.float())
            if i == 0 :
                x_fmri = x_fmri_i
            else:
                x_fmri = torch.cat((x_fmri, x_fmri_i), dim=0)  
        # x_meg = x_fmri
        # fmri_sub_index = [(self.fmri_rand_sub[i]+meg_dir_list[i]) for i in range(len(meg_dir_list))]  
        x_meg = torch.nan_to_num(x_meg, nan=2.0, posinf=0.0)
        return (x_meg, x_fmri, y_meg) #, meg_sub_index, fmri_sub_index)  

    def __len__(self):
        return self.n_batches 
 







































# # import pdb;pdb.set_trace()
# print('--------------------------------------')
# for j in range(5):  
#     fmri_meg_data = MiniuteLoader(batch_size = 10)
#     for i, (x_m, x_f, y, meg_index, fmri_index) in enumerate(fmri_meg_data): 
#         # import pdb;pdb.set_trace() 
#         print(x_m.shape, x_f.shape, len(y)) #, sys.getsizeof(x_m)/(1024**3)) 
        # import pdb;pdb.set_trace() 
            # print(meg_index)
        # print(fmri_index)  



















# class MiniuteLoader():
    
#     def __init__(self, batch_size = 12):  
        
#         self.batch_size = batch_size
#         self.fmri_dir = '/home/aa2793/scratch60/datasets/fmri-meg/fmri/samples/' 
#         self.meg_dir = '/home/aa2793/scratch60/datasets/fmri-meg/meg/samples/'  
#         self.meg_dir_list = os.listdir(self.meg_dir)   
#         self.fmri_dir_list = os.listdir(self.fmri_dir) 
#         self.fmri_sub_list = set([x[:11] for x in self.fmri_dir_list]) 
#         self.index_randomization()  
        
        
#     def index_randomization(self):
#         # discarding the last batch  
#         batch_vector = random.sample(range(len(self.meg_dir_list)), len(self.meg_dir_list))
#         self.n_batches = int(math.floor(len(self.meg_dir_list)/self.batch_size ))*self.batch_size 
#         batch_vector = batch_vector[:self.n_batches] 
#         batch_vector = np.expand_dims(batch_vector, axis=0)
#         self.batch_vector = np.reshape(batch_vector, (-1, self.batch_size ))
#         self.batch_conter = 0

#         self.fmri_rand_sub = [random.sample(self.fmri_sub_list, 1) for i in range(self.batch_size)] 
#         self.fmri_rand_sub = np.reshape(self.fmri_rand_sub, (-1, self.batch_size))[0]
        
        
#     def __getitem__(self, i):
#         """ loads a subject's right and left hippocampus recording.  
#         """  
#         self.batch_conter += 1
#         if self.batch_conter > self.n_batches: 
#             self.batch_conter = 0
#             self.index_randomization() 
#         meg_sub_index = [self.meg_dir_list[k] for k in self.batch_vector[self.batch_conter]]   
#         # x_meg = [np.load(self.meg_dir+self.meg_dir_list[k]) for k in self.batch_vector[self.batch_conter]]
#         y_meg = [int(self.meg_dir_list[k][11:-4]) for k in self.batch_vector[self.batch_conter]]
         

#         # upload the fmri with the same mini of the meg
#         meg_dir_list = [self.meg_dir_list[k][11:] for k in self.batch_vector[self.batch_conter]]
#         fmri_sub_index = [(self.fmri_rand_sub[i]+meg_dir_list[i]) for i in range(len(meg_dir_list))]
#         x_fmri = [np.load(self.fmri_dir+self.fmri_rand_sub[i]+meg_dir_list[i]) for i in range(len(meg_dir_list))]
#         x_fmri = np.array(x_fmri).squeeze(1)

#         x_meg = x_fmri

#         x_meg, y_meg = torch.tensor(x_meg).squeeze(1), torch.tensor(y_meg)  
 
#         return (x_meg, x_fmri, y_meg, meg_sub_index, fmri_sub_index) 

#     def __len__(self):
#         return self.n_batches