import os 
import numpy as np
import tqdm 
import pandas as pd 
import math 
import torch 
from preproc_01_brain_loader import BrainDataLoader 
# class BrainDataLoader:
#     def __init__(self, dataset_mode = 'fmri'):
#         """Convert fMRI and MEG's recordings in different rurns/parts to miniute.  
#            there are two methods (*_sub_dic_list) uploading the fmri and meg recordings. 
#         Args:
#             dataset_mode {fmri or meg}: types of the data which could be fmri or meg.
#         """ 
        
#         self.sub_run_dic, self.sub_run_list = {}, []
#         self.run_list, self.sub_list = [], [] 

#         if dataset_mode == 'fmri':
#             data_dir = '/gpfs/milgram/project/turk-browne/projects/StudyForrest/fmri_meg_aligned_segments/fmri/fsaverage_trimmed/'
#             self.data_dir = data_dir
#             self.dir_list = os.listdir(data_dir) 
#             self.fmri_sub_dic_list()  

#         elif dataset_mode == 'meg':
#             data_dir = '/gpfs/milgram/project/turk-browne/projects/StudyForrest/fmri_meg_aligned_segments/meg/trimmed_new/'
#             self.data_dir = data_dir
#             self.dir_list = os.listdir(data_dir)  
#             self.meg_sub_dic_list()  

#         else:
#             raise TypeError("only fmri or meg is allowed!")


#     def fmri_sub_dic_list(self):
#         """ loads a subject's right and left hippocampus fmri recording. 
#         """ 

#         # iterating through the subjects 
#         file_counter = 0 
#         for i in range(1, len(self.dir_list)): 
#             sub_add_list = True
#             if i < 10:
#                 sub_i = '0'+ str(i)
#             else: 
#                 sub_i = str(i)   
#             # iterating through the runs 
#             run_list = []
#             for j in range(1, len(self.dir_list)):
#                 if j < 10:
#                     run_j = '0'+ str(j)
#                 else: 
#                     run_j = str(j)    
#                 lh_dir = self.data_dir + 'fmri_sub-'+sub_i+'_ses-movie_task-movie_run-'+run_j+'_fmri_resampled_trimmed_lh.npy'
#                 rh_dir = self.data_dir + 'fmri_sub-'+sub_i+'_ses-movie_task-movie_run-'+run_j+'_fmri_resampled_trimmed_rh.npy'
#                 if os.path.exists(lh_dir) and os.path.exists(rh_dir):  
#                     self.sub_run_dic[str(file_counter)] = [lh_dir , rh_dir]   
#                     self.sub_run_list.append('sub-'+sub_i+'-run-'+run_j) 
#                     file_counter += 1 
#                     if sub_add_list:
#                         sub_add_list = False
#                         self.sub_list.append('sub-'+sub_i)
#                     run_list.append(run_j)
#             if len(run_list) > 0:
#                 self.run_list.append(run_list)
    

#     def meg_sub_dic_list(self):    
#         # iterating through the subjects 
#         n_sub, n_run, n_parts = 20, 20, 20  
#         file_counter = 0
#         for i in range(1, n_sub): 
#             sub_add_list = True
#             if i < 10:
#                 sub_i = '0'+ str(i)
#             else: 
#                 sub_i = str(i)   
#             # iterating through the runs 
#             run_list = []
#             for j in range(1, n_run):
#                 if j < 10:
#                     run_j = '0'+ str(j)
#                 else: 
#                     run_j = str(j)   
#                 # iterating through the parts
#                 for k in range(1, n_parts):
#                     if k < 10:
#                         part_k = '0'+ str(k)
#                     else: 
#                         part_k = str(k)   
#                     mge_dir = self.data_dir + 'meg_sub-'+sub_i+'_ses-movie_task-movie_run-'+run_j+'_trimmed_src_part-'+part_k+'.npy'   
#                     if os.path.exists(mge_dir):    
#                         self.sub_run_dic[str(file_counter)] = [mge_dir]   
#                         self.sub_run_list.append('sub-'+sub_i+'-run-'+run_j+'-part-'+part_k) 
#                         file_counter += 1   
#                         if sub_add_list:
#                             sub_add_list = False
#                             self.sub_list.append('sub-'+sub_i)
#                         run_list.append(run_j)
#                 if len(run_list) > 0:
#                     self.run_list.append(run_list)
 
#     def __getitem__(self, i):
#         """ loads a subject's right and left hippocampus recording. 
#            Args: 
#                 dir_i path of the *.npy dataset to be download 
#             Return:
#                 [xh_i, xh_i] for fmri 
#                 [x_i] for meg
#         """
#         sub_run_index = self.sub_run_list[i]
#         out = []
#         for x in self.sub_run_dic[str(i)]:
#             x = torch.tensor(np.load(x)).to(torch.float16)
#             out.append(x) 
#         return out, sub_run_index 


def meg2min_subdevision(u = 30*400): 
    """Convert MEG's recordings in different rurns/parts to the samples/miniute.   
    Args:
        u: unified_intervals, 30 in fmri means 1 miniute of the movie since each sample of fmri = 2 sec of movie  
            , and each samples of fmri equates to 400 samples in meg
    """     
    data_meg = BrainDataLoader(dataset_mode = 'meg') 
    data_sub_list = data_meg.sub_list
    # data_run_list = data_meg.run_list
    data_run_list = ['01', '02', '03', '04', '05', '06', '07', '08']
    # each run has 1275200 (should be 1272000; and 3200 more)  
    sub_conter, num_s_run, run_conter, num_s, min_counter = 0, 0, 0, 0, 1
    first_run, x_sub_residual = True, None 
    save_path = '/home/aa2793/scratch60/datasets/fmri-meg/meg/samples_240/'
    for i, (x, sub_run_index) in enumerate(data_meg):    
        print(sub_run_index)
        if (sub_run_index[11:13] != data_run_list[run_conter]) or (sub_run_index[:6] != data_sub_list[sub_conter]):
            num_s += num_s_run 
            print('there is '+str(num_s_run)+'/'+str(num_s)+' in meg... for run/all parts of '+sub_run_index[:13]) 
            run_conter +=1 
            if (sub_run_index[:6] != data_sub_list[sub_conter]):
                print('--------------- there is '+str(num_s)+' in meg... for all parts of '+sub_run_index[:13])
                num_s = 0    

                print(x_sub_u.shape, x_sub_u[:, -1, :].shape)
                # remove 1600 samples from end to be devidable to 30*400 
                # (1 min; fMRI recordings are in 2 sec and 400 samples is equal to 1 samples in fmri) 
                x_sub = x_sub[:-1600] 
                
                u_i = int(math.floor((x_sub.shape[0]/u)))
                x_sub_u = x_sub[:u*u_i]
                x_sub_u = x_sub_u.reshape((-1, u_i, d))  
            else:    
                if first_run: x_sub = x_sub[1600:]; first_run = False 
                u_i = int(math.floor((x_sub.shape[0]/u)))
                x_sub_u = x_sub[:u*u_i]
                x_sub_u = x_sub_u.reshape((-1, u_i, d))  
                x_sub_residual = x_sub[u*u_i:]
            
            print(x_sub_u.shape, x_sub_u[:, -1, :].shape)

            for n in range(x_sub_u.shape[1]):  
                # if sub_conter<10:
                path = save_path+data_sub_list[sub_conter]
                # else:
                    # path = save_path+'sub-'+str(sub_conter)
                if not os.path.exists(path):   
                    os.mkdir(path)  
                # import pdb;pdb.set_trace()
                x_n = np.expand_dims(x_sub_u[:, n, :], axis = 0)
                # x_n = torch.tensor(x_n).view(30, 400, -1).mean(1).unsqueeze(0) 
                x_n = torch.tensor(x_n).view(240, 50, -1).mean(1).unsqueeze(0) 
                x_n = torch.nan_to_num(x_n, nan=2.0, posinf=0.0)
                if min_counter<10:
                    np.save(path+'/'+data_sub_list[sub_conter]+'-min-00'+str(min_counter), x_n) 
                elif min_counter>99:
                    np.save(path+'/'+data_sub_list[sub_conter]+'-min-'+str(min_counter), x_n) 
                else:
                    np.save(path+'/'+data_sub_list[sub_conter]+'-min-0'+str(min_counter), x_n) 
                min_counter += 1
            
            if (sub_run_index[:6] != data_sub_list[sub_conter]):
                run_conter = 0
                sub_conter += 1
                x_sub_residual = None 
                del x_sub
                min_counter = 1  
            num_s_run = 0 
            
        x = np.array(x[0]).transpose()  
        s, d = x.shape    
        if num_s_run == 0: 
            if x_sub_residual is not None:
                x_sub = x
                x_sub = np.concatenate((x_sub_residual, x), axis = 0) 
            else:
                x_sub = x
        else:  
            x_sub = np.concatenate((x_sub, x), axis = 0)   
        num_s_run += s


def fmri2min_subdevision(u = 30): 
    """Convert fMRI's recordings in different rurns/parts to the samples/miniute.   
    Args:
        u: unified_intervals, 30 in fmri means 1 miniute of the movie since each sample of fmri = 2 sec of movie  
    """      
    data_fmri = BrainDataLoader(dataset_mode = 'fmri') 
    data_sub_list = data_fmri.sub_list
    print(data_sub_list)

    # # 180200
    save_path = '/home/aa2793/scratch60/datasets/fmri-meg/fmri/samples_30/'
    sub_conter, num_s = 0, 0
    for i, (x, sub_run_index) in enumerate(data_fmri):   
        if sub_run_index[:6] != data_sub_list[sub_conter]:
            print('there is '+str(num_s)+' in fmri, and should be '+ str(num_s*400) + ' samples in meg for '+ data_sub_list[sub_conter])
            sub_conter += 1 
            # remove 5 samples in fMRI to be devidable to 30 (1 min; fMRI recordings are in 2 sec)
            x_sub = x_sub[2:-3] 
            x_sub = x_sub.reshape((-1, u, d)) 
            x_sub = torch.tensor(x_sub)
            # np.save('/home/aa2793/scratch60/datasets/fmri-meg/fmri/subjects/'+sub_run_index[:6], x_sub) 
            for n in range(x_sub.shape[0]): 
                min_counter = n+1
                # import pdb; pdb.set_trace()
                path = save_path+data_sub_list[sub_conter]+'/' 
                if not os.path.exists(path):   
                    os.mkdir(path)   
                if min_counter<10:
                    np.save(path+sub_run_index[:6]+'-min-00'+str(min_counter), 
                            np.expand_dims(x_sub[n, :, :], axis = 0)) 
                elif min_counter>99:
                    np.save(path+sub_run_index[:6]+'-min-'+str(min_counter), 
                            np.expand_dims(x_sub[n, :, :], axis = 0)) 
                else:
                    np.save(path+sub_run_index[:6]+'-min-0'+str(min_counter), 
                            np.expand_dims(x_sub[n, :, :], axis = 0)) 

            num_s = 0 
        x = np.concatenate((x[0], x[1]), axis = 1)
        s, d = x.shape    
        if num_s == 0: 
            x_sub = x
        else:  
            x_sub = np.concatenate((x_sub, x), axis = 0) 
        num_s += s





# meg2min_subdevision()
fmri2min_subdevision()












# def meg2min(u = 30*400): 
#     """Convert MEG's recordings in different rurns/parts to the samples/miniute.   
#     Args:
#         u: unified_intervals, 30 in fmri means 1 miniute of the movie since each sample of fmri = 2 sec of movie  
#             , and each samples of fmri equates to 400 samples in meg
#     """     
#     data_meg = BrainDataLoader(dataset_mode = 'meg') 
#     data_sub_list = data_meg.sub_list
#     # data_run_list = data_meg.run_list
#     data_run_list = ['01', '02', '03', '04', '05', '06', '07', '08']
#     # each run has 1275200 (should be 1272000; and 3200 more)  
#     sub_conter, num_s_run, run_conter, num_s, min_counter = 0, 0, 0, 0, 1
#     first_run, x_sub_residual = True, None 

#     for i, (x, sub_run_index) in enumerate(data_meg):    
#         print(sub_run_index)
#         if (sub_run_index[11:13] != data_run_list[run_conter]) or (sub_run_index[:6] != data_sub_list[sub_conter]):
#             num_s += num_s_run 
#             print('there is '+str(num_s_run)+'/'+str(num_s)+' in meg... for run/all parts of '+sub_run_index[:13]) 
#             run_conter +=1 
#             if (sub_run_index[:6] != data_sub_list[sub_conter]):
#                 print('--------------- there is '+str(num_s)+' in meg... for all parts of '+sub_run_index[:13])
#                 num_s = 0    

#                 print(x_sub_u.shape, x_sub_u[:, -1, :].shape)
#                 # remove 1600 samples from end to be devidable to 30*400 
#                 # (1 min; fMRI recordings are in 2 sec and 400 samples is equal to 1 samples in fmri) 
#                 x_sub = x_sub[:-1600] 
                
#                 u_i = int(math.floor((x_sub.shape[0]/u)))
#                 x_sub_u = x_sub[:u*u_i]
#                 x_sub_u = x_sub_u.reshape((-1, u_i, d))  
#             else:    
#                 if first_run: x_sub = x_sub[1600:]; first_run = False 
#                 u_i = int(math.floor((x_sub.shape[0]/u)))
#                 x_sub_u = x_sub[:u*u_i]
#                 x_sub_u = x_sub_u.reshape((-1, u_i, d))  
#                 x_sub_residual = x_sub[u*u_i:]
            
#             print(x_sub_u.shape, x_sub_u[:, -1, :].shape)

#             for n in range(x_sub_u.shape[1]): 
#                 x_n = np.expand_dims(x_sub_u[:, n, :], axis = 0)
#                 # x_n = torch.tensor(x_n).view(30, 400, -1).mean(1).unsqueeze(0)
#                 x_n = torch.tensor(x_n).view(150, 80, -1).mean(1).unsqueeze(0)
#                 # import pdb;pdb.set_trace()
#                 np.save('/home/aa2793/scratch60/datasets/fmri-meg/meg/samples_150/'+data_sub_list[sub_conter]+'-min-'+str(min_counter),
#                 x_n
#                 ) 
#                 min_counter += 1
            
#             if (sub_run_index[:6] != data_sub_list[sub_conter]):
#                 run_conter = 0
#                 sub_conter += 1
#                 x_sub_residual = None 
#                 del x_sub
#                 min_counter = 1  
#             num_s_run = 0 
            
#         x = np.array(x[0]).transpose()  
#         s, d = x.shape    
#         if num_s_run == 0: 
#             if x_sub_residual is not None:
#                 x_sub = x
#                 x_sub = np.concatenate((x_sub_residual, x), axis = 0) 
#             else:
#                 x_sub = x
#         else:  
#             x_sub = np.concatenate((x_sub, x), axis = 0)   
#         num_s_run += s


# def fmri2min(u = 30): 
#     """Convert fMRI's recordings in different rurns/parts to the samples/miniute.   
#     Args:
#         u: unified_intervals, 30 in fmri means 1 miniute of the movie since each sample of fmri = 2 sec of movie  
#     """      
#     data_fmri = BrainDataLoader(dataset_mode = 'fmri') 
#     data_sub_list = data_fmri.sub_list
#     print(data_sub_list)

#     # # 180200
#     sub_conter, num_s = 0, 0
#     for i, (x, sub_run_index) in enumerate(data_fmri):   
#         if sub_run_index[:6] != data_sub_list[sub_conter]:
#             print('there is '+str(num_s)+' in fmri, and should be '+ str(num_s*400) + ' samples in meg for '+ data_sub_list[sub_conter])
#             sub_conter += 1 
#             # remove 5 samples in fMRI to be devidable to 30 (1 min; fMRI recordings are in 2 sec)
#             x_sub = x_sub[2:-3] 
#             x_sub = x_sub.reshape((-1, u, d)) 
#             np.save('/home/aa2793/scratch60/datasets/fmri-meg/fmri/subjects/'+sub_run_index[:6], x_sub) 
#             for n in range(x_sub.shape[0]): 
#                 np.save('/home/aa2793/scratch60/datasets/fmri-meg/fmri/samples/'+sub_run_index[:6]+'-min-'+str(n+1), 
#                         np.expand_dims(x_sub[n, :, :], axis = 0)) 

#             num_s = 0 
#         x = np.concatenate((x[0], x[1]), axis = 1)
#         s, d = x.shape    
#         if num_s == 0: 
#             x_sub = x
#         else:  
#             x_sub = np.concatenate((x_sub, x), axis = 0) 
#         num_s += s


# fmri2min(u = 30)
# meg2min(u = 30*400)