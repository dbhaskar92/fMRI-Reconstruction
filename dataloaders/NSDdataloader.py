import h5py
import pandas as pd
import h5py
import torch
from torch.utils.data import Dataset, DataLoader
from einops import rearrange
from torchvision import transforms
import numpy as np
import tqdm
import os
import re

label_path = '/gpfs/milgram/scratch60/turk-browne/ejl53/NSD/coco_areas.csv'
image_path = '/gpfs/milgram/data/nsd/nsddata_stimuli/stimuli/nsd/nsd_stimuli.hdf5'
fmri_path = '/gpfs/milgram/scratch60/turk-browne/ejl53/NSD/betas_sample-163842/fsaverage/'

class ImageDataset(Dataset):

    def __init__(self, image_path, label_path):
        """
            ImageDataset: builds a dataset over images and labels
            the class builds soft labels of the categories according to the areas
            args:
                image_path: the path to the datase in h5py format
                label_path: the path to the labels in label format
            example:
            dataset = ImageDataset(image_path, label_path)
            transform = transforms.Compose([transforms.ToTensor()])
            dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
            for epoch in range(10):  # 10 epochs
                for i, (x, y) in enumerate(dataloader):
                    import pdb;pdb.set_trace()
                    pass
        """
        # load data and labels
        self.fmri_path = fmri_path
        self.x = h5py.File(image_path, 'r')
        self.labels = pd.read_csv(label_path)
        # take parts of the label
        self.image_index = np.array([x for x in self.labels['imgIndex']])
        self.category = np.array([x for x in self.labels['category']])
        self.area = np.array([x for x in self.labels['area']])
        image_index_set = list(set(self.image_index))
        category_set = list(set(self.category))
        # if doesn't exist, build the soft label vectors
        self.x = self.x['imgBrick']
        if not os.path.isfile(os.path.join('./', 'conditioned_labels.npy')):
            self.y = []
            for i in tqdm.tqdm(image_index_set, desc="creating label set..."):
                y_i = torch.zeros(len(category_set))
                xi_category = self.category[self.image_index==i]
                xi_area = self.area[self.image_index==i]
                xi_area = xi_area/sum(xi_area)
                for c in range(len(xi_category)):
                    y_i[category_set.index(xi_category[c])] += xi_area[c]
                self.y.append(y_i)
            np.save('conditioned_labels.npy', self.y)
        else:
            self.y = np.load('conditioned_labels.npy', allow_pickle=True)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        image = rearrange(self.x[idx], 'w h d -> d w h')
        label = self.y[idx]
        return image, label
    
class ImagefmriDataset(Dataset):

    def __init__(self, fmri_path, image_path, label_path):
        """
            ImagefmriDataset: builds a dataset over fmri, images and labels
            the class builds soft labels of the categories according to the areas
            args:
                image_path: the path to the datase in h5py format
                label_path: the path to the labels in label format
            example:
            dataset = ImagefmriDataset(fmri_path, image_path, label_path)
            transform = transforms.Compose([transforms.ToTensor()])
            dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
            for epoch in range(10):  # 10 epochs
                for i, (x_fmri_lh, x_fmri_rh, x_image, label, [rand_sub, rand_sample_index] ) in enumerate(dataloader):
                    import pdb;pdb.set_trace()
                    pass
        """
        # load data and labels
        self.fmri_path = fmri_path
        self.x = h5py.File(image_path, 'r')
        self.labels = pd.read_csv(label_path)
        # take parts of the label
        self.image_index = np.array([x for x in self.labels['imgIndex']])       # len 515899  (imgIndex, nsdId, cocoId, cocoSplit, category, area)
        self.category = np.array([x for x in self.labels['category']])
        self.area = np.array([x for x in self.labels['area']])
        image_index_set = list(set(self.image_index))
        category_set = list(set(self.category))                                 # we have 80 classes in total
        self.x = self.x['imgBrick']                                             # 73000, 425, 425, 3
        if not os.path.isfile(os.path.join('./', 'conditioned_labels.npy')):
            self.y = []
            for i in tqdm.tqdm(image_index_set, desc="creating label set..."):
                y_i = torch.zeros(len(category_set))
                xi_category = self.category[self.image_index==i]
                xi_area = self.area[self.image_index==i]
                xi_area = xi_area/sum(xi_area)
                for c in range(len(xi_category)):
                    y_i[category_set.index(xi_category[c])] += xi_area[c]
                self.y.append(y_i)
            np.save('conditioned_labels.npy', self.y)
        else:
            self.y = np.load('conditioned_labels.npy', allow_pickle=True)
        self.unique_numbers = []
        self.fmri_subs = [x for x in os.listdir(self.fmri_path)]
        for i in range(len(self.fmri_subs)):
            self.unique_numbers.append(self.sub_unique_index(self.fmri_subs[i]))

    def sub_unique_index(self, sub):
        filenames = [x for x in os.listdir(self.fmri_path+'/'+sub)]
        unique_numbers = []
        for filename in filenames:
            unique_numbers.append(filename[16:-7])
        return unique_numbers
    
    def rand_sub(self):
        rand_fmri_sub_index = np.random.choice(range(len(self.fmri_subs)), 1, replace=False).tolist()[0]
        unique_number = self.unique_numbers[int(rand_fmri_sub_index)]
        rand_sub = self.fmri_subs[int(rand_fmri_sub_index)]
        rand_sample_index = np.random.choice(unique_number, 1, replace=False).tolist()[0]
        return rand_sub, rand_sample_index
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        rand_sub, rand_sample_index = self.rand_sub()
        path = os.path.join(self.fmri_path, rand_sub, rand_sub)+'-betas_nsd'
        path = path + str(rand_sample_index)
        x_fmri_lh = np.load(path+'_lh.npy')
        x_fmri_rh = np.load(path+'_rh.npy')
        x_image = self.x[int(rand_sample_index)-1]         # -1 because images are 0-indexed, but the betas and labels are 1-indexed.
        x_image = rearrange(x_image, 'w h d -> d w h')
        label = self.y[int(rand_sample_index)-1]                             # -1 because images are 0-indexed, but the betas and labels are 1-indexed.
        return x_fmri_lh, x_fmri_rh, x_image, label, [rand_sub, rand_sample_index]
    
class SpecifiedImagefmriDataset(Dataset):

    def __init__(self, fmri_path, image_path, label_path, sub_name):
        """
            SpecifiedImagefmriDataset: builds a dataset over fmri, images and labels based on the spcific subject
            the class builds soft labels of the categories according to the areas
            args:
                fmri_path: the path to the fmri dataset in folder format
                image_path: the path to the image dataset in h5py format
                label_path: the path to the labels in label format
                sub_name: the folder's name with the spacific subject
            example:
            dataset = SpecifiedImagefmriDataset(fmri_path, image_path, label_path, sub_name='sub-01')
            transform = transforms.Compose([transforms.ToTensor()])
            dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
            for epoch in range(10):  # 10 epochs
                for i, (x_fmri_lh, x_fmri_rh, x_image, label, [sub_inx, sample_idx]) in enumerate(dataloader):
                    import pdb;pdb.set_trace()
                    # Put your training code here
                    pass
        """
        # load data and labels
        self.fmri_path = fmri_path
        self.x = h5py.File(image_path, 'r')
        self.labels = pd.read_csv(label_path)
        # take parts of the label
        self.image_index = np.array([x for x in self.labels['imgIndex']])
        self.category = np.array([x for x in self.labels['category']])
        self.area = np.array([x for x in self.labels['area']])
        image_index_set = list(set(self.image_index))
        category_set = list(set(self.category))
        self.x = self.x['imgBrick']
        if not os.path.isfile(os.path.join('./', 'conditioned_labels.npy')):
            self.y = []
            for i in tqdm.tqdm(image_index_set, desc="creating label set..."):
                y_i = torch.zeros(len(category_set))
                xi_category = self.category[self.image_index==i]
                xi_area = self.area[self.image_index==i]
                xi_area = xi_area/sum(xi_area)
                for c in range(len(xi_category)):
                    y_i[category_set.index(xi_category[c])] = xi_area[c]
                self.y.append(y_i)
            np.save('conditioned_labels.npy', self.y)
        else:
            self.y = np.load('conditioned_labels.npy', allow_pickle=True)
        self.fmri_sub = sub_name
        self.sample_index = self.sub_unique_index(self.fmri_sub)

    def sub_unique_index(self, sub):
        unique_numbers = []
        self.filenames = [x for x in os.listdir(self.fmri_path+'/'+sub)]
        for filename in self.filenames:
            unique_numbers.append(filename[16:-7])
        return unique_numbers
    
    def __len__(self):
        return len(self.sample_index)
    
    def __getitem__(self, idx):
        rand_sample_index = np.random.choice(self.sample_index, 1, replace=False).tolist()[0]
        path = os.path.join(self.fmri_path, self.fmri_sub, self.fmri_sub)+'-betas_nsd'
        path = path + str(rand_sample_index)
        x_fmri_lh = np.load(path+'_lh.npy')
        x_fmri_rh = np.load(path+'_rh.npy')
        x_image = self.x[int(rand_sample_index)-1]                           # -1 because images are 0-indexed, but the betas and labels are 1-indexed.
        x_image = rearrange(x_image, 'w h d -> d w h')
        label = self.y[int(rand_sample_index)-1]                             # -1 because images are 0-indexed, but the betas and labels are 1-indexed.
        return x_fmri_lh, x_fmri_rh, x_image, label, [self.fmri_sub, rand_sample_index]