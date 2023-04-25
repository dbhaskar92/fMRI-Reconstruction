import numpy as np
import os, sys, glob
from scipy.stats import zscore

MRI_subjects = [1,2,3,4,5,6,9,10,14,15,16,17,18,19,20]
n_runs = 8
n_parts_per_run = {8:7, 7:10, 6:7, 5:8, 4:9, 3:9, 2:9, 1:10}
n_segs_per_run = {1:np.arange(16), 2:np.range(15), 3:np.arange(14), 
 4:np.arange(15), 5:np.arange(12), 6:np.arange(11),
 7: np.arange(16), 8:np.arange(12)}
MEG_subjects = np.arange(2,12)
root_dir = '/gpfs/milgram/project/turk-browne/projects/StudyForrest/fmri_meg_aligned_segments'
MRI_dir = os.path.join(root_dir, 'fmri','fsaverage_trimmed')
MEG_dir = os.path.join(root_dir, 'meg','trimmed_new')
n_vertices = 20484
seconds_per_run = [900, 872, 832, 870, 708, 614, 904, 670]
sampling_hz = {'meg': 200, 'fmri': 0.5}

def _single_sub_meg(sub_id, run, part=None, z=True):
    if part == None:
        data=[]
        for p in range(1, n_parts_per_run[run]+1):
            data.append(_single_sub_meg(sub_id, run, part=p))
        data = np.concatenate(data,axis=0)
    else:
        fn = f'{MEG_dir}/meg_sub-{sub_id:02d}_ses-movie_task-movie_run-{run:02d}_trimmed_src_part-{part:02d}.npy'
        data = np.nan_to_num(np.load(fn))
        if data.shape[1] != n_vertices//2: data = data.T
        if z: data = zscore(data, axis=0)
    return data

def _single_sub_mri(sub_id, run, hemi=None, z=True):
    #fmri_sub-20_ses-movie_task-movie_run-06_fmri_resampled_trimmed_lh.npy
    if hemi == None:
        data = np.concatenate([_single_sub_mri(sub_id, run, H, z) for H in ['lh','rh']], axis=1)
    else:
        fn = f'{MRI_dir}/fmri_sub-{sub_id:02d}_ses-movie_task-movie_run-{run:02d}_fmri_resampled_trimmed_{hemi}.npy'
        data = np.nan_to_num(np.load(fn))
        if data.shape[1] != n_vertices//2: data = data.T
        if z: data = zscore(data, axis=0)
    return data

def load_all_subjects(run, imtype='fmri', hemi=None, part=None, z=True):
    if imtype == 'fmri':
        return np.array([_single_sub_mri(s, run, hemi=hemi, z=z) for s in MRI_subjects])
    else:
        return np.array([_single_sub_meg(s, run, part=part, z=z) for s in MEG_subjects])
    
def _single_sub_chunk_meg(sub_id, run, min_chunk, part=None, z=True):
    samples_per_chunk = sampling_hz['meg'] * 60
    
        


        
        