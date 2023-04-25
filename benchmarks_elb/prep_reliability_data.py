import numpy as np
import pandas as pd
import nilearn
import nibabel
import os,sys,glob
from scipy.spatial.distance import cdist, pdist
import utils
from joblib import Parallel, delayed

labels = np.load('Schaefer2018_200Parcels_Kong2022_17Networks_order_labels.npy')
names = np.load('Schaefer2018_200Parcels_Kong2022_17Networks_order_names.npy')
outstr = '/gpfs/milgram/scratch60/turk-browne/elb77/fmri_meg'

def run_jobs(sub, run):
    n_part = utils.n_parts_per_run[run]
    for la, name in enumerate(names):
        name = name.decode()
        if name == 'Background':
            continue
        mask = np.where(labels == la)[0]
        data_per_parts = []
        for part in range(1,n_part+1):
            d=utils._single_sub_meg(sub, run, part, z=True)
            d=d[:,mask]
            data_per_parts.append(d)
        dat = np.concatenate(data_per_parts, axis=0)
        for i, t0 in enumerate(np.arange(0, len(dat), 200*60)): # loop yhrough all 1-min segments
            t1 = t0 + 200*60
            fn = f'{outstr}/meg_sub-{sub:02d}_ses-movie_task-movie_run-{run:02d}_trimmed_src_seg-{i:03d}_region-{name}.npy'
            subdat = dat[t0:t1, :]
            np.save(fn,subdat)
        print(f"Done {sub} {run} {name}")
    print(f"Done all regions for run {run}")
        
        
def build_joblist():
    joblist = []
    for run in range(1,8+1):
        joblist.append(delayed(run_jobs)(sub, run))
    with Parallel(n_jobs=NJOBS) as parallel:
        parallel(joblist)
    print("Done jobs")
    
if __name__ == "__main__":
    NJOBS=16
    sub = int(sys.argv[1])
    build_joblist()