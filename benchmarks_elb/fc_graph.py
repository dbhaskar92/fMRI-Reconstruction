## create functional connectivity graphs
import numpy as np
import os, sys, glob
import utils
from scipy.stats import zscore
from scipy.spatial.distance import cdist, pdist, squareform


def parcellate_data(data, labels):
    samples, features = data.shape
    n_parcels = len(np.unique(labels))-1 # account for background
    parcellated_data = np.zeros((samples, n_parcels))
    for i in range(n_parcels):
        # get data at that parcel
        parcel_data = data[:, np.where((labels == i+1))[0]]
        parcellated_data[:, i] = np.nanmean(parcel_data,axis=1) # average across voxels
    return parcellated_data

def create_fmri_graph(subject_id, filename, labels):
    all_runs_data = []
    for run in np.arange(1, utils.n_runs+1):
        d = utils._single_sub_mri(subject_id, run, hemi=None, z=True)
        p = parcellate_data(d, labels)
        all_runs_data.append(p)
    pts = np.concatenate(all_runs_data, axis=0)
    graph = 1-squareform(pdist(pts.T, 'correlation'))
    np.save(filename, graph)
    
def create_meg_graph(subject_id, filename, labels, downsample=20):
    all_runs_data = []
    for run in np.arange(1, utils.n_runs+1):
        d = utils._single_sub_meg(subject_id, run, z=True, downsamp=downsample)
        p = parcellate_data(d, labels)
        all_runs_data.append(p)
    pts = np.concatenate(all_runs_data, axis=0)
    graph = 1-squareform(pdist(pts.T, 'correlation'))
    np.save(filename, graph)


if __name__ == "__main__":
    modality = sys.argv[1]
    labels = np.load('Schaefer2018_200Parcels_Kong2022_17Networks_order_labels.npy')
    for sub in utils.subjects[modality]:
        out_dir = f'{utils.root_dir}/{modality}/graphs'
        os.makedirs(out_dir, exist_ok=True)
        outfn = f'{out_dir}/{modality}_sub-{sub:02d}_ses-movie_task-movie_FC_graph_shaefer_parcellation.npy'
        if modality == 'meg':
            create_meg_graph(sub, outfn, labels)
        else:
            create_fmri_graph(sub, outfn, labels)
        print(f'{modality} sub {sub} finished')
        
        
    
