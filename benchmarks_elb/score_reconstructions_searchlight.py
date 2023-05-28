import numpy as np
import brainplotlib as bpl 
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error
from scipy.stats import pearsonr 


def ret_pearsonr(x,y):
    return pearsonr(x,y)[0]


def score_reconstruction_sls(orig_data, recon_data, searchlights_lh, searchlights_rh, score_fn, vertices_per_hemi=10242):
    '''
    - orig_data and recon_data are the same shape, representing one value per vertex (probably both (20484,) numpy arrays)
    - searchlights_lh and searchlights_rh are lists of the nodes included in searchlights for each hemisphere (so, two lists of length vertices_per_hemi minus the number of vertices in the medial wall, composed of lists of various lengths).
    - score_fn can be any way of scoring the match -- ssim, correlation, MSE -- that returns a single value
    - vertices_per_hemi tells how to divide the data per hemisphere (since searchlights are per-hemisphere)
    
    returns an array of single values for each vertex
    '''
    results_bh = []
    for i, searchlights_hemi in enumerate([searchlights_lh, searchlights_rh]):
        # get the data for each hemisphere, separate
        orig_hemi = orig_data[i*vertices_per_hemi:(i+1)*vertices_per_hemi]
        recon_hemi = recon_data[i*vertices_per_hemi:(i+1)*vertices_per_hemi]
        results = np.zeros_like(orig_hemi)
        # loop through the indices in each searchlight for this hemisphere
        for j, sl_idx in enumerate(searchlights_hemi):
            # pull out the same indices and score
            orig_sl = orig_hemi[sl_idx]
            recon_sl = recon_hemi[sl_idx]
            results[j] = score_fn(orig_sl, recon_sl)
        results_bh.append(results)
    return np.concatenate(results_bh)
        
# load files with searchlight indices (array of arrays of various lengths)
sls_lh, sls_rh = np.load('fsaverage_searchlights_lh_20mm.npy',allow_pickle=True), np.load('fsaverage_searchlights_rh_20mm.npy',allow_pickle=True)
# compute the reconstruction between two identical data (will return 0 for MSE)
recon_map = score_reconstruction_sls(original_data, original_data, sls_lh, sls_rh, mean_squared_error)

# visualize
img,sc = bpl.brain_plot(recon_map, vmin=-1, vmax=1, return_scale=True) 
fig = plt.figure(figsize=(img.shape[1] / 200, img.shape[0] / 200), dpi=50)
plt.axis('off')
plt.title('similarity between reconstruction & original')
cbar = plt.colorbar(sc)
plt.imshow(img)