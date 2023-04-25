import numpy as np
import pandas as pd
import os,sys,glob
import utils
from joblib import Parallel, delayed
import brainplotlib as bpl
import matplotlib.pyplot as plt


labels = np.load('Schaefer2018_200Parcels_Kong2022_17Networks_order_labels.npy')
names = np.load('Schaefer2018_200Parcels_Kong2022_17Networks_order_names.npy')
outstr = '/gpfs/milgram/scratch60/turk-browne/elb77/fmri_meg'
NJOBS = 16
plot=True

def run_isc(data):
    results = []
    n_subjects = len(data)
    for test_sub in range(n_subjects):
        train_subjects = np.setdiff1d(np.arange(n_subjects),test_sub)
        train_data = np.squeeze(np.mean(data[train_subjects,:,:],axis=0))
        test_data = np.squeeze(data[test_sub,:,:])
        r=np.corrcoef(train_data.ravel(), test_data.ravel())[0,1]
        results.append(r)
    return results

def run_meg_job(region, run, segment):
    data = []
    for sub in utils.MEG_subjects:
        fn = f'{outstr}/meg_sub-{sub:02d}_ses-movie_task-movie_run-{run:02d}_trimmed_src_seg-{segment:03d}_region-{region}.npy'
        data.append(np.load(fn))
    data = np.array(data)
    # run iscs on this data
    ISCs = run_isc(data)
    temp=pd.DataFrame({"region":region, 'run':run, 'segment':segment, 'isc':ISCs, 'type':'MEG', 'test_sub':utils.MEG_subjects, 'abs':"false", "MVPA":"true"})
    # now run after running absolute values
    ISCs = run_isc(np.abs(data))
    temp1=pd.DataFrame({"region":region, 'run':run, 'segment':segment, 'isc':ISCs, 'type':'MEG', 'test_sub':utils.MEG_subjects, 'abs':"true", "MVPA":"true"})
    print(f' == done run: {run} seg: {segment} region: {region} == ')
    d1 = np.mean(np.abs(data),axis=2)
    d1 = d1[:,:,np.newaxis]
    ISCs = run_isc(d1) # average
    temp2=pd.DataFrame({"region":region, 'run':run, 'segment':segment, 'isc':ISCs, 'type':'MEG', 'test_sub':utils.MEG_subjects, 'abs':"true", "MVPA":"False"})
    return pd.concat([temp, temp1, temp2])

def run_MEG_isc():
    fn = "results/MEG_isc_results.csv"
    if os.path.exists(fn):
        print("Loaded df")
        df = pd.read_csv(fn, index_col=0)
    else:
        joblist=[]
        for name in names:
            name = name.decode()
            if name == 'Background':
                continue
            for run in range(1,utils.n_runs+1):
                sub=3
                NSEGS = len(glob.glob(f'{outstr}/meg_sub-{sub:02d}*run-{run:02d}_*_region-{name}.npy'))
                for seg in range(NSEGS):
                    joblist.append(delayed(run_meg_job)(name, run, seg))
        print(f"Made {len(joblist)} jobs")
        with Parallel(n_jobs=NJOBS) as parallel:
            all_results = parallel(joblist)
        df = pd.concat(all_results)
        df.to_csv(fn)
    if plot:
        print(f"saved to {fn}; making plots")
        make_plots(df)
        

def run_fMRI_job(run):
    results = []
    data = np.nan_to_num(utils.load_all_subjects(run, imtype='fmri', hemi=None, part=None, z=True))
    for la, name in enumerate(names):
        name=name.decode()
        if name == "Background":
            continue
        mask=np.where(labels == la)[0]
        d=np.nan_to_num(data[:, :, mask])
        # run ISC for this run 
        ISCs = run_isc(d)
        temp=pd.DataFrame({"region":name, 'run':run, 'segment':"all_segs", 
                           'isc':ISCs, 'type':'MRI', 
                           'test_sub':utils.MRI_subjects, 'abs':"NA", "MVPA":"true"})
        results.append(temp)
        
        # run over average
        d1 = np.mean(d,axis=2)
        d1 = d1[:,:,np.newaxis]
        ISCs = run_isc(d1) # run over avg
        temp=pd.DataFrame({"region":name, 'run':run, 'segment':"all_segs", 
                           'isc':ISCs, 'type':'MRI', 
                           'test_sub':utils.MRI_subjects, 'abs':"NA", "MVPA":"false"})
        results.append(temp)
        
        # get 60s segments (30TRs) to correspond with MEG
        for i, t0 in enumerate(np.arange(0, d.shape[1], 60 * 0.5)): # loop yhrough all 1-min segments
            t1 = t0 + 60*0.5
            if t1 >= d.shape[1]:
                t1 = d.shape[1]
            subdat = d[:, np.arange(int(t0), int(t1)), :]

            ISCs = run_isc(np.nan_to_num(subdat))
            temp=pd.DataFrame({"region":name, 'run':run, 'segment':i, 
                           'isc':ISCs, 'type':'MRI', 
                           'test_sub':utils.MRI_subjects, 'abs':"NA", "MVPA":"true"})
            results.append(temp)
            
            ISCs = run_isc(np.nan_to_num(d1[:,np.arange(int(t0), int(t1)), :]))
            temp=pd.DataFrame({"region":name, 'run':run, 'segment':i, 
                           'isc':ISCs, 'type':'MRI', 
                           'test_sub':utils.MRI_subjects, 'abs':"NA", "MVPA":"False"})
            results.append(temp)
    return pd.concat(results)
  
def run_fMRI_isc():
    fn = "results/fMRI_isc_results.csv"
    if os.path.exists(fn):
        print('loaded df')
        df = pd.read_csv(fn,index_col=0)
    else:
        joblist=[]
        for run in range(1,utils.n_runs+1):
            joblist.append(delayed(run_fMRI_job)(run))
        print(f"Made {len(joblist)} jobs")
        with Parallel(n_jobs=NJOBS) as parallel:
            all_results = parallel(joblist)
        df = pd.concat(all_results)
        df.to_csv(fn)
    
    if plot:
        print(f"saved to results/fMRI_isc_results.csv; making plots")
        df['abs'] = False
        make_plots(df)
        
def make_brain_array(subset_df):
    on_brain = np.zeros((labels.shape))
    on_brain[:]=np.nan
    for la, name in enumerate(names):
        name = name.decode()
        if name == "Background":
            continue
        mask=np.where(labels == la)[0]
        res = subset_df[(subset_df['region']==name)]['isc'].mean()
        on_brain[mask] = res
    return on_brain
        

def make_plots(results_data, startstr=''):
    datatype=results_data['type'].values[0]
    if datatype == 'MEG':
        maxx = 0.3
    else:
        maxx = 0.8
    # loop through runs
    for mvpa in [True, False]:
        for run in results_data['run'].unique():
            subset_df = results_data[(results_data['MVPA']==mvpa) & (results_data['run']==run)]
            # make for all values of segment
            for seg in subset_df['segment'].unique():
                subdf=subset_df[subset_df['segment']==seg]
                # make for all values of abs
                for ab in subdf['abs'].unique():
                    res = subdf[subdf['abs']==ab]
                    array = make_brain_array(res)
                    title = f"ISC {datatype}, mvpa={mvpa}, run={run}, segment={seg}, absolute_val={ab}"
                    img, sc = bpl.brain_plot(array, vmin=-1*maxx, vmax=maxx, return_scale=True, cmap='seismic') 
                    fig = plt.figure(figsize=(img.shape[1] / 200, img.shape[0] / 200), dpi=50)
                    plt.axis('off')
                    plt.title(title)
                    plt.imshow(img)
                    cbar = plt.colorbar(sc, shrink=0.8, aspect=30)
                    plt.savefig(f'plots/{startstr}{datatype}_mvpa-{mvpa}_run-{run}_segment-{seg}_ab-{ab}.png', bbox_inches='tight')
                    print(title)
                    plt.close('all')




if __name__ == "__main__":
    if sys.argv[1] == 'meg':
        print("Running MEG ISCs!")
        run_MEG_isc()
    else:
        run_fMRI_isc()
    