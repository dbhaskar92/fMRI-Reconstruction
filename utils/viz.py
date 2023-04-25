import sys 
sys.path.append('../')  
import torch  
import matplotlib.pylab as plt
from sklearn.decomposition import PCA
import phate
from nilearn.datasets import fetch_surf_fsaverage
from nilearn import surface
from nilearn import plotting


def brain_map_left_mean(data):  
    data_lh = data[:, :10242].mean(0).detach().numpy() 
    fsaverage_surface = fetch_surf_fsaverage(mesh='fsaverage5') # has 10242 per hemisphere, so 20848 total
    inflated_mesh_lh = fsaverage_surface["infl_left"] # specify hemispheres separately
    sulc_mesh_lh = fsaverage_surface['sulc_left'] 
    plotting.plot_surf_stat_map(inflated_mesh_lh, 
                                data_lh, 
                                hemi='left', 
                                view='lateral',  # {‘lateral’, ‘medial’, ‘dorsal’, ‘ventral’, ‘anterior’, ‘posterior’}
                                bg_map=sulc_mesh_lh) 
#                                 axes=axes1) # visualizes the first sample 

def two_data_time_viz(z_viz, z_viz_hat):
    print("z_viz.shape", z_viz.shape)
    data1 = torch.tensor(z_viz)  
    data2 = torch.tensor(z_viz_hat)

    data1 = data1.reshape(106, -1, 2)  
    data2 = data2.reshape(106, -1, 2)
    fig, axs = plt.subplots(nrows=21, ncols=5, figsize=(12, 8))

    time = 0
    for i in range(21):
        for j in range(5):
            x1 = data1[time, :, :].numpy() 
            x2 = data2[time, :, :].numpy() 
            time += 1
            if j == 0:
                axs[i,j].plot(x1[:, 0], color='red', alpha=0.6, linewidth=1) 
                axs[i,j].plot(x2[:, 0], color='blue', alpha=0.4, linewidth=1) 
            elif j == 1:
                axs[i,j].plot(x1[:, 0], color='red', alpha=0.6, linewidth=1) 
                axs[i,j].plot(x2[:, 0], color='blue', alpha=0.3, linewidth=1)
            else:
                axs[i,j].plot(x1[:, 0], color='red', alpha=0.6, linewidth=1)  
                axs[i,j].plot(x2[:, 0], color='blue', alpha=0.3, linewidth=1)
            axs[i,j].set_xlabel('')
            axs[i,j].set_ylabel('')

            axs[i,j].set_xticks([])
            axs[i,j].set_yticks([])
            axs[i,j].spines['top'].set_alpha(0.2)
            axs[i,j].spines['right'].set_alpha(0.2)
            axs[i,j].spines['bottom'].set_alpha(0.2)
            axs[i,j].spines['left'].set_alpha(0.2) 

    # show the plot
    plt.show()
    

def one_data_time_viz(z_viz, color='red'):
    print("z_viz.shape", z_viz.shape)
    data1 = torch.tensor(z_viz)   

    data1 = data1.reshape(106, -1, 2)   
    fig, axs = plt.subplots(nrows=21, ncols=5, figsize=(12, 8))

    time = 0
    for i in range(21):
        for j in range(5):
            x1 = data1[time, :, :].numpy()  
            time += 1
            if j == 0:
                axs[i,j].plot(x1[:, 0], color=color, alpha=0.6, linewidth=1)  
            elif j == 1:
                axs[i,j].plot(x1[:, 0], color=color, alpha=0.6, linewidth=1)  
            else:
                axs[i,j].plot(x1[:, 0], color=color, alpha=0.6, linewidth=1)   
            axs[i,j].set_xlabel('')
            axs[i,j].set_ylabel('')

            axs[i,j].set_xticks([])
            axs[i,j].set_yticks([])
            axs[i,j].spines['top'].set_alpha(0.2)
            axs[i,j].spines['right'].set_alpha(0.2)
            axs[i,j].spines['bottom'].set_alpha(0.2)
            axs[i,j].spines['left'].set_alpha(0.2) 

    # show the plot
    plt.show()
    
def pca_phate(x, viz_method = 'phate', colors= 'red'):
    plt.figure(figsize=(10, 10))
    plt.rcParams.update({'font.size': 9}) 
    x_viz = x.reshape(-1, x.shape[-1]).numpy()
    if viz_method == 'phate':
        # z_viz = phate.PHATE(2,  k=15, t = 5).fit_transform(x_viz)
        z_viz = phate.PHATE(verbose = False).fit_transform(x_viz)
    elif viz_method == 'pca': 
        z_viz = PCA(2).fit_transform(x_viz) 
    plt.scatter(z_viz[:, 0], z_viz[:, 1], marker="o", s=15, color=colors, alpha=0.3)
    print(z_viz.shape)
    return z_viz

def brain_map_left(data, sample_index):
#     fig, (axes1, axes2, axes3, axes4, axes5) = plt.subplots(5,1, figsize=(10,10))
#     fig, (axes1) = plt.subplots(1,1, figsize=(5, 5))

    data_lh = data[:, :10242].detach().numpy()
    fsaverage_surface = fetch_surf_fsaverage(mesh='fsaverage5') # has 10242 per hemisphere, so 20848 total
    inflated_mesh_lh = fsaverage_surface["infl_left"] # specify hemispheres separately
    sulc_mesh_lh = fsaverage_surface['sulc_left'] 
    plotting.plot_surf_stat_map(inflated_mesh_lh, 
                                data_lh[sample_index], 
                                hemi='left', 
                                view='lateral',  # {‘lateral’, ‘medial’, ‘dorsal’, ‘ventral’, ‘anterior’, ‘posterior’}
                                bg_map=sulc_mesh_lh) 
#                                 axes=axes1) # visualizes the first sample 
def brain_map_right(data, sample_index): 
    data_rh = data[:, 10242:].detach().numpy()
    fsaverage_surface = fetch_surf_fsaverage(mesh='fsaverage5') # has 10242 per hemisphere, so 20848 total
    inflated_mesh_rh = fsaverage_surface["infl_right"] # specify hemispheres separately
    sulc_mesh_rh = fsaverage_surface['sulc_right'] 
    plotting.plot_surf_stat_map(inflated_mesh_lh, 
                                data_rh[sample_index], 
                                hemi='right', 
                                view='lateral', # {‘lateral’, ‘medial’, ‘dorsal’, ‘ventral’, ‘anterior’, ‘posterior’}
                                bg_map=sulc_mesh_lh)
#                                 axes=axes[1]) # visualizes the first sample

















# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.manifold import TSNE
# import os
# import torch 
# import phate

# # Define path to the directory containing the npy file
# data_dir = './results-mar23/'

# # Load the npy file using NumPy
# epoch = 10
# # sub_list = ['sub-06', 'sub-07', 'sub-08', 'sub-09', 'sub-10'] 
# sub_list = ['sub-06', 'sub-07', 'sub-08', 'sub-09', 'sub-10'] 
# epoch = 20
# # {f'./results-mar23/data_embeddings_{epoch}.npy'}
# first_sub = True
# for sub_index in range(5):
#     first_epoch = True
#     for epoch in range(400):
#         if epoch%10 == 0 and epoch>0: 
#             data_embeddings = np.load('./results-mar23/data_embeddings_'+str(epoch)+'.npy', allow_pickle=True)
#             data_embeddings = data_embeddings.item()
#             time_sub = []
#             first_time = True
#             for t ,key in enumerate(data_embeddings):
#                 # print('------------------------------------------------------')
#                 # print(key)
#                 # print('------------------------------------------------------')
#                 [time, zm, zf] = data_embeddings[key]
#                 zf, zm, time = torch.tensor(zf), torch.tensor(zm), torch.tensor(time)
#                 for batch_i, sub_i in enumerate(key): 
#                     if sub_i == sub_list[sub_index]: 
#                         if first_time:
#                             first_time = False
#                             zf_sub = zf[batch_i, :].unsqueeze(0) 
#                             zm_sub = zm[batch_i, :].unsqueeze(0)    
#                             time_sub = time[batch_i].unsqueeze(0)  
#                         else:
#                             zf_sub = torch.cat((zf_sub, zf[batch_i, :].unsqueeze(0) ), dim=0)
#                             zm_sub = torch.cat((zm_sub, zm[batch_i, :].unsqueeze(0) ), dim=0)
#                             time_sub = torch.cat((time_sub, time[batch_i].unsqueeze(0)), dim=0)
#             if first_epoch:
#                 first_epoch = False
#                 zf_sub_all_j = zf_sub.unsqueeze(0) 
#                 zm_sub_all_j = zm_sub.unsqueeze(0) 
#                 time_sub_all_j = time_sub.unsqueeze(0) 
#             else: 
#                 zf_sub_all_j = torch.cat((zf_sub_all_j, zf_sub.unsqueeze(0)), dim=0)
#                 zm_sub_all_j = torch.cat((zm_sub_all_j, zm_sub.unsqueeze(0)), dim=0)
#                 time_sub_all_j = torch.cat((time_sub_all_j, time_sub.unsqueeze(0)), dim=0)
    
#     if first_sub:
#         first_sub = False
#         zf_sub_all = zf_sub_all_j[:, :45, :].unsqueeze(0) 
#         zm_sub_all = zm_sub_all_j[:, :45, :].unsqueeze(0) 
#         time_sub_all = time_sub_all_j[:, :45].unsqueeze(0) 
#     else:  
#         zf_sub_all = torch.cat((zf_sub_all, zf_sub_all_j[:, :45, :].unsqueeze(0)), dim=0)
#         zm_sub_all = torch.cat((zm_sub_all, zm_sub_all_j[:, :45, :].unsqueeze(0)), dim=0)
#         time_sub_all = torch.cat((time_sub_all, time_sub_all_j[:, :45].unsqueeze(0)), dim=0)



# # colors = [    "cyan",    "aquamarine",    "turquoise",    "lightseagreen",    "azure",    "teal",    "steelblue",    "cornflowerblue",    "cadetblue",    "mediumturquoise",    "darkcyan",    "deepskyblue",    "dodgerblue",    "blue",    "royalblue",    "mediumblue",    "navy",    "indigo",    "purple",    "darkorchid",    "mediumorchid",    "fuchsia",    "orchid",    "pink",    "hotpink",    "crimson",    "red",    "orangered",    "darkorange",    "orange",    "coral",    "tomato",    "salmon",    "darksalmon",    "lightcoral",    "maroon",    "brown",    "firebrick",    "sienna",    "chocolate",    "saddlebrown",    "goldenrod",    "peru",    "darkgoldenrod",    "olive",    "olivedrab",    "forestgreen",    "green",    "darkgreen"]
# markers = ['o', '+', '*', '^', 's']
# blues = [    "#F7FBFF",    "#EBF3FB",    "#DEEBF7",    "#C6DBEF",    "#BDD7EE",    "#B3D2E9",    "#9ECae1",    "#9ecae1",    "#6BAED6",    "#4292C6",    "#3182BD",    "#2171B5",    "#08519C",    "#08306B",    "#eff3ff",    "#c6dbef",    "#9ecae1",    "#6baed6",    "#4292c6",    "#2171b5",    "#084594",    "#f1eef6",    "#d0d1e6",    "#a6bddb",    "#74a9cf",    "#3690c0",    "#0570b0",    "#034e7b",    "#f7fcfd",    "#e5f5f9",    "#ccece6",    "#99d8c9",    "#66c2a4",    "#41ae76",    "#238b45",    "#006d2c",    "#00441b",    "#edf8fb",    "#b3cde3",    "#8c96c6",    "#8856a7",    "#810f7c",    "#4d004b",    "#f7fbff",    "#deebf7",    "#c6dbef",    "#9ecae1",    "#6baed6",    "#4292c6",    "#2171b5",    "#08519c"]
# reds = [    "#FFF5F0",    "#FEE0D2",    "#FCBBA1",    "#FC9272",    "#FB6A4A",    "#EF3B2C",    "#CB181D",    "#a50026",    "#fff5f0",    "#fee0d2",    "#fcbba1",    "#fc9272",    "#fb6a4a",    "#ef3b2c",    "#cb181d",    "#a50f15",    "#fff5f0",    "#fee0d2",    "#fcbba1",    "#fc9272",    "#fb6a4a",    "#ef3b2c",    "#cb181d",    "#a50f15",    "#67000d",    "#fff5f0",    "#fee0d2",    "#fcbba1",    "#fc9272",    "#fb6a4a",    "#ef3b2c",    "#cb181d",    "#a50f15",    "#67000d",    "#FFF5F0",    "#FEE0D2",    "#FCBBA1",    "#FC9272",    "#FB6A4A",    "#EF3B2C",    "#CB181D",    "#a50026",    "#8C2D04",    "#FFF5F0",    "#FEE0D2",    "#FCBBA1",    "#FC9272",    "#FB6A4A",    "#EF3B2C",    "#CB181D",    "#a50026",    "#7F0000",    "#7F0000",    "#800000",    "#800000"]
# greens = [    "#F7FCF5",    "#E5F5E0",    "#C7E9C0",    "#A1D99B",    "#74C476",    "#41AB5D",    "#238B45",    "#005A32",    "#00441B",    "#F7FCF5",    "#E5F5E0",    "#C7E9C0",    "#A1D99B",    "#74C476",    "#41AB5D",    "#238B45",    "#006D2C",    "#00441B",    "#F7FCF5",    "#E5F5E0",    "#C7E9C0",    "#A1D99B",    "#74C476",    "#41AB5D",    "#238B45",    "#005A32",    "#00441B",    "#F7FCF0",    "#E0F3DB",    "#CCEBC5",    "#A8DDB5",    "#7BCCC4",    "#4EB3D3",    "#2B8CBE",    "#08589E",    "#084081",    "#F7FCF0",    "#E0F3DB",    "#CCEBC5",    "#A8DDB5",    "#7BCCC4",    "#4EB3D3",    "#2B8CBE",    "#0868AC",    "#084081",    "#F7FCF0",    "#E0F3DB",    "#CCEBC5",    "#A8DDB5",    "#7BCCC4",    "#4EB3D3",    "#2B8CBE",    "#0868AC",    "#084081"]
# blacks = [    "#000000",    "#1A1A1A",    "#333333",    "#4D4D4D",    "#666666",    "#808080",    "#999999",    "#B3B3B3",    "#CCCCCC",    "#E6E6E6",    "#F2F2F2",    "#FFFFFF"]
# colors = [    "#440154",    "#482878",    "#3E4A89",    "#31688E",    "#26838F",    "#1F9E89",    "#35B779",    "#6DCD59",    "#B4DE2C",    "#FDE725",    "#fff5eb",    "#fee6ce",    "#fdd0a2",    "#fdae6b",    "#fd8d3c",    "#f16913",    "#d94801",    "#a63603",    "#7f2704",    "#440154",    "#482878",    "#3E4A89",    "#31688E",    "#26838F",    "#1F9E89",    "#35B779",    "#6DCD59",    "#B4DE2C",    "#FDE725",    "#fff5eb",    "#fee6ce",    "#fdd0a2",    "#fdae6b",    "#fd8d3c",    "#f16913",    "#d94801",    "#a63603",    "#7f2704",    "#440154",    "#482878",    "#3E4A89",    "#31688E",    "#26838F",    "#1F9E89",    "#35B779",    "#6DCD59",    "#B4DE2C",    "#FDE725",]

# # color[blues, reds, greens, ]

# # colors = [
# #     "#007FFF",   # deep blue
# #     "#FFD700",   # bright yellow
# #     "#FF6347",   # vivid red
# #     "#00FF7F",   # bright green
# #     "#9400D3"    # dark purple
# # ]

# # import pdb; pdb.set_trace()
# sub, epoch, time, d = zf_sub_all.shape
# for sub_index in range(sub): 
#     epoch_i_all_subj_i = zf_sub_all[sub_index, :, :, :].reshape(-1, d)
#     coords = phate.PHATE().fit_transform(epoch_i_all_subj_i.numpy())  
#     coords = torch.tensor(coords).unsqueeze(0).view(epoch, time, 2).numpy()
#     fig, ax = plt.subplots(figsize=(8,8)) 
#     epoch_list = [10, 15, 20, 25, 30, 35, 37]
#     for ep in epoch_list:
#         for t in range(time): 
#             coords_t = coords[ep, t, :]
#             plt.scatter(coords_t[0], coords_t[1], marker=markers[0], s=30, color=reds[t], alpha=0.9) 
#         print("ep", ep)
#         plt.savefig('/gpfs/gibbs/pi/krishnaswamy_smita/fmri-meg-results-mar23/sub'+str(sub_index)+'-epoch'+str(ep)+'.png')
        
#     # plt.savefig('./results-mar23/viz/epoch'+str(ep)+'.pdf' )

# # epoch = 0
# # for epoch in range(zf_sub_all.shape[1]):
# #     epoch_i_all_subj = zf_sub_all[:, epoch, :, :]
# #     sub, time, d = epoch_i_all_subj.shape
    
# #     epoch_i_all_subj = epoch_i_all_subj.reshape(-1, d) 
# #     # coords = phate.PHATE(2, k=5,t=50).fit_transform(epoch_i_all_subj.numpy())
# #     coords = phate.PHATE(2, k=2).fit_transform(epoch_i_all_subj.numpy())  
# #     coords = torch.tensor(coords).unsqueeze(0).view(sub, time, 2).numpy()

# #     fig, ax = plt.subplots(figsize=(8,8)) 
# #     for t in range(time):
# #         # import pdb; pdb.set_trace()    
# #         for sub_index in range(sub): 
# #             coords_ = coords[sub_index, t, :]
# #             plt.scatter(coords_[0], coords_[1], marker=markers[sub_index], s=50, color=colors[t], alpha=0.7) 
# #     plt.savefig('./results-mar23/viz/epoch'+str(epoch)+'.png' )
# #     plt.savefig('./results-mar23/viz/epoch'+str(epoch)+'.pdf' )

# import cv2
# import os

# # Path to the folder containing the plots
# path = "./results-mar23/viz/"

# # Get a list of all the image files in the folder
# # files = [os.path.join(path, file) for file in os.listdir(path) if file.endswith(".png")]

# files = ['./results-mar23/viz/epoch'+str(x)+'.png' for x in range(39)]
# # import pdb; pdb.set_trace()    

# # Set video parameters
# fps = 1
# delay = int(1000 / fps)  # Delay in milliseconds
# fourcc = cv2.VideoWriter_fourcc(*"mp4v")
# size = cv2.imread(files[0]).shape[:2]
# output_path = "output_video.mp4"

# # Create the video writer object
# out = cv2.VideoWriter(output_path, fourcc, fps, size)

# # Loop through the image files and add them to the video
# for i, file in enumerate(files):
#     # Read the image file
#     img = cv2.imread(file)
#     # Add the epoch index to the image
#     text = f"Epoch: {i}"
#     cv2.putText(img, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
#     # Write the image to the video
#     out.write(img)
#     # Wait for the specified delay
#     cv2.waitKey(delay)

# # Release the video writer object and close the file
# out.release()

# print(f"Video saved to {output_path}")




 