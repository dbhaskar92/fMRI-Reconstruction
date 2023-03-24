from data.dataloader02 import NumpyDataset, NumpyMetaDataset
# from vit import ViT

import argparse
from argparse import ArgumentParser
import torch
from torch.optim import Adam #, SGD
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DataParallel 
import torch
from torch.autograd import Variable
import numpy as np
import torch.nn as nn 
import torch.nn.functional as F
import umap
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.manifold import TSNE
import pandas as pd
import scprep 

colors = []
for key in mcolors.CSS4_COLORS:
    colors.append(key) 

from tqdm import tqdm
from nn.lstmae import DualLSTMAutoencoder
 
class Averager():
    def __init__(self):
        self.n = 0
        self.v = 0

    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    def item(self):
        return self.v 
    
def save_list_to_txt(name, input_list):
    f = open(name, mode='w')
    for item in input_list:
        f.write(item + '\n')
    f.close()
 

class OneLayerClfHead(nn.Module):
    def __init__(
        self,
        embedding_dim,
        projection_dim= 1024, #CFG.projection_dim,
        dropout= 0.1, #CFG.dropout
    ):
        super().__init__()
        self.out = nn.Linear(embedding_dim, 107) 
    
    def forward(self, x): 
        return self.out(x)

class ProjectionHead(nn.Module):
    def __init__(
        self,
        embedding_dim,
        projection_dim= 1024, #CFG.projection_dim,
        dropout= 0.1, #CFG.dropout
    ):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)
    
    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x

def euclidean_dist(x, y):
        # x: N x D
        # y: M x D
        n = x.size(0)
        m = y.size(0)
        d = x.size(1)
        assert d == y.size(1)

        x = x.unsqueeze(1).expand(n, m, d)
        y = y.unsqueeze(0).expand(n, m, d)

        return torch.pow(x - y, 2).sum(2)

def pairwise_distances(x, y):
    """ Calculate l2 distance between each element of x and y.
    Cosine similarity can also be used
    """
    n_x = x.shape[0]
    n_y = y.shape[0]
        
    distances = (
        x.unsqueeze(1).expand(n_x, n_y, -1) -
        y.unsqueeze(0).expand(n_x, n_y, -1)
    ).pow(2).sum(dim=2)
    return distances

class CLIPModel(nn.Module):
    def __init__(
        self,
        temperature=1.0,
        image_embedding= 1024,  #CFG.image_embedding,
        text_embedding=1024, #CFG.text_embedding,
    ):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss()
        self.f_meg =  ViT(
                    image_size = 200,
                    patch_size = 25,
                    num_classes = 1024,
                    dim = 1024,
                    depth = 4, #6,
                    heads = 8, #16,
                    mlp_dim = 2048,
                    dropout = 0.1,
                    emb_dropout = 0.1
                )  

        self.f_mri =  ViT(
                    image_size = 200,
                    patch_size = 25,
                    num_classes = 1024,
                    dim = 1024,
                    depth = 4, #6,
                    heads = 8, #16,
                    mlp_dim = 2048,
                    dropout = 0.1,
                    emb_dropout = 0.1
                )  
 
        self.g_mri = ProjectionHead(embedding_dim=image_embedding)
        self.g_meg = ProjectionHead(embedding_dim=text_embedding)

        self.clf_head = OneLayerClfHead(embedding_dim=text_embedding)

        self.temperature = temperature
        self.log_softmax = nn.LogSoftmax(dim=-1)
        self.proto_loss_fn = torch.nn.NLLLoss()

    def clip_loss(xm_i, xf_i, y_i): 
        b, s, d = xf_i.shape  
        xm_i = xm_i.view(-1, s, d) 

        image_embeddings = self.g_fmri(self.f_meg(xf_i))            #[40, 1024]
        text_embeddings = self.m_meg(xm_i).unsqueeze(1).view(b, -1, image_embeddings.shape[-1]).mean(1) 
        text_embeddings = self.g_meg(text_embeddings)             # [200, 1024] 
         
        # Calculating the Loss
        logits = (text_embeddings @ image_embeddings.T) / self.temperature
        images_similarity = image_embeddings @ image_embeddings.T
        texts_similarity = text_embeddings @ text_embeddings.T
        targets = F.softmax(
            (images_similarity + texts_similarity) / 2 * self.temperature, dim=-1
        )
        texts_loss = clip_cross_entropy(logits, targets, reduction='none')
        images_loss = clip_cross_entropy(logits.T, targets.T, reduction='none')
        loss =  (images_loss + texts_loss) / 2.0 # shape: (batch_size)
        
        acc_f, acc_m = 0, 0
        return loss.mean(), acc_f, acc_m

    def proto_loss(self, xm, xf, y):
        device = torch.device("cuda:0")
        xm = Variable(xm.float()).squeeze(1).to(device)
        xf = Variable(xf.float()).squeeze(1).to(device)
        y = Variable(y).to(device) 

        b, s, d = xf.shape  
        xm = xm.view(-1, s, d)  

        
        queries = self.g_mri(self.f_mri(xf))                                                              #[40, 1024] 
        prototypes = self.g_meg(self.f_meg(xm).unsqueeze(1).view(b, -1, queries.shape[-1]).mean(1))
          
        # 
        distances = pairwise_distances(queries, prototypes)
        log_p_y = (-distances).log_softmax(dim=1)
        y_pred = (-distances).softmax(dim=1)
        preds = torch.argmax(y_pred, dim=1)  
   
        # import pdb; pdb.set_trace()
        acc = (preds==y).cpu().float().mean().item() 
        loss = self.proto_loss_fn(log_p_y, y) 

        return acc, loss*0.1 
 


    def clf_loss(self, xm, xf, y):  
        device = torch.device("cuda:0")
        xm = Variable(xm.float()).squeeze(1).to(device)
        xf = Variable(xf.float()).squeeze(1).to(device)
        y = Variable(y).to(device)  

        b, s, d = xf.shape  
        xm = xm.view(-1, s, d)  

        zf = self.g_mri(self.f_mri(xf))                                                              #[40, 1024] 
        zm = self.g_meg(self.f_meg(xm).unsqueeze(1).view(b, -1, zf.shape[-1]).mean(1))   
        # zm = self.g_meg(self.m_meg(xm).unsqueeze(1).view(b, -1, image_embeddings.shape[-1]).mean(1))             # [200, 1024] 
        logit_f = self.clf_head(zf)
        logit_m = self.clf_head(zm)

        log_softmax = nn.LogSoftmax(dim=-1) 
        loss = self.loss_fn(log_softmax(logit_m), y)/2 + self.loss_fn(log_softmax(logit_f), y)/2


        _, y_hat = logit_f.max(1) 
        acc_f = torch.eq(y_hat, y).float().mean().cpu().numpy()

        _, y_hat = logit_m.max(1) 
        acc_m = torch.eq(y_hat, y).float().mean().cpu().numpy()
 
        return loss*0.9, acc_f, acc_m
    
    def viz_forward(self, xf_i, xm_i): 
        with torch.no_grad():
            b, s, d = xf_i.shape  
            # import pdb; pdb.set_trace()
            xm_i = xm_i.view(-1, s, d)
            xm_i = torch.nan_to_num(xm_i, nan=2.0, posinf=0.0)  
            image_embeddings = self.g_fmri(self.f_meg(xf_i))            #[40, 1024]
            text_embeddings = self.m_meg(xm_i).unsqueeze(1).view(b, -1, image_embeddings.shape[-1]).mean(1) 
            text_embeddings = self.g_meg(text_embeddings)             # [200, 1024] 

        return image_embeddings, text_embeddings
      
def clip_cross_entropy(preds, targets, reduction='none'):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()


# batch_size = 1
# num_workers = 4
# meg_dir = '/home/aa2793/scratch60/datasets/fmri-meg/meg/samples_240/train/'
# fmri_dir = '/home/aa2793/scratch60/datasets/fmri-meg/fmri/samples_30/train/'  
# train_loader = DataLoader(
#     dataset=NumpyDataset(meg_dir=meg_dir, fmri_dir=fmri_dir),
#     batch_size=batch_size,
#     num_workers=num_workers,
#     shuffle=True,
#     drop_last=True
# ) 

 
def training_fun(args):

    device = torch.device("cuda:0")  # 
    input_sequence_length = 240
    output_sequence_length = 30
    input_size = 20484
    hidden_size = 1024
    num_layers = 3

    print("Initialize model")
    
    # model = LSTMAutoencoder(input_size, hidden_size, output_sequence_length, num_layers).cuda() 
    # model = model().cuda() 
    model = DualLSTMAutoencoder(input_size, hidden_size, num_layers, device)
    
    print("Model initialized")
	
    model.to(device) 

    print("Model sent to device")

    # model = DataParallel(model, device_ids=[0, 1]) 
    # optimizer = Adam(model.parameters(), lr=args.lr)
    optimizer = Adam(model.parameters(), lr=0.001)  
    result_list = []

    meg_dir = '/gpfs/gibbs/pi/krishnaswamy_smita/fmri-meg/meg/samples_240/train/'
    fmri_dir = '/gpfs/gibbs/pi/krishnaswamy_smita/fmri-meg/fmri/samples_30/train/'

    dataloader = DataLoader(NumpyDataset(meg_dir = meg_dir, fmri_dir = fmri_dir), 
                        batch_size=30, 
                        num_workers=12, 
                        shuffle=False)
    
    print("Dataloader Done!")

    loss_function = nn.MSELoss()
    
    for epoch in range(1, args.n_epochs + 1): 
        data_embeddings = {}
        model.train() 
        tlf, tlm, tlpn  = Averager(), Averager(), Averager()  
        
        for i, (xm, xf, time, subj) in enumerate(tqdm(dataloader)):   
            
            xf, xm = xf.squeeze(1).float().to(device), xm.float().squeeze(1).to(device)

            [loss_pn, loss_f, loss_m], acc_pn, [xm_hat, xf_hat], [zm_enc, zf_enc] = model(xm, xf)  
            loss = (loss_f + loss_m) # + loss_pn

            # loss = loss_function(xf_hat, xf) 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tlf.add(loss_f.item()) 
            tlm.add(loss_m.item()) 
            tlpn.add(loss_pn) 

            if epoch%1==0:
                data_embeddings[subj] = [time, zm_enc.detach().cpu().numpy(), zf_enc.detach().cpu().numpy()]
 
        prt_text =  '     ep.%d       l(f/m): %4.2f/ %4.2f || ' 
        print(prt_text % (epoch,   tlf.item()*100, tlm.item()*100)) 
        
        result_list.append(prt_text % (epoch,   tlf.item()*100, tlm.item()*100)) 
        save_list_to_txt('./mar23-1344-farnam', result_list)
        tlf, tlm, tlpn  = Averager(), Averager(), Averager()   
        if epoch%1==0:
            np.save(f'./results/data_embeddings_{epoch}.npy', data_embeddings) 

# def criterion_scoring_fun(args):  
#     if args.criterion == 'CrossEntropyLoss':
#         criterion_fun = torch.nn.CrossEntropyLoss() 
#     if args.criterion == 'BCEWithLogitsLoss':
#         criterion_fun = torch.nn.BCEWithLogitsLoss()
    
#     if args.scoring == 'LogSoftmax':
#         scoring_fun = torch.nn.LogSoftmax(dim=1)  
#     elif args.scoring == 'Sigmoid':
#         scoring_fun = torch.nn.Sigmoid()
#     elif args.scoring == 'Softmax':
#         scoring_fun = torch.nn.Softmax()
#     return criterion_fun, scoring_fun 

# def loss_fun(psi_hat, targets, scoring_fun, criterion_fun):   
#     psi_true = targets[0]  
#     psi_hat_sig = scoring_fun(psi_hat) #, dim=1)  
#     psi_true_bin = (psi_true > 0.5).float().squeeze(1)
#     # import pdb; pdb.set_trace()
#     psi_true_bin = psi_true_bin.type(torch.LongTensor).to(psi_true_bin.device)
#     psi_loss = criterion_fun(psi_hat_sig, psi_true_bin)   
#     _, psi_hat_sig = psi_hat_sig.max(1)
#     acc = torch.eq(psi_hat_sig, psi_true_bin).float().mean()
#     return psi_loss, acc.item() 

 

if __name__ == "__main__":

    # tic = time.perf_counter()

    parser = ArgumentParser(add_help=True)

    # data argmuments

    parser.add_argument("--model", default="splicenn_conv2logit_get", type=str)

    parser.add_argument("--input_dim", default=5, type=int)
    parser.add_argument("--latent_dim", default=30, type=int)
    parser.add_argument("--hidden_dim", default=100, type=int)
    parser.add_argument("--embedding_dim", default=100, type=int)

    parser.add_argument("--layers", default=4, type=int)
    parser.add_argument("--nhead", default=4, type=int)
    parser.add_argument("--probs", default=0.2, type=float)

    parser.add_argument("--batch_size", default=5, type=int)
    parser.add_argument("--log_dir", default=None, type=str)
    parser.add_argument("--project_name", default="splicenn-get-valid", type=str)

    # training arguments
    parser.add_argument("--lr", default=2e-5, type=float)  # 2e-5
    parser.add_argument("--n_epochs", default=200, type=int)
    parser.add_argument("--n_steps", default=200000, type=int)
    parser.add_argument("--n_gpus", default=1, type=int)
    # parser.add_argument("--dev", default=False, type=str2bool)
    parser.add_argument("--cpu", default=False, action="store_true")

    parser.add_argument("--max_prime_seq_len", default=10000, type=int)
    parser.add_argument("--max_sj_seq_len", default=1500, type=int)  # 500
    parser.add_argument("--token_size", default=500, type=int)
    parser.add_argument("--accum_grad_batch", default=1, type=int)
    parser.add_argument("--grad_clip_val", default=1, type=float)

    parser.add_argument("--device", default='cuda', type=str)

    # ---------------------------
    # CLI Args
    # ---------------------------
    cl_args = parser.parse_args()
     
    print(torch.cuda.is_available())
    training_fun(cl_args)
