from data.dataloader02 import NumpyDataset, NumpyMetaDataset
from vit import ViT
import argparse
from argparse import ArgumentParser
import torch
from torch.optim import Adam #, SGD
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DataParallel
# This code is modified from https://github.com/jakesnell/prototypical-networks 
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
 
 


# def data_to_cuda(batch, device):
#     _, x_i, y_i = batch  
#     x_i = [x.to(device) for x in x_i]
#     y_i = [y.to(device) for y in y_i] 
#     return x_i, y_i
 

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



# def get_similiarity_map(proto, query, metric='cosine'):
#     way = proto.shape[0]
#     num_query = query.shape[0]
#     query = query.view(query.shape[0], query.shape[1], -1)
#     proto = proto.view(proto.shape[0], proto.shape[1], -1)

#     proto = proto.unsqueeze(0).repeat([num_query, 1, 1, 1])
#     query = query.unsqueeze(1).repeat([1, way, 1, 1])
#     proto = proto.permute(0, 1, 3, 2)
#     query = query.permute(0, 1, 3, 2)
#     feature_size = proto.shape[-2]

#     if metric == 'cosine':
#         proto = proto.unsqueeze(-3)
#         query = query.unsqueeze(-2)
#         query = query.repeat(1, 1, 1, feature_size, 1)
#         similarity_map = F.cosine_similarity(proto, query, dim=-1)
#     if metric == 'l2':
#         proto = proto.unsqueeze(-3)
#         query = query.unsqueeze(-2)
#         query = query.repeat(1, 1, 1, feature_size, 1)
#         similarity_map = (proto - query).pow(2).sum(-1)
#         similarity_map = 1 - similarity_map
#     return similarity_map


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


# class CLIPModel(nn.Module):
#     def __init__(
#         self,
#         temperature=CFG.temperature,
#         image_embedding=CFG.image_embedding,
#         text_embedding=CFG.text_embedding,
#     ):
#         super().__init__()
#         self.image_encoder = ImageEncoder()
#         self.text_encoder = TextEncoder()
#         self.image_projection = ProjectionHead(embedding_dim=image_embedding)
#         self.text_projection = ProjectionHead(embedding_dim=text_embedding)
#         self.temperature = temperature

#     def forward(self, batch):
#         # Getting Image and Text Features
#         image_features = self.image_encoder(batch["image"])
#         text_features = self.text_encoder(
#             input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
#         )
#         # Getting Image and Text Embeddings (with same dimension)
#         image_embeddings = self.image_projection(image_features)
#         text_embeddings = self.text_projection(text_features)

#         # Calculating the Loss
#         logits = (text_embeddings @ image_embeddings.T) / self.temperature
#         images_similarity = image_embeddings @ image_embeddings.T
#         texts_similarity = text_embeddings @ text_embeddings.T
#         targets = F.softmax(
#             (images_similarity + texts_similarity) / 2 * self.temperature, dim=-1
#         )
#         texts_loss = cross_entropy(logits, targets, reduction='none')
#         images_loss = cross_entropy(logits.T, targets.T, reduction='none')
#         loss =  (images_loss + texts_loss) / 2.0 # shape: (batch_size)
#         return loss.mean()

 
    # def forward(self, xm_i, xf_i, y_i): 

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

        # yj, nquery  = [], 8  
        # for c in range(nquery):  yj.append(y.cpu().numpy().tolist())
        # yj = torch.tensor(yj).view(-1).cuda()
   
        # import pdb; pdb.set_trace()
        acc = (preds==y).cpu().float().mean().item() 
        loss = self.proto_loss_fn(log_p_y, y) 

        return acc, loss*0.1 

    # def proto_loss(self, xm, xf, y):
    #     device = torch.device("cuda:0")
    #     xm = Variable(xm.float()).squeeze(1).to(device)
    #     xf = Variable(xf.float()).squeeze(1).to(device)
    #     y = Variable(y).to(device)

    #     b, s, d = xf.shape  
    #     xm = xm.view(-1, s, d)  

        
    #     prototypes = self.g_mri(self.f_mri(xf))                                                              #[40, 1024] 
    #     queries = self.g_meg(self.f_meg(xm)) #.unsqueeze(1).view(b, -1, prototypes.shape[-1]) 
          
    #     # import pdb; pdb.set_trace()
    #     distances = pairwise_distances(queries, prototypes)
    #     log_p_y = (-distances).log_softmax(dim=1)
    #     y_pred = (-distances).softmax(dim=1)
    #     preds = torch.argmax(y_pred, dim=1) 

    #     yj, nquery  = [], 8  
    #     for c in range(nquery):  yj.append(y.cpu().numpy().tolist())
    #     yj = torch.tensor(yj).view(-1).cuda()

    #     acc = (preds==yj).cpu().float().mean().item() 
    #     loss = self.proto_loss_fn(log_p_y, yj) 

    #     return acc, loss 


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


batch_size = 1
num_workers = 4
meg_dir = '/home/aa2793/scratch60/datasets/fmri-meg/meg/samples_240/train/'
fmri_dir = '/home/aa2793/scratch60/datasets/fmri-meg/fmri/samples_30/train/'  
# train_loader = DataLoader(
#     dataset=NumpyDataset(meg_dir=meg_dir, fmri_dir=fmri_dir),
#     batch_size=batch_size,
#     num_workers=num_workers,
#     shuffle=True,
#     drop_last=True
# ) 

 
def training_fun(args): 
    # optimizer = Adam([{'params': net_f.parameters()},
    #                   {'params': net_m.parameters()}], lr=args.lr)
    # model = nn.DataParallel(CLIPModel().cuda()) # , gpu_ids = [0,1])
    # train_loader = DataLoader(
            #     dataset=datase,
            #     batch_size=batch_size,
            #     num_workers=num_workers,
            #     shuffle=True,
            #     drop_last=True
            # )   

    device = torch.device("cuda:0")
    model = CLIPModel() 
    # model = model().cuda() 
    model.to(device)
    # model = nn.DataParallel(CLIPModel()).cuda() # , gpu_ids = [0,1])
    model = DataParallel(model, device_ids=[0, 1]) 
    optimizer = Adam(model.parameters(), lr=args.lr)  
    result_list = []
 
    for epoch in range(1, args.n_epochs + 1): 
        model.train() 
        tl, ta_f, ta_m = Averager(), Averager(),  Averager()   
        p_l, p_a = Averager(), Averager(),
        # training   
        datase = NumpyMetaDataset(meg_dir = meg_dir, fmri_dir = fmri_dir, n_way = 30, batch_size = batch_size, shuffle=True)
        for i, (xm, xf, y_batch, y_meta) in enumerate(datase):   
             
            y_batch = torch.tensor(y_batch)
            
            xm = torch.stack([torch.from_numpy(arr) for arr in xm]).squeeze(1)
            xf = torch.stack([torch.from_numpy(arr) for arr in xf]).squeeze(1)

            # print(xm.shape) 
            proto_acc, proto_loss = model.module.proto_loss(xm, xf, y_meta)

            clf_loss, acc_f, acc_m = model.module.clf_loss(xm, xf, y_batch) 

            loss  = (proto_acc+clf_loss)/2 
        
            loss.backward()

            for device in range(torch.cuda.device_count()):
                with torch.cuda.device(device):
                    torch.cuda.empty_cache()
            optimizer.step() 
            optimizer.zero_grad()

            tl.add(loss.item())
            ta_f.add(acc_f)   
            ta_m.add(acc_m)   
            p_l.add(proto_loss.item())
            p_a.add(proto_acc) 
 
            if i>0 and i%15 ==0:
                prt_text =  '     ep.%d     l: %4.2f//%4.2f     a: %4.2f/%4.2f//%4.2f || ' 
                print(prt_text % (epoch,  tl.item(), p_l.item(), ta_f.item() * 100, ta_m.item() * 100, p_a.item() * 100)) 
                result_list.append(prt_text % (epoch,  tl.item(), p_l.item(), ta_f.item() * 100, ta_m.item() * 100, p_a.item() * 100)) 
                save_list_to_txt('./mar21-1631-farnam', result_list)
                tl, ta_f, ta_m = Averager(), Averager(),  Averager()

        
        # fmri_meg_data_viz = MiniuteLoader(batch_size = 100)
        # model.eval()
        
        # for i, (xm_viz_i, xf_vz_i, y_i) in enumerate(fmri_meg_data_viz): 
        #     with torch.no_grad():
        #         xf_vz_i, xm_viz_i =  xf_vz_i.cuda(), xm_viz_i.float().cuda()   
        #         xf_vz_i, xm_viz_i = model.viz_forward(xf_vz_i, xm_viz_i)    

        #         if i==0:
        #             xf_vz = xf_vz_i.cpu().numpy()
        #             xm_vz = xm_viz_i.cpu().numpy()
        #             y = y_i
        #         else:
        #             xf_vz = np.concatenate((xf_vz, xf_vz_i.cpu().numpy()), axis=0)
        #             xm_vz = np.concatenate((xm_vz, xm_viz_i.cpu().numpy()), axis=0)
        #             y = torch.cat((y, y_i), dim=0)
        #     if i == 5:
        #         break

        # # import pdb; pdb.set_trace()
        # # loss, acc_f, acc_m, zf_i, zm_i = model(xf_vz, xm_viz, y)  
        # b, d = xf_vz.shape 
        # x = np.concatenate((xf_vz, xm_vz), axis=0)  
        # z_viz = umap.UMAP(n_neighbors=20, min_dist=0.3, metric='correlation').fit_transform(x) 
        # z_f, z_m = z_viz[:b, :], z_viz[b:, :] 

        # plt.figure(figsize=(10, 10))
        # plt.rcParams.update({'font.size': 9})
        # alpha = [0.6, 0.2] 
        # k = 0
        # for c in range(107):
        #     if c>60:
        #         k =1 
        #     # import pdb; pdb.set_trace()
        #     z_f_c = z_f[y==c]  
        #     z_m_c = z_m[y==c] 
        #     if len(z_m[y==c])>0:
        #         plt.scatter(z_f_c[:, 0], z_f_c[:, 1], marker=".", s=70, color=colors[c], alpha=alpha[k])
        #         plt.scatter(z_m_c[:, 0], z_m_c[:, 1], marker="*", s=70, color=colors[c], alpha=alpha[k])
        # plt.savefig('./viz/ep_'+str(epoch)+'_raw_data_col_umap_visualization_classes.pdf')
        # plt.savefig('./viz/ep_'+str(epoch)+'_raw_data_col_umap_visualization_classes.jpg')
        # plt.show
 

def criterion_scoring_fun(args):  
    if args.criterion == 'CrossEntropyLoss':
        criterion_fun = torch.nn.CrossEntropyLoss() 
    if args.criterion == 'BCEWithLogitsLoss':
        criterion_fun = torch.nn.BCEWithLogitsLoss()
    
    if args.scoring == 'LogSoftmax':
        scoring_fun = torch.nn.LogSoftmax(dim=1)  
    elif args.scoring == 'Sigmoid':
        scoring_fun = torch.nn.Sigmoid()
    elif args.scoring == 'Softmax':
        scoring_fun = torch.nn.Softmax()
    return criterion_fun, scoring_fun 

def loss_fun(psi_hat, targets, scoring_fun, criterion_fun):   
    psi_true = targets[0]  
    psi_hat_sig = scoring_fun(psi_hat) #, dim=1)  
    psi_true_bin = (psi_true > 0.5).float().squeeze(1)
    # import pdb; pdb.set_trace()
    psi_true_bin = psi_true_bin.type(torch.LongTensor).to(psi_true_bin.device)
    psi_loss = criterion_fun(psi_hat_sig, psi_true_bin)   
    _, psi_hat_sig = psi_hat_sig.max(1)
    acc = torch.eq(psi_hat_sig, psi_true_bin).float().mean()
    return psi_loss, acc.item() 

 

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

    parser.add_argument("--batch_size", default=64, type=int)
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