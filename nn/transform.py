import torch
from torch import nn
from torch.autograd import Variable
from einops import rearrange, repeat
from einops.layers.torch import Rearrange


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


# helpers
def pair(t):
    return t if isinstance(t, tuple) else (t, t)


# classes
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots) 
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, input_dim, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        # self.fc = nn.Linear(input_dim, dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
        self.out = nn.Linear(dim, input_dim)
    def forward(self, x):
        # x = self.fc(x)
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.out(x)


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)


class ConvFMT(nn.Module):
    def __init__(self, *, input_dim, dim, depth, heads, mlp_dim, dim_head = 64, dropout = 0., emb_dropout = 0., device = 'cuda'):
        super().__init__() 

        # self.fmri_rearrange = Rearrange('b n (d s) -> b (n d) s', n = rna_dim, s = seq_size)

        self.fmri_seq_length = 30
        self.meg_seq_length = 240

        kernel_num = [2048*3, 2048*2, 2048, dim]
        self.shared = nn.Sequential( 
            nn.Linear(input_dim, kernel_num[0]),
            nn.ReLU(),
            nn.Linear(kernel_num[0], kernel_num[1]),
            nn.ReLU(),
            nn.Linear(kernel_num[1], kernel_num[2]),
            nn.ReLU(),
            nn.Linear(kernel_num[2], kernel_num[3])
        ) 

        
        kernel_num = [64, 128, 256, 240]
        krenel_size = 3
        # self.shared = nn.Linear(input_dim, dim)
        self.fmri2mri_encoder = nn.Sequential(   
            nn.Conv1d(self.fmri_seq_length, kernel_num[0], krenel_size, stride=1, padding=1),
            nn.ReLU(True),
            nn.BatchNorm1d(kernel_num[0]), 
            nn.Conv1d(kernel_num[0], kernel_num[1], krenel_size, stride=1, padding=1),
            nn.ReLU(True),  
            nn.BatchNorm1d(kernel_num[1]),
            nn.Conv1d(kernel_num[1], kernel_num[2], krenel_size, stride=1, padding=1),
            nn.ReLU(True),  
            nn.BatchNorm1d(kernel_num[2]),
            nn.Conv1d(kernel_num[2], kernel_num[3], krenel_size, stride=1, padding=1),
            nn.ReLU(True),  
            nn.BatchNorm1d(kernel_num[3])
        )

        kernel_num = [256, 128, 64, 30]
        self.mri2fmri_encoder = nn.Sequential(   
            nn.Conv1d(self.meg_seq_length, kernel_num[0], krenel_size, stride=1, padding=1),
            nn.ReLU(True),
            nn.BatchNorm1d(kernel_num[0]), 
            nn.Conv1d(kernel_num[0], kernel_num[1], krenel_size, stride=1, padding=1),
            nn.ReLU(True),  
            nn.BatchNorm1d(kernel_num[1]),
            nn.Conv1d(kernel_num[1], kernel_num[2], krenel_size, stride=1, padding=1),
            nn.ReLU(True),  
            nn.BatchNorm1d(kernel_num[2]),
            nn.Conv1d(kernel_num[2], kernel_num[3], krenel_size, stride=1, padding=1),
            nn.ReLU(True),  
            nn.BatchNorm1d(kernel_num[3])
        )


        # self.fmri_transformer = Transformer(input_dim, dim, depth, heads, dim_head, mlp_dim, dropout) 
        self.meg_transformer = Transformer(input_dim, dim, depth, heads, dim_head, mlp_dim, dropout)

        # self.fmri_dropout = nn.Dropout(emb_dropout)
        self.meg_dropout = nn.Dropout(emb_dropout)
        # self.fmri_pos_embedding = nn.Parameter(torch.randn(1, self.fmri_seq_length + 1, input_dim))
        # self.fmri_cls_token = nn.Parameter(torch.randn(1, 1, input_dim))
        


        
        

        
        # # TODO:  change the parameters;  
        # seq_size = 505
        # patch_dim = 505
        # self.num_patches = 40 
        # kernel_num = [10, 20, 30, 10]
        # krenel_size = 3
        # emb_dropout = 0.1  
        
        
        # # self.rearrange = Rearrange('b n (d s) -> b (n d) s', n = rna_dim, s = seq_size)
        # self.rearrange = Rearrange('b k (d s) -> b (k d) s', k = kernel_num[3], s = seq_size)   


        # self.to_patch_embedding = nn.Sequential(
        #     nn.LayerNorm(patch_dim),                                     
        #     nn.Linear(patch_dim, dim), 
        #     nn.LayerNorm(dim)
        #     )  

        # neuron_dim = 15
        # dim = 505
        # self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, dim))
        # self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        # self.dropout = nn.Dropout(emb_dropout)

        

        # self.pool = pool
        # self.to_latent = nn.Identity()
        # self.norm = nn.LayerNorm(dim) 

        # self.mlp_head = nn.Sequential(
        #     nn.LayerNorm(dim),
        #     nn.Linear(dim, 2)  
        # )


        self.device = device
        self.proto_criterion = torch.nn.NLLLoss()  
        self.recon_criterion = nn.MSELoss()

        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)

    def normalize_input(self, x, range_type='[-1,1]'): 
        
        if range_type == '[0,1]':
            # Normalize the tensor to the range [0, 1]
            x_min = torch.min(x)
            x_max = torch.max(x)
            x_norm = (x - x_min) / (x_max - x_min)
        elif range_type == '[-1,1]':
            # Normalize the tensor to the range [-1, 1]
            x_min = torch.min(x)
            x_max = torch.max(x)
            x_norm = ((x - x_min) / (x_max - x_min)) * 2 - 1
        else:
            raise ValueError("Invalid range type. Valid options are '[0,1]' or '[-1,1]'.")

        return x_norm
    
    def recon_loss(self, xm, xf, xm_hat, xf_hat):  
        loss_m = self.recon_criterion(xm_hat, xm)
        loss_f = self.recon_criterion(xf_hat, xf) 
        return loss_m, loss_f


    def proto_loss(self, zs, zq): 
        b, c, d = zs.shape 

        y = Variable(torch.tensor([x for x in range(b)])).to(self.device)    
        # y = torch.tensor([x for x in range(b)]) 
        # y = Variable(y.repeat(c)).to(self.device)  

        # import pdb;pdb.set_trace()
        prototypes = zs.mean(1) #.view(-1, zf.shape[-1])
        queries = zq.mean(1) #zm.view(-1, zm.shape[-1])
        # queries = zm.
        # prototypes = zf.mean(1)
        # queries = zm.unsqueeze(2).view(b, c, -1, d).mean(2).view(-1, d)
 
        distances = pairwise_distances(queries, prototypes)
        log_p_y = (-distances).log_softmax(dim=1)
        y_pred = (-distances).softmax(dim=1)
        preds = torch.argmax(y_pred, dim=1)   

        # import pdb;pdb.set_trace()
        # print(preds)
        acc_pn = (preds==y).cpu().float().mean().item()  
        loss_pn = self.proto_criterion(log_p_y, y)  

        return loss_pn, acc_pn   
          

    def forward(self, xm, xf): # [64, 3, 500] 
    
        xm = self.normalize_input(xm)
        xf = self.normalize_input(xf) 

        b, cm, cf = xm.shape[0], xm.shape[1], xf.shape[1]
        # import pdb;pdb.set_trace()
        xf_hat = self.mri2fmri_encoder(xm) #.view(-1, xm.shape[-1]).unsqueeze(1)).view(b, cm, -1)
        xm_hat = self.fmri2mri_encoder(xf) #.view(-1, xf.shape[-1])).view(b, cf, -1)
        
        zf_hat = self.shared(xf_hat)
        zm_hat = self.shared(xm_hat)
        
        loss_rec2, loss_rec1 = self.recon_loss(zm_hat, zf_hat, self.shared(xm), self.shared(xf))  
        loss_pn = (loss_rec2+loss_rec1)/2 
        # import pdb;pdb.set_trace()
        # loss_pn_f, acc_pn_f = self.proto_loss(zf_hat, self.shared(xf))
        # loss_pn_m, acc_pn_m = self.proto_loss(zm_hat, self.shared(xm))

        # loss_pn = (loss_pn_f + loss_pn_m)/2
        # acc_pn = (acc_pn_f + acc_pn_m)/2

        # loss_pn, acc_pn = self.proto_loss(zf_hat, zm_hat)


        # fmri_cls_tokens = repeat(self.fmri_cls_token, '1 1 d -> b 1 d', b = xf.shape[0])
        # xf = torch.cat((fmri_cls_tokens, xf), dim=1)
        # xf += self.fmri_pos_embedding[:, :(xf.shape[1] + 1)] 
        # xf_hat = self.fmri_transformer(self.fmri_dropout(zf_hat)) 
        xm_hat = self.meg_transformer(self.meg_dropout(zm_hat)) 
        xf_hat = self.meg_transformer(self.meg_dropout(zf_hat)) 
        # import pdb;pdb.set_trace()
        loss_m, loss_f = self.recon_loss(xm, xf, xm_hat, xf_hat)  

        # loss_pn, acc_pn = 0, 0
        acc_pn = 0
  
        return [loss_pn*0.5, loss_m, loss_f], acc_pn, [zm_hat, zf_hat, xm_hat, xf_hat] 


        # xm = self.rearrange(xm) 

        # x_patch_emb = self.to_patch_embedding(x)   
        # b, n, _ = x.shape # [64, 20, 1024] 
        # cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        # x = torch.cat((cls_tokens, x), dim=1)
        # x += self.pos_embedding[:, :(n + 1)]
        # x = self.dropout(x) 
        # x_transformer = self.transformer(x) 
        # x_mean = x_transformer.mean(dim = 1) #if self.pool == 'mean' else x[:, 0] 

        # neuron_type = self.mlp_neron(neuron_type).squeeze(1) 
        # rna = torch.cat((rna, neuron_type), dim=1)  
        # rna = torch.nn.functional.normalize(rna)  
        # rna = rna.unsqueeze(1)
        # rna = self.encoder(rna) 
        # rna = rna.squeeze(1)  
        # x = self.rearrange(rna) 
        # x_patch_emb = self.to_patch_embedding(x)   
        # b, n, _ = x.shape # [64, 20, 1024] 
        # cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        # x = torch.cat((cls_tokens, x), dim=1)
        # x += self.pos_embedding[:, :(n + 1)]
        # x = self.dropout(x) 
        # x_transformer = self.transformer(x) 
        # x_mean = x_transformer.mean(dim = 1) #if self.pool == 'mean' else x[:, 0] 
        # x = self.to_latent(x_mean)
        # return self.mlp_head(x), [x_patch_emb, x_transformer, x_mean] 


# class ConvFMT(nn.Module):
#     def __init__(self, *, dim, depth, heads, mlp_dim, dim_head = 64, dropout = 0., emb_dropout = 0.):
#         super().__init__() 
        
#         # TODO:  change the parameters;  
#         seq_size = 505
#         patch_dim = 505
#         self.num_patches = 40 
#         kernel_num = [10, 20, 30, 10]
#         krenel_size = 3
#         emb_dropout = 0.1  
#         self.encoder = nn.Sequential(   
#             nn.Conv1d(1, kernel_num[0], krenel_size, stride=1, padding=1),
#             nn.ReLU(True),
#             nn.BatchNorm1d(kernel_num[0]), 
#             nn.Conv1d(kernel_num[0], kernel_num[1], krenel_size, stride=1, padding=1),
#             nn.ReLU(True),  
#             nn.BatchNorm1d(kernel_num[1]),
#             nn.Conv1d(kernel_num[1], kernel_num[2], krenel_size, stride=1, padding=1),
#             nn.ReLU(True),  
#             nn.BatchNorm1d(kernel_num[2]),
#             nn.Conv1d(kernel_num[2], kernel_num[3], krenel_size, stride=1, padding=1),
#             nn.ReLU(True),  
#             nn.BatchNorm1d(kernel_num[3]),
#         )
        
#         # self.rearrange = Rearrange('b n (d s) -> b (n d) s', n = rna_dim, s = seq_size)
#         self.rearrange = Rearrange('b k (d s) -> b (k d) s', k = kernel_num[3], s = seq_size)   


#         self.to_patch_embedding = nn.Sequential(
#             nn.LayerNorm(patch_dim),                                     
#             nn.Linear(patch_dim, dim), 
#             nn.LayerNorm(dim)
#             )  

#         neuron_dim = 15
#         dim = 505
#         self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, dim))
#         self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
#         self.dropout = nn.Dropout(emb_dropout)

#         self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

#         self.pool = pool
#         self.to_latent = nn.Identity()
#         self.norm = nn.LayerNorm(dim) 

#         self.mlp_head = nn.Sequential(
#             nn.LayerNorm(dim),
#             nn.Linear(dim, 2)  
#         )


#         init_weights(self.encoder)   
#         init_weights(self.pos_embedding) 
#         init_weights(self.to_patch_embedding)   
#         init_weights(self.cls_token) 
#         init_weights(self.transformer)  
#         init_weights(self.mlp_head)   
#         self.mlp_neron = nn.Embedding(46, neuron_dim)
#         init_weights(self.mlp_neron)
          

#     def forward(self, xm, neuron_type): # [64, 3, 500] 

#         neuron_type = self.mlp_neron(neuron_type).squeeze(1) 
#         rna = torch.cat((rna, neuron_type), dim=1)  
#         rna = torch.nn.functional.normalize(rna)  
#         rna = rna.unsqueeze(1)
#         rna = self.encoder(rna) 
#         rna = rna.squeeze(1)  
#         x = self.rearrange(rna) 
#         x_patch_emb = self.to_patch_embedding(x)   
#         b, n, _ = x.shape # [64, 20, 1024] 
#         cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
#         x = torch.cat((cls_tokens, x), dim=1)
#         x += self.pos_embedding[:, :(n + 1)]
#         x = self.dropout(x) 
#         x_transformer = self.transformer(x) 
#         x_mean = x_transformer.mean(dim = 1) #if self.pool == 'mean' else x[:, 0] 
#         x = self.to_latent(x_mean)
#         return self.mlp_head(x), [x_patch_emb, x_transformer, x_mean] 



    # def forward(self, rna, neuron_type): # [64, 3, 500] 

    #     neuron_type = self.mlp_neron(neuron_type).squeeze(1) 
    #     rna = torch.cat((rna, neuron_type), dim=1)  
    #     rna = torch.nn.functional.normalize(rna)  
    #     rna = rna.unsqueeze(1)
    #     rna = self.encoder(rna) 
    #     rna = rna.squeeze(1)  
    #     x = self.rearrange(rna) 
    #     x_patch_emb = self.to_patch_embedding(x)   
    #     b, n, _ = x.shape # [64, 20, 1024] 
    #     cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
    #     x = torch.cat((cls_tokens, x), dim=1)
    #     x += self.pos_embedding[:, :(n + 1)]
    #     x = self.dropout(x) 
    #     x_transformer = self.transformer(x) 
    #     x_mean = x_transformer.mean(dim = 1) #if self.pool == 'mean' else x[:, 0] 
    #     x = self.to_latent(x_mean)
    #     return self.mlp_head(x), [x_patch_emb, x_transformer, x_mean] 
    #     # import pdb; pdb.set_trace()
 

















        