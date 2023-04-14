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


def normalize_input(x, range_type='[-1,1]'):  
 
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
        # self.out = nn.Linear(dim, input_dim)
    def forward(self, x):
        # x = self.fc(x)
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x #, self.out(x)


class OneDConv(nn.Module):
    def __init__(self, seq_length, kernel_list, krenel_size):
        super().__init__()

        self.layers = nn.Sequential(    
            nn.Conv1d(seq_length, kernel_list[0], krenel_size, stride=1, padding=1),
            nn.BatchNorm1d(kernel_list[0]),
            nn.ReLU(True),
            # nn.MaxPool1d(kernel_size = 3, stride=2),  
            nn.Conv1d(kernel_list[0], kernel_list[1], krenel_size, stride=1, padding=1),
            nn.BatchNorm1d(kernel_list[1]),
            nn.ReLU(True),  
            # nn.MaxPool1d(kernel_size = 3, stride=2),  
            nn.Conv1d(kernel_list[1], kernel_list[2], krenel_size, stride=1, padding=1),
            nn.BatchNorm1d(kernel_list[2]),
            nn.ReLU(True),  
            # nn.MaxPool1d(kernel_size = 3, stride=2),  
            nn.Conv1d(kernel_list[2], kernel_list[3], krenel_size, stride=1, padding=1),
            nn.BatchNorm1d(kernel_list[3]),
            nn.ReLU(True),  
            # nn.MaxPool1d(kernel_size = 3, stride=2), 
        ) 
    def forward(self, x):
        return self.layers(x)


class Spatio1DConv(nn.Module):
    def __init__(self, input_dim, kernel_list, krenel_size):
        super().__init__()

        self.layers = nn.Sequential(   
            Rearrange('b c d -> b d c'),  
            nn.Conv1d(input_dim, kernel_list[0], krenel_size, stride=1, padding=1),
            nn.BatchNorm1d(kernel_list[0]),
            nn.ReLU(True), 
            nn.Conv1d(kernel_list[0], kernel_list[1], krenel_size, stride=1, padding=1),
            nn.BatchNorm1d(kernel_list[1]),
            nn.ReLU(True),  
            nn.Conv1d(kernel_list[1], kernel_list[2], krenel_size, stride=1, padding=1),
            nn.BatchNorm1d(kernel_list[2]),
            nn.ReLU(True),  
            nn.Conv1d(kernel_list[2], kernel_list[3], krenel_size, stride=1, padding=1),
            nn.BatchNorm1d(kernel_list[3]),
            nn.ReLU(True),  
            Rearrange('b d c -> b c d')
        ) 
    def forward(self, x):
        return self.layers(x)


class Spatio1DConvSig(nn.Module):
    def __init__(self, input_dim, kernel_list, krenel_size):
        super().__init__()

        self.layers = nn.Sequential(   
            nn.Linear(input_dim, kernel_list[3]),
            nn.Tanh()
            # Rearrange('b c d -> b d c'),  
            # nn.Conv1d(input_dim, kernel_list[0], krenel_size, stride=1, padding=1),
            # nn.BatchNorm1d(kernel_list[0]),
            # nn.ReLU(True), 
            # nn.Conv1d(kernel_list[0], kernel_list[1], krenel_size, stride=1, padding=1),
            # nn.BatchNorm1d(kernel_list[1]),
            # nn.ReLU(True),  
            # nn.Conv1d(kernel_list[1], kernel_list[2], krenel_size, stride=1, padding=1),
            # nn.BatchNorm1d(kernel_list[2]),
            # nn.ReLU(True),  
            # nn.Conv1d(kernel_list[2], kernel_list[3], krenel_size, stride=1, padding=1),
            # nn.Sigmoid(), 
            # Rearrange('b d c -> b c d')
        ) 
    def forward(self, x):
        return self.layers(x)





class CrossFormer(nn.Module):
    def __init__(self, *, input_dim, dim, depth, heads, mlp_dim, dim_head = 64, dropout = 0., emb_dropout = 0., device = 'cuda'):
        super().__init__()  
        
        self.fmri_seq_length = 30
        self.meg_seq_length = 240 

        self.spatio_encoder = Spatio1DConv(
            input_dim = input_dim, 
            kernel_list = [2048*3, 2048*2, 2048, dim], 
            krenel_size = 3
        )   
 
        self.fmri2mri_encoder = OneDConv(
            self.fmri_seq_length, 
            kernel_list = [64, 128, 256, 240], 
            krenel_size=3
        ) 

        self.mri2fmri_encoder = OneDConv(
            self.meg_seq_length, 
            kernel_list = [256, 128, 64, 30], 
            krenel_size=3
        )  
 
        self.shared_transformer = Transformer(
            input_dim, 
            dim, 
            depth, 
            heads, 
            dim_head, 
            mlp_dim, 
            dropout
        ) 
 
        self.spatio_decoder = Spatio1DConvSig(
            input_dim = dim, 
            kernel_list = [2048, 2048*2, 2048*3, input_dim],  
            krenel_size = 3
        )    
        
        self.dropout = nn.Dropout(emb_dropout)  
        self.device = device
        self.proto_criterion = torch.nn.NLLLoss()  
        self.recon_criterion = nn.MSELoss()

        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)


    def recon_loss(self, xm, xf, xm_hat, xf_hat):  
        loss_m = self.recon_criterion(xm_hat, xm)
        loss_f = self.recon_criterion(xf_hat, xf) 
        return loss_m, loss_f


    def alignment_loss(self, support, querry):   

        b, c, d = querry.shape  
        y = torch.tensor([x for x in range(b)]).to(self.device)   
        # y = Variable(y.repeat(c))  

        prototypes = support.mean(1)
        queries = querry.mean(1)  

        distances = pairwise_distances(queries, prototypes)
        log_p_y = (-distances).log_softmax(dim=1)
        y_pred = (-distances).softmax(dim=1)
        preds = torch.argmax(y_pred, dim=1)   
         
        acc_pn = (preds==y).cpu().float().mean().item()  
        loss_pn = self.proto_criterion(log_p_y, y)  

        return loss_pn, acc_pn 


    def forward_contrastive(self, xm, xf):  
        xm = torch.nn.functional.normalize(xm, p=2.0, dim=-1) 
        xf = torch.nn.functional.normalize(xf, p=2.0, dim=-1)  

        # spatial encoder
        zm_sp = self.spatio_encoder(xm)             # [b, c_m, D] --> [b, c_m, d], where d << D
        zf_sp = self.spatio_encoder(xf)             # [b, c_f, D] --> [b, c_f, d], where d << D       

        # cross convolution meg <--> fmri 
        zf = self.mri2fmri_encoder(zm_sp)           # [b, c_m, d] --> [b, c_f, d], where c_m < c_f
        zm = self.fmri2mri_encoder(zf_sp)           # [b, c_f, d] --> [b, c_m, d], where c_f > c_m

        # alignment loss
        loss_af, acc_af = self.alignment_loss(zf, zf_sp)
        loss_am, acc_am = self.alignment_loss(zm, zm_sp)
        loss_a, acc_a = (loss_af + loss_am)/2, (acc_af + acc_am)/2

        # cross domain reconstructions
        zf_tf = self.shared_transformer(self.dropout(zf)) 
        zm_tf = self.shared_transformer(self.dropout(zm))  
        xf_hat = self.spatio_decoder(zf_tf)
        xm_hat = self.spatio_decoder(zm_tf)

        loss_m, loss_f = self.recon_loss(xm, xf, xm_hat, xf_hat)

        # # in-domain reconstructions
        # zf_sp_tf, xf_sp_hat = self.shared_transformer(self.dropout(zf_sp)) 
        # zm_sp_tf, xm_sp_hat = self.shared_transformer(self.dropout(zm_sp))  
        # loss_sp_m, loss_sp_f = self.recon_loss(xm_sp_hat, xf_sp_hat, xm_hat, xf_hat)
        # loss_m, loss_f = (loss_m + loss_sp_m)/2, (loss_f + loss_sp_f)/2
 
        return [loss_a*0.5, loss_m*0.8, loss_f*0.8], acc_a, [xm, xf, zf, zm, zf_tf, zm_tf, xm_hat, xf_hat] 
