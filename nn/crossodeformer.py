import torch
from torch import nn
from torch.autograd import Variable
from einops import rearrange, repeat
from einops.layers.torch import Rearrange 
from torchdiffeq import odeint 
from torch.distributions.normal import Normal
from torch.distributions import kl_divergence, Independent
import pdb
import torch
import torch.nn as nn
from torchdiffeq import odeint

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
        # self.out = nn.Linear(dim, input_dim)
    def forward(self, x):
        # x = self.fc(x)
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x #, self.out(x)


class OneDConvTemporal(nn.Module):
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
    def __init__(self, input_dim, kernel_list, kernel_size):
        super().__init__()

        self.layers = nn.Sequential(   
            nn.Linear(input_dim, kernel_list[3]),
            nn.Tanh()
            # Rearrange('b c d -> b d c'),  
            # nn.Conv1d(input_dim, kernel_list[0], kernel_size, stride=1, padding=1),
            # nn.BatchNorm1d(kernel_list[0]),
            # nn.ReLU(True), 
            # nn.Conv1d(kernel_list[0], kernel_list[1], kernel_size, stride=1, padding=1),
            # nn.BatchNorm1d(kernel_list[1]),
            # nn.ReLU(True),  
            # nn.Conv1d(kernel_list[1], kernel_list[2], kernel_size, stride=1, padding=1),
            # nn.BatchNorm1d(kernel_list[2]),
            # nn.ReLU(True),  
            # nn.Conv1d(kernel_list[2], kernel_list[3], kernel_size, stride=1, padding=1),
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

        param = {     
            'd': dim,              
            'fmri_seq_length': self.fmri_seq_length,                      
            'meg_seq_length': self.meg_seq_length,                  
            'LO_hidden_size': dim,  
            'OF_layer_dim': 100, 
            'gru_n_layers': 2, 
            'rtol': 1e-3,
            'atol': 1e-4, 
            'device': device, 
        } 

        self.upsamp = UpSample(param)
        self.downsamp = DownSample(param)
 
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
            kernel_size = 3
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


    def alignment_loss_contrastive(self, support, querry):   
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
        zm_sp = self.spatio_encoder(xm)             # [b, T_m, D] --> [b, T_m, d], where d << D
        zf_sp = self.spatio_encoder(xf)             # [b, T_f, D] --> [b, T_f, d], where d << D       

        # cross convolution meg <--> fmri 
        (zf_hidden, zf_hidden_interped, zm) = self.upsamp(zf_sp)     # [b, T_f, d] --> [b, T_m, d], where T_f > T_m
        (zm_hidden, zf) = self.downsamp(zm_sp)                       # [b, T_m, d] --> [b, T_f, d], where T_m < T_f
        
        # zf_hidden_interped: [240, 2, 1024]         zm_hidden: [2, 240, 1024]  
        # loss_a = torch.nn.MSELoss(zf_hidden_interped, rearrange(zm_hidden, 'l c d -> c l d')) 

        # alignment loss
        loss_af, acc_af = self.alignment_loss_contrastive(zf, zf_sp)
        loss_am, acc_am = self.alignment_loss_contrastive(zm, zm_sp)
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
 
        return [loss_a*0.25, loss_m*0.75, loss_f*0.75], acc_a, [xm, xf, zf, zm, zf_tf, zm_tf, xm_hat, xf_hat] 


class OR_ODE_Func(nn.Module):
	
	def __init__(self, param):
		super(OR_ODE_Func, self).__init__()
		self.param = param
		self.hidden_layer = nn.Linear(self.param['LO_hidden_size'], self.param['OF_layer_dim'])
		self.tanh = nn.Tanh()
		self.hidden_layer2 = nn.Linear(self.param['OF_layer_dim'], self.param['OF_layer_dim'])
		self.tanh2 = nn.Tanh()
		self.output_layer = nn.Linear(self.param['OF_layer_dim'], self.param['LO_hidden_size'])
		
	def forward(self, t, input):
		x = input
		x = self.hidden_layer(x)
		x = self.tanh(x)
		x = self.hidden_layer2(x)
		x = self.tanh2(x)
		x = self.output_layer(x)	
		return x

class Regressor(nn.Module):
	
	def __init__(self, param):
		super(Regressor, self).__init__()
		self.param = param
		self.hidden_layer = nn.Linear(self.param['LO_hidden_size'], self.param['OF_layer_dim'])
		self.tanh = nn.Tanh()
		self.hidden_layer2 = nn.Linear(self.param['OF_layer_dim'], self.param['OF_layer_dim'])
		self.tanh2 = nn.Tanh()
		self.output_layer = nn.Linear(self.param['OF_layer_dim'], self.param['LO_hidden_size'])
		
	def forward(self, t, input):
		x = input
		x = self.hidden_layer(x)
		x = self.tanh(x)
		x = self.hidden_layer2(x)
		x = self.tanh2(x)
		x = self.output_layer(x)	
		return x


class UpSample(nn.Module):
	
    def __init__(self, param):
        super(UpSample, self).__init__()
        
        self.param = param
        self.ode_func = OR_ODE_Func(param)
        
        self.encoder_gru = torch.nn.GRU(
            input_size = self.param['d'],
            hidden_size = self.param['LO_hidden_size'], 
            num_layers = self.param['gru_n_layers'], 
            batch_first = True
        )
        
        self.decoder_gru = torch.nn.GRU(
            input_size = self.param['d'],
            hidden_size = self.param['LO_hidden_size'], 
            num_layers = self.param['gru_n_layers'], 
            batch_first = True
        )
        self.input_seq_len = 30
        self.output_seq_len = 240
        self.sampling_rate = 1/8
        
        self.h0 = torch.zeros(self.param['gru_n_layers'], self.param['LO_hidden_size'], device = self.param['device'])
 

    def forward(self, x):  
        
        # encode the input sequence 
        gru_cellstate_t, first_time = self.h0, True 
        for t in range(self.input_seq_len):
            gru_x, gru_cellstate_t = self.encoder_gru(x[:, t, :], gru_cellstate_t)
            
            if first_time:
                first_time = False
                encoder_hidden_states = gru_cellstate_t.unsqueeze(1)
            else:
                 encoder_hidden_states = torch.cat((encoder_hidden_states, gru_cellstate_t.unsqueeze(1)), dim = 1)

        assert  encoder_hidden_states.shape[1] == self.input_seq_len

        # interpolate with neural-ode
        first_time = True
        for t in range(self.input_seq_len): 
            if t == (self.input_seq_len-1):
                bv1, bv2 = encoder_hidden_states[:, t, :], 0.0
            else:
                bv1, bv2 = encoder_hidden_states[:, t, :], encoder_hidden_states[:, t+1, :] 
            h_interp_t = odeint(
                self.ode_func,  
                bv1,            # initial value for the ode problem
                Variable(torch.arange(t, t+1, self.sampling_rate).float()).to(self.param['device']),   
                rtol = self.param['rtol'], 
                atol = self.param['atol']
            ) 
            tau = (t - 0)/self.input_seq_len
            h_interp_reparam_t = ((1-tau)*bv1) + tau*bv2 + (1- torch.exp((torch.tensor(1.0)-tau)*tau))*h_interp_t
             
            if first_time:
                first_time = False
                interped_hidden_states = h_interp_reparam_t 
            else:
                interped_hidden_states = torch.cat((interped_hidden_states, h_interp_reparam_t), dim = 0)

        assert interped_hidden_states.shape[0] == self.output_seq_len
        
        # decode the output sequence
        gru_cellstate_t, first_time = self.h0, True 
        for t in range(self.output_seq_len): 
            gru_y_t, gru_cellstate_t = self.decoder_gru(x[:, int(t*self.sampling_rate), :], gru_cellstate_t)

            if first_time:
                first_time = False
                interped_seq = gru_y_t.unsqueeze(1)
            else:
                interped_seq = torch.cat((interped_seq, gru_y_t.unsqueeze(1)), dim = 1)
     
        return (encoder_hidden_states, interped_hidden_states, interped_seq)


class DownSample(nn.Module):
	
    def __init__(self, param):
        super(DownSample, self).__init__()
        
        self.param = param
        self.reg = Regressor(param)
        
        self.encoder_gru = torch.nn.GRU(
            input_size = self.param['d'],
            hidden_size = self.param['LO_hidden_size'], 
            num_layers = self.param['gru_n_layers'], 
            batch_first = True
        )
        
        self.decoder_gru = torch.nn.GRU(
            input_size = self.param['d'],
            hidden_size = self.param['LO_hidden_size'], 
            num_layers = self.param['gru_n_layers'], 
            batch_first = True
        )

        self.input_seq_len = 240
        self.output_seq_len = 30
        self.sampling_rate = 1/8

        self.h0 = torch.zeros(self.param['gru_n_layers'], self.param['LO_hidden_size'], device = self.param['device'])
 

    def forward(self, x):  

        # encode the input sequence 
        gru_cellstate_t, first_time = self.h0, True 
        for t in range(self.input_seq_len):
            gru_x, gru_cellstate_t = self.encoder_gru(x[:, t, :], gru_cellstate_t)
            
            if first_time:
                first_time = False
                encoder_hidden_states = gru_cellstate_t.unsqueeze(1)
            else:
                 encoder_hidden_states = torch.cat((encoder_hidden_states, gru_cellstate_t.unsqueeze(1)), dim = 1)

        assert  encoder_hidden_states.shape[1] == self.input_seq_len
        
        # decode the output sequence
        gru_cellstate_t, first_time = self.h0, True 
        for t in range(self.output_seq_len): 
            gru_y_t, gru_cellstate_t = self.decoder_gru(x[:, int(t*(1.0/self.sampling_rate)), :], gru_cellstate_t)

            if first_time:
                first_time = False
                interped_seq = gru_y_t.unsqueeze(1)
            else:
                interped_seq = torch.cat((interped_seq, gru_y_t.unsqueeze(1)), dim = 1)
        
        return (encoder_hidden_states, interped_seq)