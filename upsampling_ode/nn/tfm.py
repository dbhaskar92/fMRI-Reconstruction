import torch
import torch.nn as nn
import torch.optim as optim

 
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

  
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()

        self.self_attn = nn.MultiheadAttention(d_model, nhead)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, mask=None):
        # Multi-head self-attention
        src2, _ = self.self_attn(src, src, src, attn_mask=mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # Position-wise feedforward network
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        return src

class TransformerEncoder(nn.Module):
    def __init__(self, input_size, d_model, nhead, num_layers, device):
        super(TransformerEncoder, self).__init__()
        
        
        hidden_size2, hidden_size1 = d_model*3, d_model*2
        self.pre_fc_fmri = nn.Sequential( 
            nn.Linear(input_size, d_model),
            # nn.ReLU(),
            # nn.Linear(hidden_size2, hidden_size1),
            # nn.ReLU(),
            # nn.Linear(hidden_size1, d_model)
        ) 

        self.pre_fc_meg = nn.Sequential( 
            nn.Linear(input_size, d_model),
            # nn.ReLU(),
            # nn.Linear(hidden_size2, hidden_size1),
            # nn.ReLU(),
            # nn.Linear(hidden_size1, d_model)
        ) 
 
        self.fmri_transformer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.meg_transformer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        
        # nn.TransformerEncoder(
        #     nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead),
        #     num_layers=num_layers
        # ) 

        # self.meg_transformer = nn.TransformerEncoder(
        #     nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead),
        #     num_layers=num_layers
        # ) 


        self.shred_layer = nn.Sequential(  
            nn.Linear(d_model, d_model),
            # nn.ReLU(),
            # nn.Linear(hidden_size2, hidden_size1),
            # nn.ReLU(),
            # nn.Linear(hidden_size1, d_model),
            # nn.Sigmoid()
        ) 


        hidden_size2, hidden_size1 = d_model*2, d_model*3
        self.post_fc_fmri = nn.Sequential(
            # nn.LayerNorm(d_model),
            # nn.ReLU(),
            nn.Linear(d_model, input_size),
            # nn.ReLU(),
            # nn.Linear(hidden_size2, hidden_size1),
            # nn.ReLU(),
            # nn.Linear(hidden_size1, input_size),
            # nn.Sigmoid()
        ) 

         
        self.post_fc_meg = nn.Sequential(
            # nn.LayerNorm(d_model),
            # nn.ReLU(),
            nn.Linear(d_model, input_size),
            # nn.ReLU(),
            # nn.Linear(hidden_size2, hidden_size1),
            # nn.ReLU(),
            # nn.Linear(hidden_size1, input_size),
            # nn.Sigmoid()
        ) 

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


    def proto_loss(self, zm, zf): 
        b, c, d = zf.shape 

        # y = Variable(torch.tensor([x for x in range(b)])).to(self.device)    
        y = torch.tensor([x for x in range(b)]) 
        y = Variable(y.repeat(c)).to(self.device)  

        # import pdb;pdb.set_trace()
        prototypes = zf.mean(1) 
        # queries = zm.
        # prototypes = zf.mean(1)
        queries = zm.unsqueeze(2).view(b, c, -1, d).mean(2).view(-1, d)
 
        distances = pairwise_distances(queries, prototypes)
        log_p_y = (-distances).log_softmax(dim=1)
        y_pred = (-distances).softmax(dim=1)
        preds = torch.argmax(y_pred, dim=1)   

        # import pdb;pdb.set_trace()
        # print(preds)
        acc_pn = (preds==y).cpu().float().mean().item()  
        loss_pn = self.proto_criterion(log_p_y, y)  

        return loss_pn, acc_pn   


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
 
    def forward(self, xm, xf): 

        xm = self.normalize_input(xm)
        xf = self.normalize_input(xf)
 
        zm_pre = self.pre_fc_meg(xm) 
        zm = self.meg_transformer(zm_pre)
        zm_shared = self.shred_layer(zm)
        xm_hat = self.post_fc_meg(zm_shared) 
  
        zf_pre  = self.pre_fc_fmri(xf) 
        zf = self.fmri_transformer(zf_pre)
        zf_shared = self.shred_layer(zf)
        xf_hat = self.post_fc_fmri(zf_shared) 

        # import pdb;pdb.set_trace()
        loss_m, loss_f = self.recon_loss(xm, xf, xm_hat, xf_hat)  
        loss_pn, acc_pn = self.proto_loss(zm_shared, zf_shared)

        return [loss_pn, loss_m, loss_f], acc_pn 


       