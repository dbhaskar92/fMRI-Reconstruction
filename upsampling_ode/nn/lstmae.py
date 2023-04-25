import torch
import random
import torch.nn as nn
from torch.autograd import Variable
   

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



class Encoder(nn.Module):
    def __init__(self, num_layers, seq_len, no_features, embedding_size):
        super().__init__()
        
        self.seq_len = seq_len
        self.no_features = no_features    # The number of expected features(= dimension size) in the input x
        self.embedding_size = embedding_size   # the number of features in the embedded points of the inputs' number of features
        self.hidden_size = (2 * embedding_size)  # The number of features in the hidden state h
        self.LSTM1 = nn.LSTM(
            input_size = no_features,
            hidden_size = embedding_size,
            num_layers = num_layers,
            batch_first=True
        )
        
    def forward(self, x):
        # Inputs: input, (h_0, c_0). -> If (h_0, c_0) is not provided, both h_0 and c_0 default to zero.
        x, (hidden_state, cell_state) = self.LSTM1(x)  
        last_lstm_layer_hidden_state = hidden_state[-1,:,:]
        return last_lstm_layer_hidden_state, x
    
    
# (2) Decoder
class Decoder(nn.Module):
    def __init__(self, num_layers, seq_len, no_features, output_size):
        super().__init__()

        self.seq_len = seq_len
        self.no_features = no_features
        self.hidden_size = (2 * no_features)
        self.output_size = output_size
        self.LSTM1 = nn.LSTM(
            input_size = no_features,
            hidden_size = self.hidden_size,
            num_layers = num_layers,
            batch_first = True
        )

        self.fc = nn.Linear(self.hidden_size, output_size)
        
    def forward(self, x):
        # import pdb;pdb.set_trace()
        x = x.unsqueeze(1).repeat(1, self.seq_len, 1)
        x, (hidden_state, cell_state) = self.LSTM1(x)
        x = x.reshape((-1, self.seq_len, self.hidden_size))
        # return x
        # import pdb;pdb.set_trace()
        out = self.fc(x)
        return out




class LSTMAutoencoder(nn.Module):
    def __init__(self, fmri_seq_len, meg_seq_len, input_size, embedding_size, num_layers, device):
        super(LSTMAutoencoder, self).__init__()

        self.no_features = 2084
        hidden_size2, hidden_size1 = self.no_features*3, self.no_features*2
        self.pre_fc_fmri = nn.Sequential( 
            nn.Linear(input_size, hidden_size2),
            nn.ReLU(),
            nn.Linear(hidden_size2, hidden_size1),
            nn.ReLU(),
            nn.Linear(hidden_size1, self.no_features)
        ) 

        self.pre_fc_meg = nn.Sequential( 
            nn.Linear(input_size, hidden_size2),
            nn.ReLU(),
            nn.Linear(hidden_size2, hidden_size1),
            nn.ReLU(),
            nn.Linear(hidden_size1, self.no_features)
        ) 

        
        self.fmri_encoder = Encoder(num_layers, seq_len = fmri_seq_len, no_features=self.no_features, embedding_size=embedding_size) 
        self.meg_encoder = Encoder(num_layers, seq_len = meg_seq_len, no_features=self.no_features, embedding_size=embedding_size)  

        hidden_size1, hidden_size2,  alignment_dim = 1024, 1024, 512
        self.shared_encoder = nn.Sequential(
            nn.Linear(self.no_features, hidden_size1),
            nn.ReLU(),
            nn.Linear(hidden_size1, hidden_size2),
            nn.ReLU(),
            nn.Linear(hidden_size2, alignment_dim)
        ) 

        no_features = 1024
        self.shared_decoder = nn.Sequential(
            nn.Linear(alignment_dim, hidden_size2),
            nn.ReLU(),
            nn.Linear(hidden_size2, hidden_size1),
            nn.ReLU(),
            nn.Linear(hidden_size1, no_features)
        ) 
        
        self.fmri_decoder = Decoder(num_layers, seq_len = fmri_seq_len, no_features=embedding_size, output_size=no_features)  
        self.meg_decoder = Decoder(num_layers, seq_len = meg_seq_len, no_features=embedding_size, output_size=no_features)  

        self.device = device
        self.proto_criterion = torch.nn.NLLLoss()  
        self.recon_criterion = nn.MSELoss()

 
        hidden_size2, hidden_size1 = no_features*2, no_features*3
        self.post_fc_fmri = nn.Sequential(
            # nn.LayerNorm(d_model),
            # nn.ReLU(),
            nn.Linear(no_features, hidden_size2),
            nn.ReLU(),
            nn.Linear(hidden_size2, hidden_size1),
            nn.ReLU(),
            nn.Linear(hidden_size1, input_size),
            nn.Sigmoid()
        ) 

         
        self.post_fc_meg = nn.Sequential(
            # nn.LayerNorm(d_model),
            # nn.ReLU(),
            nn.Linear(no_features, hidden_size2),
            nn.ReLU(),
            nn.Linear(hidden_size2, hidden_size1),
            nn.ReLU(),
            nn.Linear(hidden_size1, input_size),
            nn.Sigmoid()
        ) 

    
    def normalize_input(self, x, range_type='[-1,1]'):
        """
        Normalize the input tensor x to the range [0, 1] or [-1, 1].

        Args:
            x (torch.Tensor): Input tensor to be normalized.
            range_type (str): Range of the normalized tensor. Valid options are '[0,1]' or '[-1,1]'.

        Returns:
            Normalized tensor.
        """

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

   
    def proto_loss(self, zm, zf): 
        # zm, zf = zm.unsqueeze(0), zf.unsqueeze(0)
        # b, c, d = zf.shape 
        
        # zm = zm.unsqueeze(2).view(b, c, -1, d).mean(2)
        c, d = zf.shape 
        y = Variable(torch.tensor([x for x in range(c)])).to(self.device)  
        # y = Variable(y.repeat(c)).to(self.device) 

        queries = zf.view(-1, d) 
        prototypes = zm.view(-1, d)  
        
        distances = pairwise_distances(queries, prototypes)
        log_p_y = (-distances).log_softmax(dim=1)
        y_pred = (-distances).softmax(dim=1)
        preds = torch.argmax(y_pred, dim=1)  
     
        acc_pn = (preds==y).cpu().float().mean().item()  
        loss_pn = self.proto_criterion(log_p_y, y)  

        return loss_pn, acc_pn 


    def recon_loss(self, xm, xf, xm_hat, xf_hat):  
        loss_m = self.recon_criterion(xm_hat, xm)
        loss_f = self.recon_criterion(xf_hat, xf) 
        return loss_m, loss_f 


    def forward(self, xm, xf):  

        xm_norm = self.normalize_input(xm)
        xf_norm = self.normalize_input(xf) 

        zm_norm = self.pre_fc_meg(xm_norm) 
        zf_norm = self.pre_fc_fmri(xf_norm) 
 
        # encoder 
        hm, zm = self.meg_encoder(zm_norm)
        hf, zf = self.fmri_encoder(zf_norm) 

        # # shared space
        # import pdb;pdb.set_trace()
        # z_hm = self.shared_encoder(hm)               # 30, 512
        # z_hf = self.shared_encoder(hf)               # 30, 512
 
        # hm_hat = self.shared_decoder(z_hm)
        # hf_hat = self.shared_decoder(z_hf)

        # decoder
        zm_hat = self.meg_decoder(hm)
        zf_hat = self.fmri_decoder(hf)

        # import pdb;pdb.set_trace()

        xm_hat = self.post_fc_meg(zm_hat) 
        xf_hat = self.post_fc_fmri(zf_hat) 

        
        loss_m, loss_f = self.recon_loss(xm, xf, xm_hat, xf_hat)
        loss_pn, acc_pn = 0, 0 # self.proto_loss(z_hm, z_hf)
          
        return [loss_pn, loss_m, loss_f], acc_pn, [hm, hf], [xm_hat, xf_hat]


    # def proto_loss(self, zm, zf): 
    #     # zm, zf = zm.unsqueeze(0), zf.unsqueeze(0)
    #     # b, c, d = zf.shape 
        
    #     # zm = zm.unsqueeze(2).view(b, c, -1, d).mean(2)
    #     c, d = zf.shape 
    #     y = Variable(torch.tensor([x for x in range(c)])).to(self.device)  
    #     # y = Variable(y.repeat(c)).to(self.device) 

    #     queries = zf.view(-1, d) 
    #     prototypes = zm.view(-1, d)  
        
    #     distances = pairwise_distances(queries, prototypes)
    #     log_p_y = (-distances).log_softmax(dim=1)
    #     y_pred = (-distances).softmax(dim=1)
    #     preds = torch.argmax(y_pred, dim=1)  
     
    #     acc_pn = (preds==y).cpu().float().mean().item()  
    #     loss_pn = self.proto_criterion(log_p_y, y)  

    #     return loss_pn, acc_pn  



     





class LSTMAutoencoder2(nn.Module):
    def __init__(self, fmri_seq_len, meg_seq_len, input_size, embedding_size, num_layers, device):
        super(LSTMAutoencoder2, self).__init__()
        
        
        self.fmri_encoder = Encoder(num_layers, seq_len = fmri_seq_len, no_features=input_size, embedding_size=embedding_size) 
        self.meg_encoder = Encoder(num_layers, seq_len = meg_seq_len, no_features=input_size, embedding_size=embedding_size)  
        
        self.fmri_decoder = Decoder(num_layers, seq_len = fmri_seq_len, no_features=embedding_size, output_size=input_size)  
        self.meg_decoder = Decoder(num_layers, seq_len = meg_seq_len, no_features=embedding_size, output_size=input_size)  

        self.device = device
        self.proto_criterion = torch.nn.NLLLoss()  
        self.recon_criterion = nn.MSELoss()

        self.fmri_out = nn.Linear(embedding_size*2, input_size)
        self.meg_out = nn.Linear(embedding_size*2, input_size) 
 
    
    # def proto_loss(self, zm, zf): 
    #     b, c, d = zf.shape 

    #     # y = Variable(torch.tensor([x for x in range(c)])).to(self.device)    
    #     y = torch.tensor([x for x in range(b)]) 
    #     y = Variable(y.repeat(c)).to(self.device)
        
    #     zm = zm.unsqueeze(2).view(b, c, -1, d).mean(2) 

    #     prototypes = zm.view(-1, d)  
    #     queries = zf.reshape(b*c, -1) 
        
    #     distances = pairwise_distances(queries, prototypes)
    #     log_p_y = (-distances).log_softmax(dim=1)
    #     y_pred = (-distances).softmax(dim=1)
    #     preds = torch.argmax(y_pred, dim=1)   

    #     acc_pn = (preds==y).cpu().float().mean().item()  
    #     loss_pn = self.proto_criterion(log_p_y, y)  

    #     return loss_pn, acc_pn  

    def proto_loss(self, zm, zf): 
        zf = zf.mean(1)
        zm = zm.mean(1)

        b, d = zf.shape 

        y = Variable(torch.tensor([x for x in range(b)])).to(self.device)      

        prototypes = zf.view(-1, d)  
        queries = zm.view(-1, d) 
        
        distances = pairwise_distances(queries, prototypes)
        log_p_y = (-distances).log_softmax(dim=1)
        y_pred = (-distances).softmax(dim=1)
        preds = torch.argmax(y_pred, dim=1)   

        print(preds)

        acc_pn = (preds==y).cpu().float().mean().item()  
        loss_pn = self.proto_criterion(log_p_y, y)  

        return loss_pn, acc_pn  


    def recon_loss(self, xm, xf, xm_hat, xf_hat):  
        loss_m = self.recon_criterion(xm_hat, xm)
        loss_f = self.recon_criterion(xf_hat, xf) 
        return loss_m, loss_f  


    def forward(self, xm, xf):    

        # encoder
        hm, zm = self.meg_encoder(xm) 
        hf, zf = self.fmri_encoder(xf)  

        xf_hat = self.fmri_decoder(hm) 
        xm_hat = self.meg_decoder(hf)

        
        # loss_pn, acc_pn = self.proto_loss(xm_hat, xf_hat)
        loss_pn, acc_pn = 0, 0

        # import pdb;pdb.set_trace()
        
        xf_hat = self.fmri_out(xf_hat)
        xm_hat = self.meg_out(xm_hat)
        loss_m, loss_f = self.recon_loss(xm, xf, xm_hat, xf_hat)  
        

        # # shared space
        # z_hm = self.shared_encoder(hm)               # 30, 512
        # z_hf = self.shared_encoder(hf)               # 30, 512


        # hm_hat = self.shared_decoder(z_hm)
        # hf_hat = self.shared_decoder(z_hf)

        # # # decoder
        # # if random.random() < 0.3:
        # #     # print("swapped!")
        # #     xm_hat = self.meg_decoder(hm)
        # #     xf_hat = self.fmri_decoder(hf)
        # # else:
        # #     # print("NOT swapped!")
        # #     xm_hat = self.meg_decoder(hf)
        # #     xf_hat = self.fmri_decoder(hm)

        # xm_hat = self.meg_decoder(hm_hat)
        # xf_hat = self.fmri_decoder(hf_hat)
        
        # # xm_hat = torch.sigmoid(xm_hat)
        # # xf_hat = torch.sigmoid(xf_hat)
        return [loss_pn, loss_m, loss_f], acc_pn, [hm, hf], [xm_hat, xf_hat]

  




class LSTMAutoencoder_C(nn.Module):
    def __init__(self, input_size, hidden_size, seq_len, num_layers=1):
        super(LSTMAutoencoder_C, self).__init__()
        
        hidden_size2, hidden_size1 = 2084, 2084
        self.fc_input = nn.Sequential(
            nn.Linear(input_size, hidden_size2),
            nn.ReLU(),
            nn.Linear(hidden_size2, hidden_size1),
            nn.ReLU(),
            nn.Linear(hidden_size1, hidden_size)
        ) 

        self.encoder = nn.LSTM(hidden_size, hidden_size, num_layers=num_layers, batch_first = True)
        self.decoder = nn.LSTM(hidden_size, hidden_size, num_layers=num_layers, batch_first = True)
        self.seq_len = seq_len
        self.hidden_size = hidden_size

        hidden_size2, hidden_size1 = hidden_size*2, hidden_size*3
        self.fc_out = nn.Sequential(
            nn.Linear(hidden_size, hidden_size2),
            nn.ReLU(),
            nn.Linear(hidden_size2, hidden_size1),
            nn.ReLU(),
            nn.Linear(hidden_size1, input_size)
        ) 

    def forward(self, x):

        x = self.fc_input(x) 
        
        _, (hidden, cell) = self.encoder(x) 
        hidden = hidden[-1,:,:]
        
        hidden = hidden.unsqueeze(1).repeat(1, self.seq_len, 1)
        x, (hidden, cell_state) = self.decoder(hidden)
        x = x.reshape((-1, self.seq_len, self.hidden_size))
        x = self.fc_out(x) 

        return [hidden, cell], x

class LSTMAutoencoder3(nn.Module):
    def __init__(self, fmri_seq_len, meg_seq_len, input_size, embedding_size, num_layers, device):
        super(LSTMAutoencoder3, self).__init__()
  
        self.fmri_model = LSTMAutoencoder_C(input_size, embedding_size, fmri_seq_len, num_layers=num_layers)
        self.meg_model = LSTMAutoencoder_C(input_size, embedding_size, meg_seq_len, num_layers=num_layers)

        # self.LSTM1 = nn.LSTM(
        #     input_size = no_features,
        #     hidden_size = self.hidden_size,
        #     num_layers = num_layers,
        #     batch_first = True
        # )

        self.device = device
        self.proto_criterion = torch.nn.NLLLoss()  
        self.recon_criterion = nn.MSELoss()    


    def recon_loss(self, xm, xf, xm_hat, xf_hat):  
        loss_m = self.recon_criterion(xm_hat, xm)
        loss_f = self.recon_criterion(xf_hat, xf) 
        return loss_m, loss_f  


    def forward(self, xm, xf):    
        # import pdb;pdb.set_trace()
        [hm, cm], xm_hat = self.meg_model(xm)
        [hf, cf], xf_hat = self.fmri_model(xf)

        loss_m, loss_f = self.recon_loss(xm, xf, xm_hat, xf_hat) 

        # encoder
        loss_pn, acc_pn = 0, 0
        return [loss_pn, loss_m, loss_f], acc_pn, [hm, hf], [xm_hat, xf_hat]


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
            nn.Linear(input_size, hidden_size2),
            nn.ReLU(),
            nn.Linear(hidden_size2, hidden_size1),
            nn.ReLU(),
            nn.Linear(hidden_size1, d_model)
        ) 

        self.pre_fc_meg = nn.Sequential( 
            nn.Linear(input_size, hidden_size2),
            nn.ReLU(),
            nn.Linear(hidden_size2, hidden_size1),
            nn.ReLU(),
            nn.Linear(hidden_size1, d_model)
        ) 



        self.fmri_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead),
            num_layers=num_layers
        ) 

        self.meg_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead),
            num_layers=num_layers
        ) 


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
            nn.Linear(d_model, hidden_size2),
            nn.ReLU(),
            nn.Linear(hidden_size2, hidden_size1),
            nn.ReLU(),
            nn.Linear(hidden_size1, input_size),
            nn.Sigmoid()
        ) 

         
        self.post_fc_meg = nn.Sequential(
            # nn.LayerNorm(d_model),
            # nn.ReLU(),
            nn.Linear(d_model, hidden_size2),
            nn.ReLU(),
            nn.Linear(hidden_size2, hidden_size1),
            nn.ReLU(),
            nn.Linear(hidden_size1, input_size),
            nn.Sigmoid()
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
        """
        Normalize the input tensor x to the range [0, 1] or [-1, 1].

        Args:
            x (torch.Tensor): Input tensor to be normalized.
            range_type (str): Range of the normalized tensor. Valid options are '[0,1]' or '[-1,1]'.

        Returns:
            Normalized tensor.
        """

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


         
# class TransformerEncoder2(nn.Module):
#     def __init__(self, input_size, d_model, nhead, num_layers, device):
#         super(TransformerEncoder2, self).__init__()

#         self.fmri_transformer = nn.Transformer(nhead=16, num_encoder_layers=12)
#         self.meg_transformer = nn.Transformer(nhead=16, num_encoder_layers=12)


#     def normalize_input(self, x, range_type='[-1,1]'): 
#         if range_type == '[0,1]': 
#             x_min = torch.min(x)
#             x_max = torch.max(x)
#             x_norm = (x - x_min) / (x_max - x_min)
#         elif range_type == '[-1,1]': 
#             x_min = torch.min(x)
#             x_max = torch.max(x)
#             x_norm = ((x - x_min) / (x_max - x_min)) * 2 - 1
#         else:
#             raise ValueError("Invalid range type. Valid options are '[0,1]' or '[-1,1]'.")

#         return x_norm

#     def forward(self, xm, xf): 

#         xm = self.normalize_input(xm)
#         xf = self.normalize_input(xf)
#         import pdb;pdb.set_trace()
#         xm_hat = self.meg_transformer(xm, xf)
#         xf_hat = self.fmri_transformer(xf, xm)

#         loss_m, loss_f = self.recon_loss(xm, xf, xm_hat, xf_hat)  
        
#         acc_pn = loss_pn = 0
#         return [loss_pn, loss_m, loss_f], acc_pn 