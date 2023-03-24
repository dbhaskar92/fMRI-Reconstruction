import torch
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
        return last_lstm_layer_hidden_state
    
    
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
        x = x.unsqueeze(1).repeat(1, self.seq_len, 1)
        x, (hidden_state, cell_state) = self.LSTM1(x)
        x = x.reshape((-1, self.seq_len, self.hidden_size))
        out = self.fc(x)
        return out



class DualLSTMAutoencoder(nn.Module):
    def __init__(self, input_size, embedding_size, num_layers, device):
        super(DualLSTMAutoencoder, self).__init__()
        
        self.encoder_meg = Encoder(num_layers, seq_len = 240, no_features=input_size, embedding_size=embedding_size)
        self.encoder_mri = Encoder(num_layers, seq_len = 30, no_features=input_size, embedding_size=embedding_size)


        self.decoder_meg = Decoder(num_layers, seq_len = 240, no_features=embedding_size, output_size=input_size)
        self.decoder_mri = Decoder(num_layers, seq_len = 30, no_features=embedding_size, output_size=input_size)
 
        self.device = device
        self.recon_loss_fun = nn.MSELoss()
        self.proto_loss_fun = torch.nn.NLLLoss()
 

        # Shared layer 
        hidden_size1, hidden_size2,  alignment_dim = 1024, 512, 256
        self.encoder_mlp = nn.Sequential(
            nn.Linear(embedding_size, hidden_size1),
            nn.ReLU(),
            nn.Linear(hidden_size1, hidden_size2),
            nn.ReLU(),
            nn.Linear(hidden_size2, alignment_dim)
        )

        self.decoder_mlp = nn.Sequential(
            nn.Linear(alignment_dim, hidden_size2),
            nn.ReLU(),
            nn.Linear(hidden_size2, hidden_size1),
            nn.ReLU(),
            nn.Linear(hidden_size1, embedding_size)
        )
 

    def forward(self, xm, xf): 
        zm = self.encoder_meg(xm)
        zf = self.encoder_mri(xf) 

        # shared encoder 
        zf_enc =  self.encoder_mlp(zf)
        zm_enc =  self.encoder_mlp(zm)

        # computing the alignment loss 
        
        # loss_pn, acc_pn = self.proto_loss(zm_enc, zf_enc)  
        loss_pn, acc_pn = 0, 0 

        zf_dec =  self.decoder_mlp(zf_enc)
        zm_dec =  self.decoder_mlp(zm_enc)

        # residuals
        zm = zm + zm_dec
        zf = zf + zf_dec
 
        xm_hat = self.decoder_meg(zm) 
        xf_hat = self.decoder_mri(zf)  

        # # auto-encoder of fmri data 
        loss_m = self.recon_loss_fun(xm_hat, xm)  
        loss_f = self.recon_loss_fun(xf_hat, xf) 

        return [loss_pn, loss_f, loss_m], acc_pn, [xm_hat, xf_hat], [zm_enc, zf_enc]


    def loss_fun(self, x_hat, x):
        return self.loss_function(x_hat, x)


    def proto_loss(self, zm_enc, zf_enc):
        zm_enc = zm_enc.unsqueeze(1).repeat(1, self.decoder_mri.seq_len, 1)
        zf_enc = zf_enc.unsqueeze(1).repeat(1, self.decoder_mri.seq_len, 1)
        b, c, d = zf_enc.shape 
        zm_enc = zm_enc.unsqueeze(2).view(b, c, -1, d).mean(2)

        y = torch.tensor([x for x in range(b)]) 
        y = Variable(y.repeat(c)).to(self.device) 

        queries = zf_enc.view(-1, d) 
        prototypes = zm_enc.view(-1, d)  
        
        distances = pairwise_distances(queries, prototypes)
        log_p_y = (-distances).log_softmax(dim=1)
        y_pred = (-distances).softmax(dim=1)
        preds = torch.argmax(y_pred, dim=1)  
    
        # import pdb; pdb.set_trace()
        acc_pn = (preds==y).cpu().float().mean().item()  
        loss_pn = self.proto_loss_fun(log_p_y, y) 

        return loss_pn, acc_pn  
        

# hidden_size = 20
# num_layers = 1
# batch_size = 2


# input_size = 204 #80
# input_sequence_length = 30
# output_sequence_length = 240

# # Create a random input tensor (batch_size, input_sequence_length, input_size)
# xf = torch.randn(batch_size, 30, input_size)
# xm = torch.randn(batch_size, 240, input_size)



# model = DualLSTMAutoencoder(input_size, hidden_size, num_layers, device='cpu')
# [loss_pn, loss_f, loss_m], acc_pn, [xm_hat, xf_hat], [zm_enc, zf_enc] = model(xm, xf)

# print('xf.shape, xf_hat.shape', xf.shape, xf_hat.shape)
# print('xm.shape, xm_hat.shape', xm.shape, xm_hat.shape)

# print('loss_pn, loss_f, loss_m', loss_pn, loss_f, loss_m)
