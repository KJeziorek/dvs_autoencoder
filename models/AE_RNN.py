import torch
import torch.nn as nn
import numpy as np 
import os

class AutoEncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, batch_size, device):
        super(AutoEncoderRNN, self).__init__()

        self.device = device
        self.batch_size = batch_size

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = input_size

        self.encoder = Encoder(input_size=self.input_size, hidden_size=self.hidden_size, batch_size=batch_size, device=self.device).to(self.device)
        self.decoder = Decoder(input_size=self.input_size, hidden_size=self.hidden_size, batch_size=batch_size, device=self.device).to(self.device)

        self.loss = nn.MSELoss(reduction='none')

        self.encoder.reset_state(random=False)

    def forward(self, input, seq_lengths):
        features = self.encoder(input, seq_lengths)
        output = self.decoder(features, seq_lengths)

        loss = self._loss_fun(input, output, seq_lengths)

        return loss, input, output, features

    def _loss_fun(self, input, output, seq_lengths):
        mask = torch.arange(max(seq_lengths))[None, :] < torch.tensor(seq_lengths)[:, None]
        mask = mask.float().unsqueeze(2).to(self.device)
        loss = self.loss(input, output)
        loss = (loss * mask).sum()
        mse_loss = loss / mask.sum()
        return mse_loss
    
    def save_models(self):
        os.makedirs('trained_models', exist_ok=True)
        torch.save(self.encoder.state_dict(), 'trained_models/encoder_gru.pth')
        torch.save(self.decoder.state_dict(), 'trained_models/decoder_gru.pth')

    def reset_state(self):
        self.encoder.reset_state(random=False)

###################################################################
############################# ENCODER #############################
###################################################################

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, batch_size, device):
        super(Encoder, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size

        self.device = device
        self.encoder = nn.RNN(input_size=self.input_size, hidden_size=self.hidden_size, batch_first=True)

        self.activation = nn.Sigmoid()  

        self.random = False  
        self.use_activation = True

        if hidden_size > 3:
            self.use_linear = True
            self.linear_enc = nn.Linear(in_features=hidden_size, out_features=3)
        else:
            self.use_linear = False

    def forward(self,padded,  seq_lengths):
        packed = nn.utils.rnn.pack_padded_sequence(padded, seq_lengths, batch_first=True, enforce_sorted=False).to(self.device)
        
        _, features = self.encoder(packed, self.hidden_state)

        if self.use_linear:
            features = self.linear_enc(features)

        if self.use_activation:
            features = self.activation(features)

        return features
    
    def reset_state(self, random=False):
        if random:
            self.hidden_state = torch.rand(1, self.batch_size, self.hidden_size).to(self.device)
        else:
            self.hidden_state = torch.zeros(1, self.batch_size, self.hidden_size).to(self.device)

###################################################################
############################# DECODER #############################
###################################################################

class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, batch_size, device):
        super(Decoder, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = input_size 

        self.batch_size = batch_size
        self.device = device

        self.decoder = nn.RNN(input_size=self.input_size, hidden_size=self.hidden_size, batch_first=False)
        self.linear_out = nn.Linear(self.hidden_size, self.output_size)

        self.activation = nn.Sigmoid()  
        self.use_linear = False

        if hidden_size > 3:
            self.use_linear = True
            self.linear_dec = nn.Linear(in_features=3, out_features=hidden_size)

    def forward(self, features, seq_lengths):
        output_vec = []
        x = torch.zeros(1, self.batch_size, 1).to(self.device)

        if self.use_linear:
            features = self.linear_dec(features)

        for _ in range(max(seq_lengths)):
            x, features = self.decoder(x, features)
            x = self.linear_out(x)
            output_vec.append(x)

        output_vec = torch.cat(output_vec, dim=0)
        return output_vec.transpose(1, 0)
    

###################################################################
######################### ENCODER TESTING #########################
###################################################################

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, batch_size, device):
        super(Encoder, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size

        self.device = device
        self.encoder = nn.RNN(input_size=self.input_size, hidden_size=self.hidden_size, batch_first=True)

        self.activation = nn.Sigmoid()  

        self.random = False  
        self.use_activation = True

        if hidden_size > 3:
            self.use_linear = True
            self.linear_enc = nn.Linear(in_features=hidden_size, out_features=3)
        else:
            self.use_linear = False

    def forward(self, x,  hidden_state):
        
        _, features = self.encoder(x, hidden_state)

        if self.use_linear:
            features = self.linear_enc(features)

        if self.use_activation:
            features = self.activation(features)

        return features