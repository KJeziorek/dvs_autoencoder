import torch
import torch.nn as nn
import numpy as np 
import os

class AutoEncoderLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, batch_size, device):
        super(AutoEncoderLSTM, self).__init__()

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
        h, c = self.encoder(input, seq_lengths)
        output = self.decoder(h, c, seq_lengths)

        loss = self._loss_fun(input, output, seq_lengths)

        return loss, input, output, h

    def _loss_fun(self, input, output, seq_lengths):
        mask = torch.arange(max(seq_lengths))[None, :] < torch.tensor(seq_lengths)[:, None]
        mask = mask.float().unsqueeze(2).to(self.device)
        loss = self.loss(input, output)
        loss = (loss * mask).sum()
        mse_loss = loss / mask.sum()
        return mse_loss
    
    def save_models(self):
        os.makedirs('trained_models', exist_ok=True)
        torch.save(self.encoder.state_dict(), 'trained_models/encoder_lstm.pth')
        torch.save(self.decoder.state_dict(), 'trained_models/decoder_lstm.pth')

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
        self.encoder = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, batch_first=True)

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
        
        _, (h, c) = self.encoder(packed, self.hidden_state)

        if self.use_linear:
            h = self.linear_enc(h)

        if self.use_activation:
            h = self.activation(h)

        return h, c
    
    def reset_state(self, random):
        if random:
            self.hidden_state = torch.rand(1, self.batch_size, self.hidden_size).to(self.device)
            self.cell_state = torch.rand(1, self.batch_size, self.hidden_size).to(self.device)
        else:
            self.hidden_state = torch.zeros(1, self.batch_size, self.hidden_size).to(self.device)
            self.cell_state = torch.zeros(1, self.batch_size, self.hidden_size).to(self.device)

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

        self.decoder = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, batch_first=False)
        self.linear_out = nn.Linear(self.hidden_size, self.output_size)

        self.activation = nn.Sigmoid()  
        self.use_linear = False

        if hidden_size > 3:
            self.use_linear = True
            self.linear_dec = nn.Linear(in_features=3, out_features=hidden_size)

    def forward(self, h, c, seq_lengths):
        output_vec = []
        x = torch.zeros(1, self.batch_size, 1).to(self.device)

        if self.use_linear:
            h = self.linear_dec(h)

        for _ in range(max(seq_lengths)):
            x, (h, c) = self.decoder(x, (h, c))
            x = self.linear_out(x)
            output_vec.append(x)

        output_vec = torch.cat(output_vec, dim=0)
        return output_vec.transpose(1, 0)