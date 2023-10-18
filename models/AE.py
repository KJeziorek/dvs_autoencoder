import torch
import torch.nn as nn
import numpy as np 
import os

class AutoEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, model, use_linear, use_activation, device):
        super(AutoEncoder, self).__init__()

        self.device = device
        self.model = model

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.use_linear = use_linear
        self.use_activation = use_activation

        self.encoder = Encoder(input_size=self.input_size, 
                               hidden_size=self.hidden_size, 
                               model=self.model,
                               use_linear=self.use_linear, 
                               use_activation=self.use_activation,
                               device=self.device).to(self.device)
        
        self.encoder.load_state_dict(torch.load(f'trained_models/encoder_{model}.pth'))

        self.decoder = Decoder(input_size=self.input_size, 
                               hidden_size=self.hidden_size, 
                               model=self.model,
                               use_linear=self.use_linear,
                               use_activation=self.use_activation,
                               device=self.device).to(self.device)

        self.decoder.load_state_dict(torch.load(f'trained_models/decoder_{model}.pth'))

        self.loss = nn.MSELoss(reduction='none')

    def forward(self, input, hidden_state):
        features = self.encoder(input, hidden_state)
        output = self.decoder(features)
        loss = self._loss_fun(input, output)
        return loss, features, input, output

    def _loss_fun(self, input, output):
        mask = torch.arange(len(input))[None, :] < torch.tensor(len(input))[:, None]
        mask = mask.float().unsqueeze(2).to(self.device)
        loss = self.loss(input, output)
        loss = (loss * mask).sum()
        mse_loss = loss / mask.sum()
        return mse_loss

###################################################################
############################# ENCODER #############################
###################################################################

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, model, use_linear, use_activation, device):
        super(Encoder, self).__init__()

        self.device = device
        self.model = model

        self.input_size = input_size
        self.hidden_size = hidden_size

        if self.model == 'lstm':
            self.encoder = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, batch_first=True)
        elif self.model == 'gru':
            self.encoder = nn.GRU(input_size=self.input_size, hidden_size=self.hidden_size, batch_first=True)
        elif self.model == 'rnn':
            self.encoder = nn.RNN(input_size=self.input_size, hidden_size=self.hidden_size, batch_first=True)
        else:
            raise ValueError('Model not supported')

        self.use_linear = use_linear
        self.linear_enc = nn.Linear(in_features=hidden_size, out_features=3)

        self.use_activation = use_activation
        self.activation = nn.Sigmoid()  

    def forward(self, x, hidden_state):
        _, features = self.encoder(x, hidden_state)

        if self.use_linear:
            if self.model == 'lstm':
                features[0] = self.linear_enc(features[0])
            else:
                features = self.linear_enc(features)

        if self.use_activation:
            if self.model == 'lstm':
                features[0] = self.activation(features[0])
            else:
                features = self.activation(features)

        return features

###################################################################
############################# DECODER #############################
###################################################################

class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, model, use_linear, use_activation, device):
        super(Decoder, self).__init__()

        self.device = device
        self.model = model

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = input_size

        if self.model == 'lstm':
            self.decoder = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, batch_first=True)
        elif self.model == 'gru':
            self.decoder = nn.GRU(input_size=self.input_size, hidden_size=self.hidden_size, batch_first=True)
        elif self.model == 'rnn':
            self.decoder = nn.RNN(input_size=self.input_size, hidden_size=self.hidden_size, batch_first=True)
        else:
            raise ValueError('Model not supported')
        
        self.linear_out = nn.Linear(self.hidden_size, self.output_size)

        self.use_linear = use_linear
        self.linear_dec = nn.Linear(in_features=3, out_features=self.hidden_size)

        self.use_activation = use_activation
        self.activation = nn.Sigmoid()  

    def forward(self, features, seq_lengths):
        output_vec = []
        x = torch.zeros(1).view(-1, 1)

        if self.use_linear:
            if self.model == 'lstm':
                features[0] = self.linear_dec(features[0])
            else:
                features = self.linear_dec(features)

        for _ in range(max(seq_lengths)):
            _, features = self.decoder(x, features)
            x = self.linear_out(features)
            output_vec.append(x)

        output_vec = torch.cat(output_vec, dim=0)
        return output_vec.transpose(1, 0)