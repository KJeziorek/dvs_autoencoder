import torch
import torch.nn as nn
import numpy as np 
import os

class EncoderEvent(nn.Module):
    def __init__(self, input_size, hidden_size, model, use_linear, use_activation, device, dataset):
        super(EncoderEvent, self).__init__()

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
        
        self.encoder.load_state_dict(torch.load(f'trained_models/encoder_{model}_{dataset}.pth'))

    def forward(self, input, hidden_state):
        features = self.encoder(input, hidden_state)
        return features

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