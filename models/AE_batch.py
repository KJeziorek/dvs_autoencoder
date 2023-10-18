import torch
import torch.nn as nn
import numpy as np 
import os

class AutoEncoderBatch(nn.Module):
    def __init__(self, input_size, hidden_size, batch_size, model, use_linear, use_activation, device, load_model=False):
        super(AutoEncoderBatch, self).__init__()

        self.device = device
        self.model = model
        self.batch_size = batch_size

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.use_linear = use_linear
        self.use_activation = use_activation

        self.encoder = Encoder(input_size=self.input_size, 
                               hidden_size=self.hidden_size, 
                               batch_size=batch_size, 
                               model=self.model,
                               use_linear=self.use_linear, 
                               use_activation=self.use_activation,
                               device=self.device).to(self.device)

        self.decoder = Decoder(input_size=self.input_size, 
                               hidden_size=self.hidden_size, 
                               batch_size=batch_size, 
                               model=self.model,
                               use_linear=self.use_linear,
                               use_activation=self.use_activation,
                               device=self.device).to(self.device)
        
        if load_model:
            self.encoder.load_state_dict(torch.load(f'trained_models/encoder_{model}.pth'))
            self.decoder.load_state_dict(torch.load(f'trained_models/decoder_{model}.pth'))

        self.loss = nn.MSELoss(reduction='none')
        self.reset_state()

    def forward(self, input, seq_lengths):
        features = self.encoder(input, seq_lengths)
        output = self.decoder(features, seq_lengths)
        loss = self._loss_fun(input, output, seq_lengths)
        return loss, features, input, output

    def _loss_fun(self, input, output, seq_lengths):
        mask = torch.arange(max(seq_lengths))[None, :] < torch.tensor(seq_lengths)[:, None]
        mask = mask.float().unsqueeze(2).to(self.device)
        loss = self.loss(input, output)
        loss = (loss * mask).sum()
        mse_loss = loss / mask.sum()
        return mse_loss
    
    def save_models(self):
        os.makedirs('trained_models', exist_ok=True)
        torch.save(self.encoder.state_dict(), f'trained_models/encoder_{self.model}.pth')
        torch.save(self.decoder.state_dict(), f'trained_models/decoder_{self.model}.pth')

    def reset_state(self):
        self.encoder.reset_state()

###################################################################
############################# ENCODER #############################
###################################################################

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, batch_size, model, use_linear, use_activation, device):
        super(Encoder, self).__init__()

        self.device = device
        self.model = model
        self.batch_size = batch_size

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

    def forward(self, padded, seq_lengths):
        packed = nn.utils.rnn.pack_padded_sequence(padded, seq_lengths, batch_first=True, enforce_sorted=False).to(self.device)
        _, features = self.encoder(packed, self.hidden_state)

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
    
    def reset_state(self):
        if self.model == 'lstm':
            self.hidden_state = (torch.zeros(1, self.batch_size, self.hidden_size).to(self.device),
                                 torch.zeros(1, self.batch_size, self.hidden_size).to(self.device))
        else:
            self.hidden_state = torch.zeros(1, self.batch_size, self.hidden_size).to(self.device)

###################################################################
############################# DECODER #############################
###################################################################

class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, batch_size, model, use_linear, use_activation, device):
        super(Decoder, self).__init__()

        self.device = device
        self.model = model
        self.batch_size = batch_size

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = input_size

        if self.model == 'lstm':
            self.decoder = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, batch_first=False)
        elif self.model == 'gru':
            self.decoder = nn.GRU(input_size=self.input_size, hidden_size=self.hidden_size, batch_first=False)
        elif self.model == 'rnn':
            self.decoder = nn.RNN(input_size=self.input_size, hidden_size=self.hidden_size, batch_first=False)
        else:
            raise ValueError('Model not supported')
        
        self.linear_out = nn.Linear(self.hidden_size, self.output_size)

        self.use_linear = use_linear
        self.linear_dec = nn.Linear(in_features=3, out_features=self.hidden_size)

        self.use_activation = use_activation
        self.activation = nn.Sigmoid()  

    def forward(self, features, seq_lengths):
        output_vec = []
        x = torch.zeros(1, self.batch_size, 1).to(self.device)

        if self.use_linear:
            if self.model == 'lstm':
                features[0] = self.linear_dec(features[0])

        for _ in range(max(seq_lengths)):
            x, features = self.decoder(x, features)
            x = self.linear_out(x)
            output_vec.append(x)

        output_vec = torch.cat(output_vec, dim=0)
        return output_vec.transpose(1, 0)