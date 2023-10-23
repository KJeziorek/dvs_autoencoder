import torch
import torch.nn as nn
import numpy as np 
import os

class AutoEncoderBatch(nn.Module):
    def __init__(self, input_size, hidden_size, batch_size, model, use_linear, use_activation, device, load_model=False, dataset='gen1'):
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
            self.encoder.load_state_dict(torch.load(f'trained_models/encoder_{model}_{dataset}.pth'))
            self.decoder.load_state_dict(torch.load(f'trained_models/decoder_{model}_{dataset}.pth'))

        self.loss = nn.MSELoss(reduction='none')
        self.reset_state()

    def forward(self, input, seq_lengths):
        output_enc, features = self.encoder(input, seq_lengths)
        output = self.decoder(output_enc, features)
        loss = self._loss_fun(input, output, seq_lengths)
        return loss, features, input, output

    def _loss_fun(self, input, output, seq_lengths):
        mask = torch.arange(max(seq_lengths))[None, :] < torch.tensor(seq_lengths)[:, None]
        mask = mask.float().unsqueeze(2).to(self.device)
        loss = self.loss(input, output)
        loss = (loss * mask).sum()
        mse_loss = loss / mask.sum()
        return mse_loss
    
    def save_models(self, name):
        os.makedirs('trained_models', exist_ok=True)
        torch.save(self.encoder.state_dict(), f'trained_models/encoder_{self.model}_{name}.pth')
        torch.save(self.decoder.state_dict(), f'trained_models/decoder_{self.model}_{name}.pth')

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

        if self.model == 'gru':
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
        output, hidden_state = self.encoder(packed, self.hidden_state)
        
        if self.use_linear:
            hidden_state = self.linear_enc(hidden_state)
        
        if self.use_activation:
            hidden_state = self.activation(hidden_state)
        
        return packed, hidden_state
    
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

        self.teacher_forcing = True

        if self.model == 'gru':
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

    def forward(self, input, hidden):
        output_vec = []

        if self.use_linear:
            hidden = self.linear_dec(hidden)
        
        input = nn.utils.rnn.pad_packed_sequence(input, batch_first=False)[0]

        in_data = input[0].unsqueeze(0)*0

        x, hidden = self.decoder(in_data, hidden)
        x = self.linear_out(x)
        output_vec.append(x)
        
        for idx in range(0, input.shape[0]-1):

            if self.teacher_forcing:
                in_data = input[idx].unsqueeze(0)
            else:
                in_data = x

            x, hidden = self.decoder(in_data, hidden)
            x = self.linear_out(x)
            output_vec.append(x)

        output_vec = torch.cat(output_vec, dim=0)
        return output_vec.transpose(1, 0)