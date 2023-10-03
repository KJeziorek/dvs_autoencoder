import torch
import torch.nn as nn
import numpy as np 

class AutoEncoderGRU(nn.Module):
    def __init__(self, input_size=1, hidden_size=3, batch_size=128, activation=nn.Sigmoid()):
        super(AutoEncoderGRU, self).__init__()

        self.input_size = input_size    # Input size = 1 -> e_t * e_p
        self.hidden_size = hidden_size  # Hidden_size = 3 -> 3-channels image
        self.output_size = input_size

        self.batch_size = batch_size

        self.encoder = nn.GRU(input_size=self.input_size, hidden_size=self.hidden_size, batch_first=True)
        self.decoder = nn.GRU(input_size=self.input_size, hidden_size=self.hidden_size, batch_first=False)
        self.linear = nn.Linear(self.hidden_size, self.output_size)

        self.activation = activation
        self.loss = nn.MSELoss()

        self.random = False
        self.reset_state()

    def forward(self, x, seq_lengths):
        features, input = self._encoder_forward(x, seq_lengths)
        output = self._decoder_forward(features, seq_lengths)
        loss = self._loss_fun(input, output, seq_lengths)

        return loss, input, output
    
    def _encoder_forward(self, x, seq_lengths):
        padded = nn.utils.rnn.pad_sequence(x, batch_first=True)    #Padd tensors with 0's
        packed = nn.utils.rnn.pack_padded_sequence(padded, seq_lengths, batch_first=True, enforce_sorted=False)
        
        _, self.hidden_state = self.encoder(packed, self.hidden_state)
        features = self.activation(self.hidden_state)
        
        return features, padded
    
    def _decoder_forward(self, features, seq_lengths):
        hidden = features
        output_vec = []
        input = torch.zeros(1, self.batch_size, 1)
        for idx in range(max(seq_lengths)):
            _, hidden = self.decoder(input, hidden)
            output = self.linear(hidden)
            output_vec.append(output)
            input = output

        output_vec = torch.cat(output_vec, dim=0)
        return output_vec.transpose(1, 0)
    
    def _loss_fun(self, input, output, seq_lengths):
        # Masking
        for idx in range(self.batch_size):
            output[idx, seq_lengths[idx]:] = 0
        
        loss = self.loss(input, output)
        return loss

    def reset_state(self):
        if self.random:
            self.hidden_state = torch.rand(1, self.batch_size, self.hidden_size)
        else:
            self.hidden_state = torch.zeros(1, self.batch_size, self.hidden_size)

    def save_models(self):
        torch.save(self.encoder.state_dict(), 'models/encoder.pth')
        torch.save(self.decoder.state_dict(), 'models/decoder.pth')

