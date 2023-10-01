import torch
import torch.nn as nn
import numpy as np 

class AutoEncoderGRU(nn.Module):
    def __init__(self, input_size=1, hidden_size=3, batch_size=128, activation=nn.Sigmoid()):
        super(AutoEncoderGRU, self).__init__()

        self.input_size = input_size    # Input size = 1 -> e_t * e_p
        self.hidden_size = hidden_size  # Hidden_size = 3 -> 3-channels image

        self.batch_size = batch_size

        self.encoder = nn.GRU(input_size=self.input_size, hidden_size=self.hidden_size, batch_first=True)
        self.decoder = nn.GRU(input_size=self.hidden_size, hidden_size=self.hidden_size, batch_first=True)

        self.activation = activation
        
        self.random = False
        self.reset_state()

    def forward(self, x, seq_lengths):
        features = self._encoder_forward(x, seq_lengths)
        return features
    
    def _encoder_forward(self, x, seq_lengths):
        pad = nn.utils.rnn.pad_sequence(x, batch_first=True)    #Padd tensors with 0's
        packed = nn.utils.rnn.pack_padded_sequence(pad, seq_lengths, batch_first=True, enforce_sorted=False)
        
        _, hidden = self.encoder(packed, self.hidden_state)
        features = self.activation(hidden)
        
        return features
    
    def _decoder_forward(self, features, seq_lengths):
        pass
    
    def reset_state(self):
        if self.random:
            self.hidden_state = torch.rand(1, self.batch_size, self.hidden_size)
        else:
            self.hidden_state = torch.zeros(1, self.batch_size, self.hidden_size)

    def save_models(self):
        torch.save(self.encoder.state_dict(), 'models/encoder.pth')
        torch.save(self.decoder.state_dict(), 'models/decoder.pth')

