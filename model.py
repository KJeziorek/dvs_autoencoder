import torch
import torch.nn as nn
import numpy as np 
import os

class AutoEncoderGRU(nn.Module):
    def __init__(self, input_size=1, hidden_size=3, batch_size=128):
        super(AutoEncoderGRU, self).__init__()

        self.input_size = input_size    # Input size = 1 -> e_t * e_p
        self.hidden_size = hidden_size  # Hidden size = 3 -> 3-channels image
        self.output_size = input_size   # Output size == input size = 1

        self.batch_size = batch_size

        self.encoder = nn.GRU(input_size=self.input_size, hidden_size=self.hidden_size, batch_first=True)
        self.decoder = nn.GRU(input_size=self.input_size, hidden_size=self.hidden_size, batch_first=False)
        self.linear = nn.Linear(self.hidden_size, self.output_size) # 

        self.activation = nn.Sigmoid()  # Activation for encoder output -> range (0, 1) 
        self.loss = nn.MSELoss(reduction='none')

        self.random = False             # Random hidden state reset -> if False hidden state = 0
        self.reset_state()

    def forward(self, input, seq_lengths):
        features = self._encoder_forward(input, seq_lengths)
        output = self._decoder_forward(features, seq_lengths)

        loss = self._loss_fun(input, output, seq_lengths)

        return loss, input, output, features
    
    def _encoder_forward(self, padded, seq_lengths):
        packed = nn.utils.rnn.pack_padded_sequence(padded, seq_lengths, batch_first=True, enforce_sorted=False)
        
        _, self.hidden_state = self.encoder(packed, self.hidden_state)
        features = self.activation(self.hidden_state)
        return features
    
    def _decoder_forward(self, features, seq_lengths):
        output_vec = []
        x = torch.zeros(1, self.batch_size, 1)
        for idx in range(max(seq_lengths)):
            x, features = self.decoder(x, features)
            x = self.linear(x)
            output_vec.append(x)

        output_vec = torch.cat(output_vec, dim=0)
        return output_vec.transpose(1, 0)
    
    def _loss_fun(self, input, output, seq_lengths):
        # Masking
        mask = torch.arange(max(seq_lengths))[None, :] < torch.tensor(seq_lengths)[:, None]
        mask = mask.unsqueeze(2)
        loss = self.loss(input, output)
        loss = (loss * mask.float()).sum()
        mse_loss = loss / mask.sum()
        return mse_loss

    def reset_state(self):
        if self.random:
            self.hidden_state = torch.rand(1, self.batch_size, self.hidden_size)
        else:
            self.hidden_state = torch.zeros(1, self.batch_size, self.hidden_size)

    def save_models(self):
        os.makedirs('models', exist_ok=True)
        torch.save(self.encoder.state_dict(), 'models/encoder.pth')
        torch.save(self.decoder.state_dict(), 'models/decoder.pth')


class AutoEncoderLSTM(nn.Module):
    pass #TODO