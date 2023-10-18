# Description: Training script for the AutoEncoder model
from models.AE_batch import AutoEncoderBatch

from time import time
from tqdm import tqdm

import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import cv2

SEED = 12345

np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=160*120, type=int) # Based on how many pixels we want to process at ones
    parser.add_argument("--lr", default=0.001, type=float) # Learning rate parameter
    parser.add_argument("--epochs", default=50, type=int) # Epochs over all training and testing preprocessed dataset
    parser.add_argument("--model", default='gru', type=str) # Model to use: lstm, gru, rnn
    parser.add_argument("--hidden_size", default=3, type=int) # Hidden size of the LSTM/GRU/RNN
    parser.add_argument("--use_activation", default=False, type=bool) # Use activation function after the encoder
    parser.add_argument("--use_linear", default=False, type=bool) # Use linear layer after the encoder

    return parser.parse_args()

def main(args):
    
    model = AutoEncoderBatch(input_size=1, 
                             hidden_size=args.hidden_size, 
                             batch_size=args.batch_size, 
                             model=args.model,
                             use_activation=args.use_activation, 
                             use_linear=args.use_linear,
                             device=device)

    model_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    preprocessed_train_data = 'preprocessed/train'
    preprocessed_test_data = 'preprocessed/test'

    train_loss_vec = []
    test_loss_vec = []

    for epoch_idx in range(args.epochs):
        print(f'Epoch {epoch_idx}')
        loss_sum = 0
        loss_iter = 0

        torch.set_grad_enabled(True)
        model.train()

        for subdir, dirs, files in os.walk(preprocessed_train_data):
            for file in tqdm(files):

                seq = torch.load(preprocessed_train_data + '/' + file).to(device)
                seq_length = torch.load(preprocessed_train_data + '_lengths/' + file + '_lengths')

                loss, features, input, output = model(seq, seq_length)

                model_optimizer.zero_grad()
                loss.backward()
                model_optimizer.step()

                loss_sum += loss.item()
                loss_iter += 1
                model.reset_state()

        train_loss_vec.append(loss_sum/loss_iter)        
        print('Train loss = ', loss_sum/loss_iter)
        
        loss_sum = 0
        loss_iter = 0

        torch.set_grad_enabled(False)
        model.eval()

        for subdir, dirs, files in os.walk(preprocessed_test_data):
            for file in tqdm(files):

                seq = torch.load(preprocessed_test_data + '/' + file).to(device)
                seq_length = torch.load(preprocessed_test_data + '_lengths/' + file + '_lengths')

                loss, features, input, output = model(seq, seq_length)

                loss_sum += loss.item()
                loss_iter += 1
                
        test_loss_vec.append(loss_sum/loss_iter)
        print('Test loss  = ', loss_sum/loss_iter)
        
    print('Input data: \n', input[0])
    print('Output data: \n', output[0])

    os.makedirs('results', exist_ok=True)

    plt.plot(train_loss_vec)
    plt.title('Train loss')
    plt.savefig('results/Train_loss.png')

    plt.clf()

    plt.plot(test_loss_vec)
    plt.title('Test loss')
    plt.savefig('results/Test_loss.png')

    model.save_models()

if __name__ == '__main__':
    args = parse_args()
    main(args)