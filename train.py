from utils.extractor import EventReader
from model import AutoEncoderGRU

from time import time
from tqdm import tqdm

import argparse
import matplotlib.pyplot as plt
import numpy as np
import h5py
import torch
import os
import itertools as it
import torch.nn as nn

SEED = 12345

np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

os.environ["CUDA_VISIBLE_DEVICES"]="0"
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=1024, type=int)
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--epochs", default=500, type=int)

    return parser.parse_args()

def main(args):
    model = AutoEncoderGRU(input_size=1, hidden_size=10, batch_size=args.batch_size)
    model_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    preprocessed_train_data = 'preprocessed/train'
    preprocessed_test_data = 'preprocessed/test'

    train_loss_vec = []
    test_loss_vec = []

    for epoch_idx in range(args.epochs):
        print(f'Epoch {epoch_idx}')
        loss_sum = 0
        loss_iter = 0

        for subdir, dirs, files in os.walk(preprocessed_train_data):
            for file in tqdm(files):

                seq = torch.load(preprocessed_train_data + '/' + file)
                seq_length = torch.load(preprocessed_train_data + '_lengths/' + file + '_lengths')

                loss, input, output, features = model(seq, seq_length)
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

        with torch.no_grad():
            for subdir, dirs, files in os.walk(preprocessed_test_data):
                for file in tqdm(files):

                    seq = torch.load(preprocessed_test_data + '/' + file)
                    seq_length = torch.load(preprocessed_test_data + '_lengths/' + file + '_lengths')

                    loss, input, output, features = model(seq, seq_length)

                    loss_sum += loss.item()
                    loss_iter += 1

        test_loss_vec.append(loss_sum/loss_iter)
        print('Test loss  = ', loss_sum/loss_iter)
    
    print('Input data: \n', input[0])
    print('Output data: \n', output[0])

    os.makedirs('results', exist_ok=True)

    plt.plot(train_loss_vec)
    plt.title('results/Train loss')
    plt.savefig('results/Train_loss.png')

    plt.clf()

    plt.plot(test_loss_vec)
    plt.title('Test loss')
    plt.savefig('Test_loss.png')

    model.save_models()

if __name__ == '__main__':
    args = parse_args()
    main(args)