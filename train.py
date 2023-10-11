from models.AE_GRU import AutoEncoderGRU
from models.AE_LSTM import AutoEncoderLSTM

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
    parser.add_argument("--batch_size", default=320*240, type=int) # Based on how many pixels we want to process at ones
    parser.add_argument("--lr", default=0.001, type=float) # Learning rate parameter
    parser.add_argument("--epochs", default=500, type=int) # Epochs over all training and testing preprocessed dataset
    parser.add_argument("--model", default='gru', type=str)

    return parser.parse_args()

def main(args):
    if args.model == 'gru':
        model = AutoEncoderGRU(input_size=1, hidden_size=3, batch_size=args.batch_size, device=device)
    elif args.model == 'lstm':
        model = AutoEncoderLSTM(input_size=1, hidden_size=3, batch_size=args.batch_size, device=device)

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

                seq = torch.load(preprocessed_train_data + '/' + file).to(device)
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

        torch.set_grad_enabled(False)
        model.eval()

        for subdir, dirs, files in os.walk(preprocessed_test_data):
            for file in tqdm(files):

                seq = torch.load(preprocessed_test_data + '/' + file).to(device)
                seq_length = torch.load(preprocessed_test_data + '_lengths/' + file + '_lengths')

                loss, input, output, features = model(seq, seq_length)

                loss_sum += loss.item()
                loss_iter += 1

                # Visualization
                img = features.cpu().detach().numpy()
                img = img.reshape(240, 320, 3)
                img = cv2.normalize(img, None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                scale_percent = 300 # percent of original size
                width = int(img.shape[1] * scale_percent / 100)
                height = int(img.shape[0] * scale_percent / 100)
                dim = (width, height)
                
                # resize image
                resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
                cv2.imshow('Features', resized)
                cv2.waitKey(1) 

        test_loss_vec.append(loss_sum/loss_iter)
        print('Test loss  = ', loss_sum/loss_iter)
        
        torch.set_grad_enabled(True)
        model.train()

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