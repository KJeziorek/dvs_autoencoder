from models.Encoder import EncoderEvent

from utils.extractor_dsec import DSECReader
from utils.extractor_gen1 import Gen1Reader

from time import time
from tqdm import tqdm

import argparse
import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import glob
import os
import cv2

SEED = 12345

np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default='gru', type=str)
    parser.add_argument("--dataset", default='gen1', type=str)

    parser.add_argument("--time_window", default=50, type=float) # Time window in miliseconds
    return parser.parse_args()

def main(args):
    # Get data files names
    if args.dataset == 'gen1':
        dataset = glob.glob('data/gen1/*.h5')

        dim = (304, 240)
    elif args.dataset == 'dsec':
        dataset = glob.glob('data/dsec/events/*')
        dataset = [os.path.join(data, 'events', 'left', 'events.h5') for data in dataset]

        dim = (640, 480)
    else:
        ValueError('Dataset not supported')

    # Model
    encoder = EncoderEvent(input_size=1,
                    hidden_size=3,
                    model=args.model,
                    use_activation=False,
                    use_linear=False,
                    device=device,
                    dataset=args.dataset)

    encoder.eval().to(device)

    for data in reversed(dataset):

        if args.dataset == 'dsec':
            data_reader = DSECReader(data)
        elif args.dataset == 'gen1':
            data_reader = Gen1Reader(data)
        else:
            ValueError('Dataset not supported')

        max_time = data_reader.max_time()

        for t_start in tqdm(range(0, int(max_time-args.time_window*1000), args.time_window*1000)):
            
            # Extract events
            t_end = t_start + args.time_window * 1000
            events = data_reader.extract_timewindow(t_start, t_end)
            ex, ey, et, ep = np.float32(events['x']), np.float32(events['y']), np.float32(events['t']), np.float32(events['p'])

            # Time normalization
            et -= et[0]
            et /= et[-1]                    
                
            events = np.column_stack([ex, ey, et])
            events = torch.tensor(events).to(device)
            
            features = torch.zeros(dim[1], dim[0], 3).to(device)

            for event in events:
                y = int(event[1])
                x = int(event[0])
                t = torch.tensor([[event[2]]]).to(device)

                features[y][x] = encoder(t, features[y][x].unsqueeze(0)).squeeze()
            
            img = features.cpu().detach().numpy()

            cv2.imshow('img', img)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                cv2.destroyAllWindows()
                exit()

if __name__ == '__main__':
    args = parse_args()
    main(args)