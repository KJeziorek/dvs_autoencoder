from models.Encoder import EncoderEvent
from utils.extractor import EventReader

from time import time
from tqdm import tqdm

import argparse
import h5py
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

datasets = ['zurich_city_14_c', 'interlaken_00_a', 'interlaken_00_b', 'interlaken_01_a', 'thun_01_a', 
                'thun_01_b', 'zurich_city_12_a', 'zurich_city_13_a', 'zurich_city_13_b', 'zurich_city_14_a', 
                'zurich_city_14_b', 'zurich_city_14_c', 'zurich_city_15_a']

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default='gru', type=str)
    return parser.parse_args()

def main(args):

    data_reader = EventReader()

    encoder = EncoderEvent(input_size=1,
                    hidden_size=3,
                    model=args.model,
                    use_activation=False,
                    use_linear=False,
                    device=device)

    encoder.eval().to(device)

    idx = 0
    for data in datasets:
        events_dir = f'events/{data}/events/left/events.h5'
        timestamps_dir = f'images/{data}/images/timestamps.txt'

        with h5py.File(str(events_dir), 'r') as h5f:
            t_offset = h5f['t_offset'][()]
            timestamps = np.loadtxt(timestamps_dir)

            for t_start in tqdm(timestamps):
                t_end = t_start + 0.05 * 1e6
                events = data_reader.extract_timewindow(h5f, t_start, t_end)

                ex, ey, et, ep = np.float32(events['x']), np.float32(events['y']), np.float32(events['t']), np.float32(events['p'])
                # Time normalization
                et -= et[0]
                et /= et[-1]
                # Change time based on polarity
                et[ep==0] *= 1                      
                
                events = np.column_stack([ex, ey, et])
                
                events = torch.tensor(events).to(device)
                
                features = torch.ones(480, 640, 3).to(device)

                for event in events:
                    features[int(event[1])][int(event[0])] = encoder(torch.tensor([[event[2]]]).to(device), features[int(event[1])][int(event[0])].unsqueeze(0)).squeeze()
                    
                cv2.imshow('img', features.cpu().detach().numpy())
                cv2.waitKey(1)

                idx += 1

                if idx == 5:
                    exit()

if __name__ == '__main__':
    args = parse_args()
    main(args)