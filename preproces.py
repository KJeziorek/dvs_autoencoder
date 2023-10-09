from utils.extractor import EventReader
from model import AutoEncoderGRU

import argparse
from time import time
from tqdm import tqdm

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

# os.environ["CUDA_VISIBLE_DEVICES"]="0"
# use_cuda = torch.cuda.is_available()
# device = torch.device("cuda:0" if use_cuda else "cpu")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=1024, type=int)
    parser.add_argument("--lim_train", default=800, type=int)
    parser.add_argument("--lim_test", default=200, type=int)
    return parser.parse_args()

def main(args):
    # # File names
    train_dataset = ['zurich_city_14_c', 'interlaken_00_a', 'interlaken_00_b', 'interlaken_01_a', 'thun_01_a', 
                    'thun_01_b', 'zurich_city_12_a', 'zurich_city_13_a', 'zurich_city_13_b', 'zurich_city_14_a']
    test_dataset = ['zurich_city_14_b', 'zurich_city_14_c', 'zurich_city_15_a']

    # Reader class
    data_reader = EventReader()

    # Train dataset

    os.makedirs(os.path.dirname('preprocessed/train/'), exist_ok=True)
    os.makedirs(os.path.dirname('preprocessed/train_lengths/'), exist_ok=True)
    
    file_idx = 0

    for train_data in train_dataset:
        events_dir = f'events/{train_data}/events/left/events.h5'
        timestamps_dir = f'images/{train_data}/images/timestamps.txt'

        with h5py.File(str(events_dir), 'r') as h5f:
            t_offset = h5f['t_offset'][()]
            timestamps = np.loadtxt(timestamps_dir)

            print(f'Train data: {train_data}')
            for t_start in tqdm(timestamps):
                t_end = t_start + 0.05 * 1e6
                events = data_reader.extract_timewindow(h5f, t_start, t_end)

                ex, ey, et, ep = np.float32(events['x']), np.float32(events['y']), np.float32(events['t']), np.float32(events['p'])
                et = (et - et[0]) * (1 / et[-1])    # Time normalization
                et[ep==0] = -1                      # If -1 add polarity
                
                events = np.column_stack([ex, ey, et, ep])

                seq = []
                seq_lengths = []
                batch_idx = 0
                
                for idx in range(640*480):
                    idx_x, idx_y = idx%640 , idx//640
                    idx_xy = (events[:,0] == idx_x) & (events[:,1] == idx_y)
                    event_window = torch.tensor(et[idx_xy])

                    if len(event_window) > 0:
                        seq.append(event_window.view(-1, 1))
                        seq_lengths.append(len(event_window))
                        batch_idx += 1
                    else:
                        continue
                    
                    if batch_idx == args.batch_size:
                        padded = nn.utils.rnn.pad_sequence(seq, batch_first=True)    #Padd tensors with 0's
                        #packed = nn.utils.rnn.pack_padded_sequence(padded, seq_lengths, batch_first=True, enforce_sorted=False)
                        
                        torch.save(padded, f'preprocessed/train/{train_data}_{file_idx}')
                        torch.save(seq_lengths, f'preprocessed/train_lengths/{train_data}_{file_idx}_lengths')
                    
                        seq = []
                        seq_lengths = []
                        batch_idx = 0
                        file_idx += 1

                        if file_idx == args.lim_train:
                            break
                else:
                    continue
                break
            else:
                continue
            break
            
    # Test dataset

    os.makedirs(os.path.dirname('preprocessed/test/'), exist_ok=True)
    os.makedirs(os.path.dirname('preprocessed/test_lengths/'), exist_ok=True)

    file_idx = 0

    for test_data in test_dataset:
        events_dir = f'events/{test_data}/events/left/events.h5'
        timestamps_dir = f'images/{test_data}/images/timestamps.txt'

        with h5py.File(str(events_dir), 'r') as h5f:
            t_offset = h5f['t_offset'][()]
            timestamps = np.loadtxt(timestamps_dir)

            print(f'Test data: {test_data}')
            for t_start in tqdm(timestamps):
                t_end = t_start + 0.05 * 1e6
                events = data_reader.extract_timewindow(h5f, t_start, t_end)

                ex, ey, et, ep = np.float32(events['x']), np.float32(events['y']), np.float32(events['t']), np.float32(events['p'])
                et = (et - et[0]) * (1 / et[-1])    # Time normalization
                et[ep==0] = -1                      # If -1 add polarity
                
                events = np.column_stack([ex, ey, et, ep])

                seq = []
                seq_lengths = []
                
                batch_idx = 0
                
                
                for idx in range(640*480):
                    idx_x, idx_y = idx%640 , idx//640
                    idx_xy = (events[:,0] == idx_x) & (events[:,1] == idx_y)
                    event_window = torch.tensor(et[idx_xy])

                    if len(event_window) > 0:
                        seq.append(event_window.view(-1, 1))
                        seq_lengths.append(len(event_window))
                        batch_idx += 1
                    else:
                        continue
                    
                    if batch_idx == args.batch_size:
                        padded = nn.utils.rnn.pad_sequence(seq, batch_first=True)    #Padd tensors with 0's
                        #packed = nn.utils.rnn.pack_padded_sequence(padded, seq_lengths, batch_first=True, enforce_sorted=False)
                        
                        torch.save(padded, f'preprocessed/test/{test_data}_{file_idx}')
                        torch.save(seq_lengths, f'preprocessed/test_lengths/{test_data}_{file_idx}_lengths')
                        
                        seq = []
                        seq_lengths = []
                        batch_idx = 0
                        file_idx += 1

                        if file_idx == args.lim_test:
                            break
                else:
                    continue
                break
            else:
                continue
            break

if __name__ == '__main__':
    args = parse_args()
    main(args)