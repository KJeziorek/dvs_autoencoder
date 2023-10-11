from utils.extractor import EventReader

from tqdm import tqdm

import argparse
import numpy as np
import h5py
import torch
import os
import torch.nn as nn

SEED = 12345

np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=320*240, type=int) # Based on how many pixels we want to process at ones
    parser.add_argument("--lim_train", default=100, type=int) # Number of training samples of size equal to "batch_size"
    parser.add_argument("--lim_test", default=20, type=int) # Number of testing samples of size equal to "batch_size"
    # TODO
    # add arguments to select % of data to split betwen train and test for each file
    return parser.parse_args()

def main(args):
    # File names
    datasets = ['zurich_city_14_c', 'interlaken_00_a', 'interlaken_00_b', 'interlaken_01_a', 'thun_01_a', 
                'thun_01_b', 'zurich_city_12_a', 'zurich_city_13_a', 'zurich_city_13_b', 'zurich_city_14_a', 
                'zurich_city_14_b', 'zurich_city_14_c', 'zurich_city_15_a']

    # Reader class
    data_reader = EventReader()

    # Train dataset folders
    os.makedirs(os.path.dirname('preprocessed/train/'), exist_ok=True)
    os.makedirs(os.path.dirname('preprocessed/train_lengths/'), exist_ok=True)
    
    # Test dataset folders
    os.makedirs(os.path.dirname('preprocessed/test/'), exist_ok=True)
    os.makedirs(os.path.dirname('preprocessed/test_lengths/'), exist_ok=True)

    file_idx = 0

    for data in datasets:
        events_dir = f'events/{data}/events/left/events.h5'
        timestamps_dir = f'images/{data}/images/timestamps.txt'

        with h5py.File(str(events_dir), 'r') as h5f:
            t_offset = h5f['t_offset'][()]
            timestamps = np.loadtxt(timestamps_dir)

            print(f'Data file: {data}')
            for t_start in tqdm(timestamps):
                t_end = t_start + 0.05 * 1e6
                events = data_reader.extract_timewindow(h5f, t_start, t_end)

                ex, ey, et, ep = np.float32(events['x']), np.float32(events['y']), np.float32(events['t']), np.float32(events['p'])
                et = (et - et[0]) * (1 / et[-1])     # Time normalization
                et[ep==0] *= -1                      # Change time based on polarity
                
                events = np.column_stack([ex, ey, et, ep])

                seq = []
                seq_lengths = []
                
                for idx in range(640*480):
                    # Split data to train and test sets
                    if file_idx < args.lim_train:
                        mode = 'train'
                    elif file_idx < args.lim_train + args.lim_test:
                        mode = 'test'
                    else:
                        exit()

                    # Find events for one pixel and create tensor
                    idx_x, idx_y = idx%640 , idx//640
                    idx_xy = (events[:,0] == idx_x) & (events[:,1] == idx_y)
                    event_window = torch.tensor(et[idx_xy])

                    if len(event_window) > 0:
                        # If there is at least one event for pixel (idx_x, idx_y)
                        seq.append(event_window.view(-1, 1))
                        seq_lengths.append(len(event_window))
                    else:
                        # If there is not events for pixel (idx_x, idx_y)
                        seq.append(torch.tensor([[np.float32(0)]]))
                        seq_lengths.append(1)

                    if len(seq_lengths) == args.batch_size:
                        # If we have batch_size number of tensors
                        # Padding so that each tensor has the same lenght - used for packing_padded_tensor later

                        padded = nn.utils.rnn.pad_sequence(seq, batch_first=True)    #Padd tensors with 0's
                        
                        torch.save(padded, f'preprocessed/{mode}/{data}_{file_idx}')
                        torch.save(seq_lengths, f'preprocessed/{mode}_lengths/{data}_{file_idx}_lengths')
                    
                        seq = []
                        seq_lengths = []
                        file_idx += 1

if __name__ == '__main__':
    args = parse_args()
    main(args)