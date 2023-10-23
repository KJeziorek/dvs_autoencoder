from utils.extractor_dsec import DSECReader
from utils.extractor_gen1 import Gen1Reader

from tqdm import tqdm

import argparse
import numpy as np
import h5py
import torch
import os
import glob
import torch.nn as nn

SEED = 12345

np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=160*120, type=int) # Based on how many pixels we want to process at ones
    parser.add_argument("--lim_train", default=200, type=int) # Number of training samples of size equal to "batch_size"
    parser.add_argument("--lim_test", default=20, type=int) # Number of testing samples of size equal to "batch_size"
    parser.add_argument("--dataset", default='gen1', type=str) # Used dataset 'gen1' or 'dsec'

    parser.add_argument("--time_window", default=50, type=float) # Time window in miliseconds

    # TODO add arguments to select % of data to split betwen train and test for each file
    return parser.parse_args()

def main(args):
    # Train dataset folders
    os.makedirs(os.path.dirname(f'preprocessed/{args.dataset}/train/'), exist_ok=True)
    os.makedirs(os.path.dirname(f'preprocessed/{args.dataset}/train_lengths/'), exist_ok=True)
    
    # Test dataset folders
    os.makedirs(os.path.dirname(f'preprocessed/{args.dataset}/test/'), exist_ok=True)
    os.makedirs(os.path.dirname(f'preprocessed/{args.dataset}/test_lengths/'), exist_ok=True)

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

    file_idx = 0
    
    print('Preprocessing dataset...')
    print(f'Batch size: {args.batch_size}')
    print(f'Number of training samples: {args.lim_train}')
    print(f'Number of testing samples: {args.lim_test}')
    print(f'Dataset: {args.dataset}')
    print(f'Time window: {args.time_window} ms')

    for data in dataset:
        print(f'Data file: {data}')

        if args.dataset == 'dsec':
            data_reader = DSECReader(data)
        elif args.dataset == 'gen1':
            data_reader = Gen1Reader(data)
        else:
            ValueError('Dataset not supported')

        max_time = data_reader.max_time()
        
        # TODO check if this is correct (maybe we should use it before the for loop)
        seq_to_save = []
        seq_lengths_to_save = []
        batch_idx = 0

        for t_start in tqdm(range(0, int(max_time-args.time_window*1000), args.time_window*1000)):
            
            # Extract events
            t_end = t_start + args.time_window * 1000
            events = data_reader.extract_timewindow(t_start, t_end)
            ex, ey, et, ep = np.float32(events['x']), np.float32(events['y']), np.float32(events['t']), np.float32(events['p'])

            # Time normalization
            et -= et[0]
            et /= et[-1]

            # Stack events
            events = np.column_stack([ex, ey, et])

            seq = [[] for _ in range(dim[0]*dim[1])]
            seq_lengths = np.zeros(dim[0]*dim[1], dtype=np.int64)

            for event in events:
                _idx = event[0] + event[1] * dim[0]

                seq[int(_idx)].append(event[2])
                seq_lengths[int(_idx)] += 1

            for idx in range(dim[0]*dim[1]):
                if seq_lengths[idx] > 0:
                    seq_to_save.append(torch.tensor(np.array(seq[idx])).view(-1, 1))
                    seq_lengths_to_save.append(seq_lengths[idx])

                    batch_idx += 1

                    if batch_idx == args.batch_size:
                        if file_idx < args.lim_train:
                            mode = 'train'
                        elif file_idx < args.lim_train + args.lim_test:
                            mode = 'test'
                        else:
                            exit()

                        padded = nn.utils.rnn.pad_sequence(seq_to_save, batch_first=True)
                        
                        torch.save(padded, f'preprocessed/{args.dataset}/{mode}/{file_idx}')
                        torch.save(seq_lengths_to_save, f'preprocessed/{args.dataset}/{mode}_lengths/{file_idx}_lengths')

                        file_idx += 1
                        batch_idx = 0

                        seq_to_save = []
                        seq_lengths_to_save = []

        data_reader.close()


if __name__ == '__main__':
    args = parse_args()
    main(args)