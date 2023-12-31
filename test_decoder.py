from models.AE_batch import AutoEncoderBatch

from utils.extractor_dsec import DSECReader
from utils.extractor_gen1 import Gen1Reader

import matplotlib.pyplot as plt
import numpy as np
import torch
import argparse
import glob
import os

from tqdm import tqdm
import h5py

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
    model = AutoEncoderBatch(input_size=1,
                             hidden_size=3,
                             batch_size=1,
                             model=args.model,
                             use_activation=True,
                             use_linear=False,
                             device=device,
                             load_model=True,
                             dataset=args.dataset)
    
    model.eval().to(device)

    #inverse loop
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

            seq = [[] for _ in range(dim[0]*dim[1])]
            seq_lengths = np.zeros(dim[0]*dim[1], dtype=np.int64)

            for event in events:
                _idx = event[0] + event[1] * dim[0]
                seq[int(_idx)].append(event[2])
                seq_lengths[int(_idx)] += 1

            for idx in range(dim[0]*dim[1]):
                if seq_lengths[idx] > 10: # TODO: change this
                    data = torch.tensor(np.array(seq[idx])).view(-1, 1)
                    len = seq_lengths[idx]
                    
                    _, _, input, output = model(data.unsqueeze(0).to(device), [len])
                    
                    real = input.view(1,-1).cpu().detach().numpy()[0]
                    decoder = output.view(1,-1).cpu().detach().numpy()[0]
                    
                    print(real, decoder)
                    plt.plot(real, label='real')
                    plt.plot(decoder, label='decoder')
                    plt.title('Real vs Decoder')
                    plt.legend()
                    plt.show()
    
if __name__ == '__main__':
    args = parse_args()
    main(args)