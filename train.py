from utils.extractor import EventReader
from model import AutoEncoderGRU

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

os.environ["CUDA_VISIBLE_DEVICES"]="0"
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

#Data load
dataset = 'zurich_city_14_c' # interlaken_00_a interlaken_00_b interlaken_01_a thun_01_a thun_01_b zurich_city_12_a zurich_city_13_a zurich_city_13_b
                            # zurich_city_14_a zurich_city_14_b zurich_city_14_c zurich_city_15_a

events_dir = f'test_events/{dataset}/events/left/events.h5'
timestamps_dir = f'test_images/{dataset}/images/timestamps.txt'

data_reader = EventReader()

batch_size = 128
model = AutoEncoderGRU(input_size=1, hidden_size=10, batch_size=batch_size)
model_optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Parameters
with h5py.File(str(events_dir), 'r') as h5f:
    t_offset = h5f['t_offset'][()]
    timestamps = np.loadtxt(timestamps_dir)
    t_start = timestamps[0]
    t_end = t_start + 0.05 * 1e6 # Timewindow equal to 0.05 s

    iter = 1
    while t_end < timestamps[-1]:
        events = data_reader.extract_timewindow(h5f, t_start, t_end)

        ex, ey, et, ep = np.float32(events['x']), np.float32(events['y']), np.float32(events['t']), np.float32(events['p'])
        et -= et[0]
        et /= et[-1]
        et[ep==0] *= 1 # If -1 add polarity
        
        events = np.column_stack([ex, ey, et, ep])

        batch_idx = 0
        seq_lengths = []
        seq = []
        loss_sum = 0
        loss_iter = 0
        for idx in tqdm(range(640*48)):
            idx_x, idx_y = idx%640 , idx//640
            idx_xy = (events[:,0] == idx_x) & (events[:,1] == idx_y)
            event_window = torch.tensor(et[idx_xy])

            if len(event_window) > 0:
                seq.append(event_window.view(-1, 1))
                seq_lengths.append(len(event_window))
                batch_idx += 1
            else:
                continue
            
            if batch_idx == batch_size:
                loss, input, output, features = model(seq, seq_lengths)
                model_optimizer.zero_grad()
                loss.backward()
                model_optimizer.step()
                loss_sum += loss
                loss_iter += 1
                model.reset_state()
                batch_idx = 0
                seq_lengths = []
                seq = []
        
        
        print('Loss for image = ', loss_sum/loss_iter)
        print('Input:')
        print(input[0])
        print('Output:')
        print(output[0])

        t_start = timestamps[iter]
        t_end = t_start + 0.05 * 1e6
        iter += 1
