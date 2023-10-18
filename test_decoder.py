from models.AE_batch import AutoEncoderBatch

import numpy as np
import torch

SEED = 12345

np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data = [0., 0.1, 0.2, 0.3, 0.4, 0.5]

def main():
    model = AutoEncoderBatch(input_size=1,
                             hidden_size=3,
                             batch_size=1,
                             model='gru',
                             use_activation=False,
                             use_linear=False,
                             device=device,
                             load_model=True)
    
    model.eval().to(device)

    _, _, input, output = model(torch.tensor(data).unsqueeze(1).unsqueeze(0).to(device), [len(data)])

    print(input, output)
        
if __name__ == '__main__':
    main()