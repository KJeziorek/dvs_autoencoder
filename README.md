## Download data

First, download test_events.zip and test_images.zip from DSEC: https://dsec.ifi.uzh.ch/dsec-datasets/download/

Extract test_events.zip to events folder.
Extract test_images.zip to images folder.


## Create python env

```
>> conda create -n dvs_enc python=3.9
>> conda activate dvs_enc
>> conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
>> conda install h5py
>> pip install matplotlib tqdm
```

## Run code
```
python preproces.py
python train.py
```