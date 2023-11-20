## Download data

First, download test_events.zip and test_images.zip from DSEC: https://dsec.ifi.uzh.ch/dsec-datasets/download/

Extract test_events.zip to "events" folder.
Extract test_images.zip to "images" folder.


## Create python env

```
>> conda create -n dvs_enc python=3.9
>> conda activate dvs_enc
>> conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
>> conda install h5py 
>> conda install -c conda-forge blosc-hdf5-plugin
>> pip install matplotlib tqdm
```

## Run code
```
python preproces.py
python train.py
```

Data and code to extract from

```bibtex
@InProceedings{Gehrig21ral,
  author  = {Mathias Gehrig and Willem Aarents and Daniel Gehrig and Davide Scaramuzza},
  title   = {{DSEC}: A Stereo Event Camera Dataset for Driving Scenarios},
  journal = {{IEEE} Robotics and Automation Letters},
  year    = {2021},
  doi     = {10.1109/LRA.2021.3068942}
}
```
## Model

train loss = 0.00130230
test loss = 0.00187137

![video](https://github.com/KJeziorek/dvs_autoencoder/assets/95488355/7c7b4f75-d267-4c8a-86e2-292ca642c12f)
