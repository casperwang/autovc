## AUTOVC: Zero-Shot Voice Style Transfer with Only Autoencoder Loss
### Modified by 22601 Casper Wang, 22625 Sean Liu @ CKHS with lots of help from Inventec 

This repository provides a PyTorch implementation of AUTOVC.

AUTOVC is a many-to-many non-parallel voice conversion framework. 

**_To ensure respect for privacy rights and responsible use of our code, we are only releasing a portion of our code to allow users to convert voices among a predefined set of speakers in VCTK. Conversions from and to other voices have been disabled._**


### Audio Demo

The audio demo for AUTOVC can be found [here](https://auspicious3000.github.io/autovc-demo/)

### Dependencies
- Python 3
- Numpy
- PyTorch >= v0.4.1
- TensorFlow >= v1.3 (only for tensorboard)
- librosa
- tqdm
- wavenet_vocoder ```pip install wavenet_vocoder```
  for more information, please refer to https://github.com/r9y9/wavenet_vocoder

### Pre-trained models

| AUTOVC | WaveNet Vocoder |
|----------------|----------------|
| [link](https://drive.google.com/file/d/1SZPPnWAgpGrh0gQ7bXQJXXjOntbh4hmz/view?usp=sharing)| [link](https://drive.google.com/file/d/1Zksy0ndlDezo9wclQNZYkGi_6i7zi4nQ/view?usp=sharing) |


### 0.Converting Mel-Spectrograms

Download pre-trained AUTOVC model, and run the ```conversion.ipynb``` in the same directory.

### 1.Mel-Spectrograms to waveform

Download pre-trained WaveNet Vocoder model, and run the ```vocoder.ipynb``` in the same the directory.

## Modified Stuff

### Dataset: 
Chinese dataset taken from https://www.data-baker.com/open_source.html, about 12 hours of Mandarin Chinese spoken by the same woman.

### Current Issues 

Cannot do anything with CPU as laptops do not have GPU :(, keep on raising error: ```RuntimeError: Attempting to deserialize object on a CUDA device but torch.cuda.is_available() is False. If you are running on a CPU-only machine, please use torch.load with map_location=torch.device('cpu') to map your storages to the CPU.```, I've tried to modify whatever it tells me to do, but it seems to be to no avail :( 
