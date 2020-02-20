import pickle
from sklearn import preprocessing
import numpy as np
import torch
from math import ceil
from torch.utils.data import Dataset, DataLoader

from resemblyzer import preprocess_wav, VoiceEncoder
from itertools import groupby
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

def pad_seq(x, base=32):
    len_out = int(base * ceil(float(x.shape[0])/base))
    len_pad = len_out - x.shape[0]
    assert len_pad >= 0
    return np.pad(x, ((0,len_pad),(0,0)), 'constant'), len_pad

class voiceDataset(Dataset):
    wav_folder = []

    def __init__(self):
        self.wav_folder = pickle.load(open('./metadata_given.pkl', "rb"))
        np.random.shuffle(self.wav_folder)
    
    def __getitem__(self, index):
        item = dict()
        item['person'] = self.wav_folder[index][0]
        item['style'] = torch.from_numpy(self.wav_folder[index][1])
        item['spectrogram'], _ = pad_seq(self.wav_folder[index][2][:96, :]) #Crops so that every file is at most 96 long
        item['spectrogram'] = torch.from_numpy(item['spectrogram'])
        # person : p001(用來train的data)
        # style : 還沒有 Style encoder
        # spectrogram : (256, 80) 的.wav頻譜圖
        return item
    
    def __len__(self):
        return len(self.wav_folder)