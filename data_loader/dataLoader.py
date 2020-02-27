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
    iter_folder = []

    def __init__(self):
        self.iter_folder = pickle.load(open('./iters.pkl', "rb"))
        self.wav_folder = pickle.load(open('./data.pkl', "rb"))
    
    def __getitem__(self, index): #Should iterate through all possible triples
        item = dict()
        idx = self.iter_folder[index][i]
        item['person'] = idx
        item['style'] = torch.from_numpy(pad_seq(self.wav_folder[idx][self.iter_folder[index][j]][:96, :]))
        item['content'], _ = torch.from_numpy(pad_seq(self.wav_folder[idx][self.iter_folder[index][k]][:96, :]))
        return item
    
    def __len__(self):
        return len(self.wav_folder)