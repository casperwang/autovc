import pickle
from sklearn import preprocessing
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class voiceDataset(Dataset):
    wav_folder = []

    def __init__(self):
        self.wav_folder = pickle.load(open('./data_loader/data.pkl', "rb"))
        np.random.shuffle(self.wav_folder)
    
    def __getitem__(self, index):
        item = dict()
        item['person'] = 'p001'
        tmp = np.zeros((256), dtype='float64')
        tmp[0] = 1
        item['style'] = torch.from_numpy(tmp).cpu().float()
        item['spectrogram'] = torch.from_numpy(self.wav_folder[index][1][np.newaxis, :, :]).cpu().float()
        # person : p001(用來train的data)
        # style : 還沒有 Style encoder
        # spectrogram : (256, 80) 的.wav頻譜圖
        return item
    
    def __len__(self):
        return len(self.wav_folder)