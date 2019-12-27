import pickle
from sklearn import preprocessing
import numpy as np

class Dataset:
    wav_folder = []
    def __init__(self):
        self.wav_folder = pickle.load(open('./data_loader/test.pkl', "rb"))
    def get_item(self, index):
        item = ['p001']
        tmp = np.arange(256, dtype='float64')
        tmp[0] = 1
        item.append(tmp)
        item.append(preprocessing.scale(np.resize(self.wav_folder[index][1], (256, 80)).astype('float64')))
        return item
    def len(self):
        return len(self.wav_folder)
