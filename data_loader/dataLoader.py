import pickle
import numpy as np

class Dataset:
    wav_folder = []
    def __init__(self):
        self.wav_folder = pickle.load(open('BZNSYP.pkl', "rb"))
    def get_item(self, index):
        item = ['p001']
        item.append([0]*255)
        item[1][0] = 1
        item.append(np.resize(self.wav_folder[index][1], (256, 80)))
        return item
    def get_len(self):
        return len(self.wav_folder)