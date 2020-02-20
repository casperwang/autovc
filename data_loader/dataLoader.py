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
<<<<<<< HEAD
        item['person'] = 'p001'
        tmp = np.zeros((256), dtype='float64')
        tmp = np.array([-2.92885415e-02,  1.67739280e-02,  6.42375797e-02,  5.12384847e-02,
        8.77934247e-02, -4.05867286e-02,  1.08849108e-02,  2.06673276e-02,
       -8.39690343e-02,  1.48199843e-02,  3.82344425e-02,  1.12243919e-02,
        4.62971367e-02,  1.66661311e-02, -5.78785129e-02,  3.60288732e-02,
        1.92339886e-02,  9.17971320e-03,  2.74707917e-02, -5.48039749e-02,
        2.50798557e-02,  4.66737375e-02, -6.14981353e-03,  3.88026945e-02,
        5.68139665e-02, -7.33052716e-02,  2.30920967e-02, -1.04292825e-01,
       -2.61898227e-02,  3.02257240e-02, -3.02889403e-02,  3.63447554e-02,
       -4.97230627e-02,  1.57715172e-01, -3.93295921e-02,  5.51161245e-02,
        4.64604087e-02, -4.59927395e-02, -4.82378080e-02, -3.81431282e-02,
        3.91379185e-02,  4.59317304e-02, -1.55072343e-02, -1.67513415e-02,
       -5.09507731e-02, -5.08496165e-02, -2.25679334e-02,  2.63878461e-02,
       -1.26611767e-02, -2.49883570e-02,  3.02621610e-02, -2.48487398e-04,
       -1.22652270e-01,  1.32972272e-02,  2.11958797e-03,  1.50569994e-02,
        1.89354662e-02,  5.65703809e-02, -7.35055134e-02,  5.03348000e-02,
        4.49209325e-02,  1.92234237e-02, -5.00078090e-02, -6.60179332e-02,
       -5.14238365e-02,  8.59818831e-02,  5.63113764e-02, -5.70376217e-03,
       -4.67663854e-02, -1.23205269e-02, -2.38060020e-02, -9.18019041e-02,
       -5.92616424e-02,  5.22904918e-02,  9.33921803e-03,  4.16528508e-02,
       -1.82117913e-02,  1.43172365e-04, -1.57222264e-02, -6.38223216e-02,
        1.35664223e-02, -3.96627523e-02, -9.89069976e-03,  5.36250416e-04,
        4.61532697e-02,  4.76618186e-02, -9.83480811e-02, -9.85761583e-02,
        8.16847682e-02, -3.19430546e-04, -9.46291238e-02,  6.24535158e-02,
        2.11758353e-03, -2.45369296e-03, -1.51609881e-02,  3.43220234e-02,
       -4.69367653e-02,  4.89003994e-02,  2.89346427e-02,  6.56676441e-02,
        3.93782370e-02,  6.51109871e-03, -2.50044372e-02,  2.59851553e-02,
       -5.71040809e-02,  1.27174407e-02, -1.64750479e-02, -3.01102847e-02,
        5.86121492e-02, -2.10250281e-02,  4.84444983e-02,  5.18408697e-03,
       -1.56546421e-02, -8.27844627e-03, -3.14547382e-02, -1.05279163e-02,
       -2.88096778e-02,  9.46586728e-02,  2.97741927e-02,  4.04976979e-02,
       -7.98548106e-03, -4.78080586e-02,  5.34602478e-02,  1.44058481e-01,
       -4.82284762e-02,  2.25759130e-02, -4.55716662e-02, -3.33905481e-02,
       -4.24340665e-02, -3.73299164e-03, -6.99824616e-02, -5.04204035e-02,
        1.85231529e-02,  9.56422314e-02, -4.30788696e-02,  5.86966577e-04,
        3.88075039e-02,  8.25675353e-02, -5.59520442e-03, -3.62152234e-02,
        1.92788001e-02,  6.00103289e-04, -6.25425130e-02, -3.70997600e-02,
       -2.17933208e-02, -4.99033928e-03,  9.04327258e-02, -2.94539090e-02,
       -4.26172428e-02,  1.58391304e-05, -6.22648001e-02,  1.66727230e-02,
       -2.02247333e-02, -2.17881799e-02, -3.11419629e-02,  1.76396538e-02,
       -1.10096768e-01, -2.11662687e-02, -5.71037736e-03, -1.02486178e-01,
       -5.27596101e-02, -3.53671238e-02, -5.02373278e-02,  1.64796822e-02,
       -6.63699657e-02, -6.01356328e-02,  5.79667799e-02,  7.97191542e-03,
       -6.97814161e-03,  2.12903805e-02,  3.57458070e-02, -1.14577964e-01,
        2.11331956e-02, -1.75991841e-02, -3.61673534e-02, -1.36390496e-02,
        1.92440730e-02, -1.31492633e-02,  4.22024317e-02, -3.43342759e-02,
        9.27030519e-02,  4.51238680e-04, -2.20819637e-02, -2.34996732e-02,
        1.19429111e-01, -1.27005447e-02, -3.02130841e-02,  2.79148128e-02,
        4.97458354e-02, -5.76778725e-02, -1.13642467e-02, -1.10709630e-02,
        2.69861682e-03,  8.76095891e-02,  6.28586709e-02, -4.06131484e-02,
        5.22754267e-02, -3.75854596e-03, -1.68104116e-02,  4.15816531e-02,
       -4.03943919e-02, -8.06394666e-02,  6.36994355e-06,  4.52433303e-02,
        1.68162603e-02,  6.27513826e-02,  7.32335914e-03,  2.77076457e-02,
       -2.71280501e-02, -3.18220854e-02, -8.10135603e-02, -4.01202366e-02,
        7.98700005e-02,  7.26153627e-02, -2.01774240e-02, -2.79837642e-02,
       -6.85842186e-02, -7.13655502e-02, -1.46079659e-02,  1.25141889e-01,
       -5.65150194e-03,  1.86081380e-02,  4.72205058e-02, -7.26464093e-02,
        8.40771720e-02, -7.62810418e-03,  4.14126590e-02,  1.04194321e-01,
       -5.16419113e-02, -3.93198840e-02,  9.98969376e-03,  1.57634858e-02,
        4.67797332e-02, -1.35338558e-02,  1.76969264e-02,  6.15395121e-02,
       -2.92499410e-03,  3.32759246e-02, -5.93688786e-02, -3.37226577e-02,
        3.41136232e-02,  4.35125194e-02,  2.47878209e-03,  6.47273436e-02,
        1.85277015e-02, -5.58293797e-03,  1.44551052e-02, -6.20163754e-02,
       -8.78251530e-03,  8.14331546e-02, -6.42475262e-02, -1.85454320e-02,
        5.63065596e-02, -1.04957800e-02,  1.53429583e-02,  2.60114018e-02])
        item['style'] = torch.from_numpy(tmp).cpu().float()
        item['spectrogram'] = torch.from_numpy(self.wav_folder[index][1][np.newaxis, :, :]).cpu().float()
=======
        item['person'] = self.wav_folder[index][0]
        item['style'] = torch.from_numpy(self.wav_folder[index][1])
        item['spectrogram'], _ = pad_seq(self.wav_folder[index][2][:96, :]) #Crops so that every file is at most 96 long
        item['spectrogram'] = torch.from_numpy(item['spectrogram'])
>>>>>>> bf8639fcdc54569acc99420c80e9011be0d59882
        # person : p001(用來train的data)
        # style : 還沒有 Style encoder
        # spectrogram : (256, 80) 的.wav頻譜圖
        return item
    
    def __len__(self):
        return len(self.wav_folder)