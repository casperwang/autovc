#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import os
import pickle
import torch
import numpy as np
import data_loader.dataLoader as datas
from math import ceil
import pdb
from model_vc import Generator

device = 'cpu'
G = Generator(32,256,512,32).eval().to(device)

g_checkpoint = torch.load('autovc.ckpt', map_location = torch.device('cpu')) #AutoVC model weights
G.load_state_dict(g_checkpoint['model'])

data = datas.voiceDataset()
metadata = [data[0]]

spect_vc = []


def pad_seq(x, base=32):
    len_out = int(base * ceil(float(x.shape[0])/base))
    len_pad = len_out - x.shape[0]
    assert len_pad >= 0
    return np.pad(x, ((0,len_pad),(0,0)), 'constant'), len_pad


datas = pickle.load(open("./metadata.pkl", "rb"))


cnt = 0

STYLE = np.array([0.07455144, 0.        , 0.12114633, 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.13354202, 0.06428103,
       0.11323338, 0.00388092, 0.01991886, 0.        , 0.        ,
       0.00472786, 0.        , 0.        , 0.09635921, 0.        ,
       0.02857794, 0.01997942, 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.11416078, 0.07932603, 0.        , 0.        , 0.        ,
       0.        , 0.07932524, 0.        , 0.10757284, 0.        ,
       0.        , 0.        , 0.04887488, 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.23951118, 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.13313983, 0.        ,
       0.        , 0.        , 0.0476586 , 0.02469627, 0.03124238,
       0.        , 0.12234554, 0.16775778, 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.03403261, 0.08957145,
       0.        , 0.        , 0.        , 0.08680469, 0.11483785,
       0.        , 0.        , 0.05680807, 0.05188186, 0.0010662 ,
       0.        , 0.        , 0.        , 0.        , 0.2253635 ,
       0.08855635, 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.1521231 , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.0920624 , 0.18382049,
       0.        , 0.        , 0.03435336, 0.        , 0.07573951,
       0.        , 0.0212077 , 0.00221749, 0.        , 0.        ,
       0.        , 0.10780004, 0.        , 0.        , 0.20989977,
       0.09164576, 0.        , 0.07453016, 0.        , 0.0996456 ,
       0.        , 0.        , 0.06146271, 0.12226561, 0.        ,
       0.        , 0.        , 0.11276817, 0.14291915, 0.03035145,
       0.05189263, 0.01310621, 0.        , 0.12610166, 0.        ,
       0.        , 0.        , 0.02976863, 0.        , 0.        ,
       0.09120885, 0.05921955, 0.        , 0.        , 0.02345497,
       0.        , 0.12934522, 0.        , 0.        , 0.        ,
       0.06953817, 0.        , 0.        , 0.04197983, 0.11345872,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.09859626, 0.        , 0.        , 0.08322733, 0.13532178,
       0.        , 0.        , 0.        , 0.01039974, 0.        ,
       0.01795498, 0.06318911, 0.05866239, 0.01411526, 0.        ,
       0.14554003, 0.        , 0.05958405, 0.0886642 , 0.05158373,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.11440271, 0.        , 0.09032299, 0.        ,
       0.        , 0.        , 0.22942148, 0.        , 0.        ,
       0.00417617, 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.08205518,
       0.        , 0.        , 0.02919286, 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.04605015, 0.        ,
       0.        , 0.06806254, 0.        , 0.        , 0.        ,
       0.        , 0.05524325, 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.06999438, 0.        , 0.        ,
       0.09677133, 0.0486298 , 0.06449135, 0.13095243, 0.09665707,
       0.06848858, 0.07414442, 0.07359253, 0.32064927, 0.        ,
       0.01135445, 0.17175235, 0.        , 0.        , 0.16320835,
       0.        , 0.04071496, 0.01670725, 0.        , 0.10546494,
       0.        ], dtype=np.float32)


i = 0
j = 2

x_org = datas[i][2]
x_org, len_pad = pad_seq(x_org)
uttr_org = torch.from_numpy(x_org[np.newaxis, :, :]).to(device)

emb_org = torch.from_numpy(datas[i][1][np.newaxis, :]).to(device)
emb_trg = torch.from_numpy(datas[j][1][np.newaxis, :]).to(device)
#emb_trg = torch.from_numpy(STYLE[np.newaxis, :]).to(device)
#pdb.set_trace()
with torch.no_grad():
    _, x_identic_psnt, _ = G(uttr_org, emb_org, emb_trg)

uttr_trg = x_identic_psnt[0, :, :].cpu().numpy()

#spect_vc.append( ('{}x{}'.format(sbmt_i["person"], sbmt_j["person"]), uttr_trg) )
spect_vc.append( ('{}x{}'.format(datas[i][0], "p226"), uttr_trg) )

with open('results.pkl', 'wb') as handle:
    pickle.dump(spect_vc, handle)
