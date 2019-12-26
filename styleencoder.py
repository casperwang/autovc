import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class StyleEncoder(nn.Module): #TODO
    def __init__(self, dim_neck, dim_emb, dim_pre):
        super(StyleEncoder, self).__init__()
        self.lstm = nn.LSTM(768, dim_neck, 2, batch_first=True, bidirectional=True)
        self.fully_connected = nn.Linear(768, 256)
    def forward(x, self):
    	_, x = self.lstm(x)
    	x = self.fully_connected(x)
    	return x
        