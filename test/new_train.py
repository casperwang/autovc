import numpy as np
from model_vc import Generator
from styleencoder import StyleEncoder
import torch.autograd as autograd
from torch.autograd import Variable
from resemblyzer import VoiceEncoder, preprocess_wav
from math import ceil
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.optim as optim
import model_vc as models
from tqdm import tqdm
import torch.functional as F
import data_loader.dataLoader as data
import pdb
import atexit

#constants 
learning_rate = 0.0001
batch_size = 2
lmb = 1
mu = 1
bottle_neck = 32
dim_style = 256
dim_pre = 512
freq = 32

#other consts 
save_path = "./train_weights.ckpt"
load_path = "./autovc.ckpt"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") #Uses GPU when available
write_to_tensorboard = True
writer = SummaryWriter() #This writes to tensorboard

#Init generator, optimizer 
G = Generator(bottle_neck, dim_style, dim_pre, freq).train().to(device).float()
g_checkpoint = torch.load(load_path, map_location = torch.device(device)) #Load from
optimizer = optim.Adam(G.parameters(), lr = learning_rate) #Not sure what the parameters do, just copying it
G.load_state_dict(g_checkpoint['model'])
optimizer.load_state_dict(g_checkpoint['optimizer'])

#Voice Encoder (Resemblyzer)
styleEncoder = VoiceEncoder()

#Init loss functions
MSELoss = torch.nn.MSELoss()
L1Loss  = torch.nn.L1Loss()

def pad_seq(x, base = 32):
	len_out = int(base * ceil(float(x.shape[0]) / base))
	len_pad = len_out - x.shape[0]
	assert len_pad >= 0
	return np.pad(x, ((0, len_pad), (0, 0)), "constant"), len_pad

def train_one_epoch(model, dataset, save_path, current_iter, doWrite = True): #Takes a PyTorch DataLoader as input and 
	#model: 		the model that you wish to train
	#dataset: 		a PyTorch DataLoader that can be enumerated
	#save_path: 	where to save the training weights
	#current_iter: 	what iteration it's currently on (running total)
	#doWrite: 		whether to write to tensorboard or not 
	for i, datai in enumerate(tqdm(dataset)):
		
		#datai: B * C * T * F
		current_iter = current_iter + 1

		uttr_org  = preprocess_wav(datai["content"]) #nparray
		uttr_tgt  = preprocess_wav(datai["style"]  ) #nparray

		style_org = [] 
		style_tgt = []

		for uttr in datai["content"]:
			style_org.append(styleEncoder.embed_utterance(preprocess_wav(uttr)))

		for uttr in datai["style"]:
			style_org.append(styleEncoder.embed_utterance(preprocess_wav(uttr)))



