import numpy as np
from new_model import Generator
import torch.autograd as autograd
from torch.autograd import Variable
from resemblyzer import VoiceEncoder, preprocess_wav
from math import ceil
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.optim as optim
import new_model as models
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

def train_one_epoch(model, dataset, save_dir, current_iter, doWrite = True): #Takes a PyTorch DataLoader as input and 
	#model: 		the model that you wish to train
	#dataset: 		a PyTorch DataLoader that can be enumerated
	#save_dir: 		directory to save the training weights
	#current_iter: 	what iteration it's currently on (running total)
	#doWrite: 		whether to write to tensorboard or not 
	running_loss = 0
	for i, datai in enumerate(tqdm(dataset)):

		#datai: B * C * T * F

		datai.squeeze(1) #Gets rid of Channel dimension

		current_iter = current_iter + 1

		uttr_org  = [] #
		uttr_tgt  = [] #This and the above will be B * T * F
		style_org = [] #
		style_tgt = [] #This and the above will be B * 1 * dim_style

		for uttr in datai["content"]:
			style_org.append(styleEncoder.embed_utterance(preprocess_wav(uttr)))
			uttr_org.append(preprocess_wav(uttr))

		for uttr in datai["style"]:
			style_trg.append(styleEncoder.embed_utterance(preprocess_wav(uttr)))
			uttr_trg.append(preprocess_wav(uttr))

		#Turn everything into PyTorch Tensors, and gives the outputs to device
		uttr_org  = torch.from_numpy(uttr_org).to(device)
		uttr_trg  = torch.from_numpy(uttr_trg).to(device)
		style_org = torch.from_numpy(style_org).to(device)
		style_trg = torch.from_numpy(style_trg).to(device)


		mel_outputs, mel_outputs_postnet, codes = G(uttr_org, emb_org, emb_trg)
		_, _, trg_codes = G(mel_outputs, emb_trg, uttr_org)
		#mel_outputs: 			the output sans postnet
		#mel_outputs_postnet: 	the above with postnet added
		#codes:					encoder output	


		#Again, get rid of channel dimension
		mel_outputs.squeeze(1)
		mel_outputs_postnet.squeeze(1)
		codes.squeeze(1)

		#Zero gradients
		optimizer.zero_grad()
		#Calculate Loss
		L_Recon = MSELoss(mel_outputs_postnet, uttr_trg)
		L_Recon0 = MSELoss(mel_outputs, uttr_trg)
		L_Content = L1Loss(codes, trg_codes)

		loss = L_Recon + mu * L_Recon0 + lmb * L_Content

		loss.backward()
		optimizer.step()

		running_loss += loss.item()

		if(doWrite == True):
				writer.add_scalar("Loss", loss.item(), current_iter)

		
		if current_iter % 100 == 99:
			torch.save({
				"epoch": epoch,
				"model": G.state_dict(),
				"optimizer": optimizer.state_dict()
			}, save_dir + "/test_ckpt_{}epo.ckpt".format(epoch))


		return current_iter
