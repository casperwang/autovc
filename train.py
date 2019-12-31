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

iters_per_epoch = 100
learning_rate = 0.001

PATH = "./train_weights.ckpt" #To train
device = 'cpu'
G = Generator(64, 256, 512, 32).eval().to(device)
G = G.float() #Turns all weights into float weights

lmb = 1
mu = 1

doWrite = True #Turns on and off writing to TensorBoard

writer = SummaryWriter()

g_checkpoint = torch.load("./train_weights.ckpt", map_location = torch.device(device)) #the file to train
G.load_state_dict(g_checkpoint['model'])
#Will train from the same file every time, if you don't have yet make sure to just comment this out
optimizer = optim.Adam(G.parameters(), lr = learning_rate) #Not sure what the parameters do, just copying it
# optimizer.load_state_dict(g_checkpoint['optimizer'])

MSELoss = torch.nn.MSELoss()
L1Loss  = torch.nn.L1Loss()

class L_Recon(torch.nn.Module):
	def __init__(self):
		super(L_Recon, self).__init__()

	def forward(self, conv, ori):
		L_recon = MSELoss(conv, ori)
		L_recon = L_recon * L_recon #L_recon is norm squared
		return L_recon #lambda = 1

class L_Content(torch.nn.Module):
	def __init__(self):
		super(L_Content, self).__init__()

	def forward(self, convcont, oricont):
		L_content = L1Loss(convcont, oricont) #This has to be a tensor lol
		return L_content #lambda = 1

class L_Recon0(torch.nn.Module):
	def __init__(self):
		super(L_Recon0, self).__init__()

	def forward(self, oriuttr, tgtuttr):
		L_recon0 = MSELoss(oriuttr, tgtuttr) #This has to be a tensor lol
		return L_recon0 #lambda = 1


lrecon = L_Recon()
lcontent = L_Content()
lrecon0 = L_Recon0()

def pad_seq(x, base = 32):
	len_out = int(base * ceil(float(x.shape[0]) / base))
	len_pad = len_out - x.shape[0]
	assert len_pad >= 0
	return np.pad(x, ((0, len_pad), (0, 0)), "constant"), len_pad

		

def train(epochs): #TODO once data loader is complete
	#Load data -> zero gradients -> forward + backward + optimize -> perhaps print stats?
	total_it = 0
	datas = data.voiceDataset()
	sz = len(datas)
	print("Dataset Size : ", sz)
	for epoch in range(epochs):
		running_loss = 0
		for it in tqdm(range(iters_per_epoch)):
			total_it = total_it + 1
			i = np.random.randint(0, sz)
			j = np.random.randint(0, sz)
			while(i == j):
				j = np.random.randint(0, sz)

			datai = datas[i]
			dataj = datas[j]
			
			x_org, _ = pad_seq(datai[2])
			uttr_org =  torch.from_numpy(x_org[np.newaxis, :, :]).cpu().float()
			emb_org = torch.from_numpy(datai[1][np.newaxis, :]).cpu().float()
			emb_trg = torch.from_numpy(dataj[1][np.newaxis, :]).cpu().float()
			#use i's content and j's style

			mels, mel_postnet, _ = G(uttr_org, emb_org, emb_trg)
			
			uttr_trg  = mel_postnet[0, 0, :, :].cpu()
			uttr_trg0 = mels[0, 0, :, :].cpu()

			uttr_trg  = uttr_trg[np.newaxis, :].cpu().float()
			uttr_trg0 = uttr_trg0[np.newaxis, :].cpu().float()
			content_org = Variable(torch.cat(G.encoder(uttr_org, emb_org)), requires_grad=True) #It's a list of tensors 
			content_trg = Variable(torch.cat(G.encoder(uttr_trg, emb_org)), requires_grad=True)

			uttr_org  = Variable(uttr_org , requires_grad=True)
			uttr_trg  = Variable(uttr_trg , requires_grad=True)
			uttr_trg0 = Variable(uttr_trg0, requires_grad=True)

			optimizer.zero_grad()

			l_recon = lrecon(uttr_org, uttr_trg)
			l_content = lcontent(content_org, content_trg)
			l_recon0 = lrecon0(uttr_trg, uttr_trg0)

			#loss = criterion(uttr_trg, uttr_org, content_trg, content_org)
			loss = l_recon + l_content * lmb + l_recon0 * mu

			loss.backward()
			optimizer.step()
			if(doWrite == True):
				writer.add_scalar("Loss", loss.item(), total_it)

			running_loss += loss.item()

		print("Epoch: " + (str)(epoch) + ", avg loss = " + (str)(running_loss / iters_per_epoch))
		torch.save({
			"epoch": epoch,
			"model": G.state_dict(),
			"optimizer": optimizer.state_dict()
		}, PATH)

train(1000)

		

#TODO: 
# 1. Data Loader - Wav File -> Turn into Mel-Spectrogram -> Turn Spectrogram into 
#    1) init 
#	 2) getelemnt  
# 2. Training: Connect the dots, add Loss function, see pytorch example for trainer 
#    Build Model ->
#    def criterion()... for loss function 

