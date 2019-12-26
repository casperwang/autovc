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

PATH = "./train_weights.ckpt" #To train
device = "cpu"
G = Generator(32, 256, 512, 32).eval().to(device)
G = G.float() #Turns all weights into float weights 
writer = SummaryWriter()


g_checkpoint = torch.load("./train_weights.ckpt", map_location = torch.device(device)) #the file to train
G.load_state_dict(g_checkpoint['model'])
#Will train from the same file every time, if you don't have yet make sure to just comment this out
optimizer = optim.Adam(G.parameters(), lr = 0.1) #Not sure what the parameters do, just copying it

class LossFunction(torch.nn.Module):
	def __init__(self):
		super(LossFunction, self).__init__()

	def forward(self, conv, ori, convcont, oricont):
		L_recon = torch.dist(conv, ori)
		L_recon = L_recon * L_recon #L_recon is norm squared
		L_content = torch.dist(convcont, oricont) #This has to be a tensor lol
		return L_recon + L_content #lambda = 1

criterion = LossFunction()

def pad_seq(x, base = 32):
	len_out = int(base * ceil(float(x.shape[0]) / base))
	len_pad = len_out - x.shape[0]
	assert len_pad >= 0
	return np.pad(x, ((0, len_pad), (0, 0)), "constant"), len_pad

		

def train(epochs): #TODO once data loader is complete
	#Load data -> zero gradients -> forward + backward + optimize -> perhaps print stats?
	total_it = 0
	datas = data.Dataset()
	sz = datas.len()
	for epoch in range(epochs):
		for it in tqdm(range(iters_per_epoch)):
			total_it = total_it + 1
			i = np.random.randint(0, sz)
			j = np.random.randint(0, sz)
			while(i == j):
				j = np.random.randint(0, sz)

			datai = datas.get_item(i)
			dataj = datas.get_item(j)

			x_org = datai[2]
			
			x_org, len_pad = pad_seq(x_org)
			uttr_org =  torch.from_numpy(x_org[np.newaxis, :, :]).to(device).float()
			emb_org = torch.from_numpy(datai[1][np.newaxis, :]).to(device).float()
			emb_trg = torch.from_numpy(dataj[1][np.newaxis, :]).to(device).float()
			#use i's content and j's style

			with torch.no_grad():
				mels, mel_postnet, codes = G(uttr_org, emb_org, emb_trg)
			
			if len_pad == 0:
				uttr_trg = mel_postnet[0, 0, :, :].cpu().numpy()
			else:
				uttr_trg = mel_postnet[0, 0, :-len_pad, :].cpu().numpy()
			uttr_trg = torch.from_numpy(uttr_trg[np.newaxis, :]).to(device).float()
			content_org = Variable(torch.cat(G.encoder(uttr_org, emb_org)), requires_grad=True) #It's a list of tensors 
			content_trg = Variable(torch.cat(G.encoder(uttr_trg, emb_org)), requires_grad=True)			

			uttr_org = Variable(uttr_org, requires_grad=True)
			uttr_trg = Variable(uttr_trg, requires_grad=True)

			loss = criterion(uttr_trg, uttr_org, content_trg, content_org)
			loss.backward()
			optimizer.step()
			writer.add_scalar("Loss", loss.item(), total_it)
		print("Epoch: " + (str)(epoch) + ", loss = " + (str)(loss.item()))
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

