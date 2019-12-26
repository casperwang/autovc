import numpy as np
from model_vc import Generator
from math import ceil
import torch
import torch.optim as optim
import model_vc as models
import torch.functional as F
import data_loader.dataLoader as data

iters_per_epoch = 100

PATH = "./train_weights.ckpt" #To train
device = "cpu"
G = Generator(32, 256, 512, 32).eval().to(device)
G = G.float()

g_checkpoint = torch.load("autovc.ckpt", map_location = torch.device(device)) #trainchk.ckpt is the file to train

optimizer = optim.Adam(G.parameters(), lr = 0.0001) #Not sure what the parameters do, just copying it


def pad_seq(x, base = 32):
	len_out = int(base * ceil(float(x.shape[0]) / base))
	len_pad = len_out - x.shape[0]
	assert len_pad >= 0
	return np.pad(x, ((0, len_pad), (0, 0)), "constant"), len_pad

def criterion(conv, ori, convcont, oricont): #TODO: Don't have L_recon0 yet, conv = converted
		L_recon = np.linalg.norm(conv - ori)
		L_recon = L_recon * L_recon #L_recon is norm squared
		L_content = np.linalg.norm(convcont - oricont)
		return L_recon + L_content #lambda = 1

def train(epochs): #TODO once data loader is complete
	total_it = 0
	datas = data.Dataset()
	sz = datas.len()
	for epoch in range(epochs):
		for it in range(iters_per_epoch):
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
			print(uttr_trg.shape)
			content_org = array(G.encoder(uttr_org, emb_org))
			content_trg = array(G.encoder(uttr_trg, emb_org))

			loss = criterion(uttr_trg, uttr_org, content_org, content_trg)
			loss.backward()
			optimizer.step()
		print("Epoch: " + epoch + ", loss = " + loss.item())
		torch.save({
			"epoch": epoch,
			"model": G.state_dict(),
			"optimizer": optimizer.state_dict()
		}, PATH)

train(2)

		#Load data -> zero gradients -> forward + backward + optimize -> perhaps print stats?
		

#TODO: 
# 1. Data Loader - Wav File -> Turn into Mel-Spectrogram -> Turn Spectrogram into 
#    1) init
#	 2) getelemnt  
# 2. Training: Connect the dots, add Loss function, see pytorch example for trainer 
#    Build Model ->
#    def criterion()... for loss function 

