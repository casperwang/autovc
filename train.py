import numpy as np
from model_vc import Generator
import torch
import torch.optim as optim
import model_vc as models
import torch.functional as F
import data_loader.dataLoader as datas

iters_per_epoch = 100

PATH = "./train_weights.ckpt"
device = "cpu"
G = Generator(32, 256, 512, 32).eval().to(device)

g_checkpoint = torch.load("trainchk.ckpt", map_location = torch.device(device)) #trainchk.ckpt is the file to train

G.load_state_dict(g_checkpoint["model"])
optimizer = optim.Adam(G.parameters(), lr = 0.0001) #Not sure what the parameters do, just copying it


def pad_seq(x, base = 32):
	len_out = int(base * ceil(float(x.shape[0]) / base))
	len_pad = len_out - x.shape[0]
	assert len_pad >= 0
	return np.pad(x, ((0, len_pad), (0, 0)), "constant"), len_pad

def criterion(self, conv, ori, convcont, oricont): #TODO: Don't have L_recon0 yet, conv = converted
		L_recon = np.linalg.norm(conv - ori)
		L_recon = L_recon * L_recon #L_recon is norm squared
		L_content = np.linalg.norm(convcont - oricont)
		return L_recon + L_content #lambda = 1

def train(epochs): #TODO once data loader is complete
	total_it = 0
	sz = datas.len()
	for epoch in range(epochs):
		for it in range(iters_per_epoch):
			total_it = total_it + 1
			i = np.randint(0, sz)
			j = np.randint(0, sz)
			while(i == j):
				j = np.randint(0, sz)
			datai = datas.get_item(i)
			dataj = datas.get_item(j)

			x_org = datai[2]
			x_org, len_pad = pad_seq(x_org)
			uttr_org =  torch.from_numpy(x_org[np.newaxis, :, :]).to(device)

			emb_org = torch.from_numpy(datai[1][np.newaxis, :]).to(device)
			emb_trg = torch.from_numpy(dataj[1][np.newaxis, :]).to(device)
			#use i's content and j's style

			with torch.no_grad():
				mels, mel_postnet, codes = G(datai, dataj)
			
			if len_pad == 0:
				uttr_trg = mel_postnet[0, 0, :, :].cpu().numpy()
			else:
				uttr_trg = mel_postnet[0, 0, :-len_pad, :].cpu().numpy()
			content_org = G.encoder(uttr_org, emb_org)
			content_trg = G.encoder(uttr_trg, emb_org)

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

