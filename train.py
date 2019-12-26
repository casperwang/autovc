import numpy as np
from model_vc import Generator
import torch
import torch.optim as optim
import model_vc as models
import torch.functional as F
import dataLoader as datas
model = sy.build_model()

iters_per_epoch = 100

device = "cpu"
G = Generator(32, 256, 512, 32).eval().to(device)

g_checkpoint = torch.load("trainchk.ckpt", map_location = torch.device(device)) #trainchk.ckpt is the file to train

G.load_state_dict(g_checkpoint["model"])

class Trainer(nn.Module):
	def __init__(self): #Build model here
		super(Trainer, self).__init__()

	def oneHot(name): #Basic one-hot encoder for style 
		res = np.zeros(256, 1)
		if name == "Sean":
			res[0][0] = 1
		elif name == "Casper":
			res[1][0] = 1
		elif name == "BoEn":
			res[2][0] = 1
		elif name == "DeiZhen":
			res[3][0] = 1
		elif name == "PingXiang":
			res[4][0] = 1

		return res
	def forward(self, x):

def criterion(self, conv, ori, convcont, oricont): #TODO: Don't have L_recon0 yet
		L_recon = np.linalg.norm(conv - ori)
		L_recon = L_recon * L_recon #L_recon is norm squared
		L_content = np.linalg.norm(convcont - oricont)
		return L_recon + L_content #lambda = 1

trainer = Trainer()

optimizer = optim.Adam(trainer.parameters(), lr = 0.0001) #Not sure what the parameters do, just copying it

def train(epochs): #TODO once data loader is complete
	total_it = 0
	for epoch in range(epochs):
		for it in range(iters_per_epoch):
			total_it = total_it + 1

		#Load data -> zero gradients -> forward + backward + optimize -> perhaps print stats?
		

#TODO: 
# 1. Data Loader - Wav File -> Turn into Mel-Spectrogram -> Turn Spectrogram into 
#    1) init
#	 2) getelemnt  
# 2. Training: Connect the dots, add Loss function, see pytorch example for trainer 
#    Build Model ->
#    def criterion()... for loss function 

