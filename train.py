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

learning_rate = 0.0001


batch_size = 2

PATH = "./train_weights.ckpt" #Save to
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") #Uses GPU when available
G = Generator(32, 256, 512, 32).train().to(device)
G = G.float() #Turns all weights into float weights

lmb = 1
mu = 1

doWrite = True #Turns on and off writing to TensorBoard

writer = SummaryWriter()

g_checkpoint = torch.load("./autovc.ckpt", map_location = torch.device(device)) #Load from
G.load_state_dict(g_checkpoint['model'])
#Will train from the same file every time, if you don't have yet make sure to just comment this out
optimizer = optim.Adam(G.parameters(), lr = learning_rate) #Not sure what the parameters do, just copying it
optimizer.load_state_dict(g_checkpoint['optimizer'])
styleEncoder = VoiceEncoder()


MSELoss = torch.nn.MSELoss()
L1Loss  = torch.nn.L1Loss()

def pad_seq(x, base = 32):
	len_out = int(base * ceil(float(x.shape[0]) / base))
	len_pad = len_out - x.shape[0]
	assert len_pad >= 0
	return np.pad(x, ((0, len_pad), (0, 0)), "constant"), len_pad

def train(epochs): #TODO once data loader is complete
	#Load data -> zero gradients -> forward + backward + optimize -> perhaps print stats?
	total_it = 0
	datas = data.voiceDataset()
	dataset = torch.utils.data.DataLoader(datas, batch_size = batch_size, shuffle = True)
	
	for epoch in range(epochs):
		running_loss = 0
		
		for i, datai in enumerate(tqdm(dataset)):
			total_it = total_it + 1
			
			uttr_org = datai["spectrogram"] #is a numpy array
			emb_trg = emb_org = datai["style"]
			#use i's content and j's style

			mels, mel_postnet, _ = G(uttr_org, emb_org, emb_trg)
			mel_postnet = mel_postnet.squeeze(0)
			#pdb.set_trace()
			#print("Getting contents")
			content_org = Variable(torch.cat(G.encoder(uttr_org, emb_org)), requires_grad=True) #It's a list of tensors 
			#print("Getting content_trg")
			content_trg = Variable(torch.cat(G.encoder(mel_postnet, emb_org)), requires_grad=True)

			optimizer.zero_grad()

			l_recon = MSELoss(uttr_org, mel_postnet)
			l_content = L1Loss(content_org, content_trg)
			l_recon0 = MSELoss(uttr_org, mels)

			#loss = criterion(uttr_trg, uttr_org, content_trg, content_org)
			loss = l_recon + l_content * lmb + l_recon0 * mu

			loss.backward()
			optimizer.step()
			if(doWrite == True):
				writer.add_scalar("Loss", loss.item(), total_it)

			running_loss += loss.item()

		if (epoch % 5 == 4):
			print("Saving on Epoch " + str(epoch))
			torch.save({
				"epoch": epoch,
				"model": G.state_dict(),
				"optimizer": optimizer.state_dict()
			}, "./test_ckpt_{}epo.ckpt".format(epoch))
		print("Avg loss = " + str(running_loss/4))
train(1000)

#TODO:
# 1. Data Loader - Wav File -> Turn into Mel-Spectrogram -> Turn Spectrogram into
#	1) init
#	2) getelemnt
# 2. Training: Connect the dots, add Loss function, see pytorch example for trainer
#	Build Model ->
#	def criterion()... for loss function

