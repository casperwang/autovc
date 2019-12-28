"""generate mel-spectrogram from wav"""
import librosa
from scipy import misc
import pickle
import numpy as np
import audio
import hparams_gen_melspec as hparams
import os
import glob
from tqdm import tqdm

wavs = []
mels = []

for i in range(1, 2000):
	wavs.append('./BZNSYP.rar/Wave/'+str(i).zfill(6)+'.wav')

write_path = './'
for wav_path in tqdm(wavs):

	basename = os.path.basename(wav_path).split('.wav')[0]
	wav = audio.load_wav(wav_path)
	wav = wav / np.abs(wav).max() * hparams.hparams.rescaling_max

	out = wav
	constant_values = 0.0
	out_dtype = np.float32

	mel_spectrogram = audio.melspectrogram(wav).astype(np.float32).T

	mels.append((basename,mel_spectrogram))

with open(os.path.join(write_path,'test.pkl'),'wb') as handle:
	pickle.dump(mels, handle)

