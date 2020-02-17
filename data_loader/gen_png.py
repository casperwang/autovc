"""Generate mel-spectrogram from wav"""
import librosa
from scipy import misc
import pickle
import numpy as np
import audio
import hparams_gen_melspec as hparams
import matplotlib.pyplot as plt
import os
import glob
import dataLoader as DataLoader
from tqdm import tqdm

wavs = []

wavs.append('./Adele - Hello.wav') # file name

write_path = './'

for wav_path in tqdm(wavs):

	basename = os.path.basename(wav_path).split('.wav')[0]
	wav = audio.load_wav(wav_path)
	wav = wav / np.abs(wav).max() * hparams.hparams.rescaling_max

	mel_spectrogram = audio.melspectrogram(wav).astype(np.float32).T

	misc.imsave(os.path.join(write_path,basename+'.png'),mel_spectrogram)