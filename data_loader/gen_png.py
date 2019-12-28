"""generate mel-spectrogram from wav"""
import librosa
from scipy import misc
import pickle
import numpy as np
import audio
import hparams_gen_melspec as hparams
import matplotlib.pyplot as plt
import os
import glob
from tqdm import tqdm
wavs = []
for i in range(1, 2):
	wavs.append('./BZNSYP.rar/Wave/'+str(i).zfill(6)+'.wav')
write_path = './'
for wav_path in tqdm(wavs):

	basename=os.path.basename(wav_path).split('.wav')[0]
	wav = audio.load_wav(wav_path)
	wav = wav / np.abs(wav).max() * hparams.hparams.rescaling_max

	out = wav
	constant_values = 0.0
	out_dtype = np.float32

	mel_spectrogram = audio.melspectrogram(wav).astype(np.float32).T
    # print(mel_spectrogram.shape)

	plt.figure(figsize=(10, 4))
	S_dB = librosa.power_to_db(mel_spectrogram, ref=np.max)
	librosa.display.specshow(S_dB, x_axis='time',
                            y_axis='mel', sr=sr,
                            fmax=8000)
	plt.colorbar(format='%+2.0f dB')
	plt.title('Mel-frequency spectrogram')
	plt.tight_layout()
	plt.show()


	misc.imsave(os.path.join(write_path,basename+'.png'),mel_spectrogram)