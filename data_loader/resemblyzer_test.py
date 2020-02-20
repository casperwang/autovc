#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 14:47:23 2020

@author: liusean
"""

import os

os.environ['KMP_DUPLICATE_LIB_OK']='True' #MacOS will return error if this line is not added
                                          #If will return Error #15: Initializing libiomp5.dylib, but found libiomp5.dylib already initialized

from resemblyzer import preprocess_wav, VoiceEncoder
from itertools import groupby
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

encoder = VoiceEncoder()

wav_fpaths = list(Path().rglob("p226_DEMO.wav"))

speaker_wav = preprocess_wav(wav_fpaths[0])
embed = encoder.embed_utterance(speaker_wav)
print("Done")
#embed = encoder.embed_utterance(speaker_wav)

#print("Finished embedding: shape = " + embed.shape)