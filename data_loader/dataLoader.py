import gen_mel

class Dataset:
    wav_folder = []
    def __init__(self):
        return NULL
    def get_item(self, index):
        y, sr = librosa.load(librosa.util.example_audio_file())
        return librosa.feature.melspectrogram(y=y, sr=sr)