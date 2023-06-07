import os
import torch
import torchaudio
import pandas as pd
import numpy as np

class RavdessDataset(object):
    """
        Create a Dataset for Ravdess. Each item is a tuple of the form:
        (waveform, sample_rate, emotion)
    """

    _ext_audio = '.wav'
    _emotions = { 'neu': 0, 'cal': 1, 'hap': 2, 'sad': 3, 'ang': 4, 'fea': 5, 'dis': 6, 'sur': 7}

    def __init__(self, root='Audio_Speech_Actors_01-24'):
        """
        Args:
            root (string): Directory containing the Actor folders
        """
        self.root = root

        # Iterate through all audio files
        data = []
        for _, _, files in os.walk(root):
            for file in files:
                if file.endswith(self._ext_audio):
                    # Truncate file extension and split filename identifiers
                    identifiers = file[:-len(self._ext_audio)].split('-')

                    # Append file path w.r.t to root directory
                    identifiers.append(os.path.join('Actor_' + identifiers[-1], file))

                    # Append identifier to data
                    data.append(identifiers)

        # Create pandas dataframe
        self.df = pd.DataFrame(data, columns=['modality', 'vocal_channel', 'emotion', 'intensity', 'statement', 'repetition', 'actor', 'file'], dtype=np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        audio_name = os.path.join(self.root, self.df.loc[idx, 'file'])
        waveform, sample_rate = torchaudio.load(audio_name)
        #print(waveform.shape)
        waveform = torchaudio.functional.resample(waveform, orig_freq=sample_rate, new_freq=16000)
        sample_rate = 16000
        #print(waveform.shape)
        emotion = int(self.df.loc[idx, 'emotion']-1)
        sample = {
            'waveform': waveform,
            'sample_rate': sample_rate,
            'emotion': emotion
        }

        return sample