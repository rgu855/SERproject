import os
import torch
import torchaudio
import pandas as pd
import numpy as np

class UrduDataset(object):
    """
        Create a Dataset for URDU. Each item is a tuple of the form:
        (waveform, sample_rate, emotion)
    """
    _ext_audio = '.wav'
    _emotions = { 'A': 0, 'H': 1, 'N': 2, 'S': 3} # W = angery, H = happy, N = neutral, S = sad
    _file_emotions = {'A':'Angry', 'H': 'Happy', 'N':'Neutral', 'S':'Sad'}
    def __init__(self, root="URDU-Dataset-master"):
        self.root = root

        # Iterate through all audio files
        data = []
        for _, _, files in os.walk(root):
            for file in files:
                if file.endswith(self._ext_audio):
                    # Construct file identifiers
                    sections = file.split('_')
                    speaker_id = sections[0]
                    speaker_id = int(speaker_id[2:])
                    emotion_str = sections[2]
                    emotion_char = emotion_str[0]
                    #print(emotion_char)
                    intermediate_folder = self._file_emotions[emotion_char]
                    identifiers = [speaker_id,emotion_char, os.path.join(intermediate_folder, file)]
                    #print(identifiers)
                    # Append identifier to data
                    data.append(identifiers)

        # Create pandas dataframe
        self.df = pd.DataFrame(data, columns=['speaker_id', 'emotion','file'], dtype=np.float32)

        # Map emotion labels to numeric values
        self.df['emotion'] = self.df['emotion'].map(self._emotions).astype(np.int64)


    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        audio_name = os.path.join(self.root, self.df.loc[idx, 'file'])
        waveform, sample_rate = torchaudio.load(audio_name)
        emotion = self.df.loc[idx, 'emotion']
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0).unsqueeze(0)
        
        if sample_rate != 16000:
            waveform = torchaudio.functional.resample(waveform, orig_freq=sample_rate, new_freq=16000)
        sample_rate = 16000
        #print(waveform.shape)
        sample = {
            'waveform': waveform,
            'sample_rate': sample_rate,
            'emotion': emotion
        }

        return sample