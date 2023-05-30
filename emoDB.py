import os
import torch
import torchaudio
import pandas as pd
import numpy as np
from torch.nn.utils.rnn import pad_sequence

class EmodbDataset(object):
    """
        Create a Dataset for Emodb. Each item is a tuple of the form:
        (waveform, sample_rate, emotion)
    """

    _ext_audio = '.wav'
    _emotions = { 'W': 1, 'L': 2, 'E': 3, 'A': 4, 'F': 5, 'T': 6, 'N': 7 } # W = anger, L = boredom, E = disgust, A = anxiety/fear, F = happiness, T = sadness, N = neutral

    def __init__(self, root='download'):
        """
        Args:
            root (string): Directory containing the wav folder
        """
        self.root = root

        # Iterate through all audio files
        data = []
        for _, _, files in os.walk(root):
            for file in files:
                if file.endswith(self._ext_audio):
                    # Construct file identifiers
                    identifiers = [file[0:2], file[2:5], file[5], file[6], os.path.join('wav', file)]

                    # Append identifier to data
                    data.append(identifiers)

        # Create pandas dataframe
        self.df = pd.DataFrame(data, columns=['speaker_id', 'code', 'emotion', 'version', 'file'], dtype=np.float32)

        # Map emotion labels to numeric values
        self.df['emotion'] = self.df['emotion'].map(self._emotions).astype(np.float32)

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
        print(waveform.shape)
        sample = {
            'waveform': waveform,
            'sample_rate': sample_rate,
            'emotion': emotion
        }

        return sample
    def collate_fn(batch):
        # Sort the batch in the descending order
        batch.sort(key=lambda x: x['waveform'].size(1), reverse=True)

        # Zip the batch
        waveforms, sample_rates, emotions = zip(*[(d['waveform'], d['sample_rate'], d['emotion']) for d in batch])

        # Pad sequences
        waveforms_padded = pad_sequence(waveforms, batch_first=True)

        # Create attention masks
        attention_masks = [torch.ones_like(waveform) for waveform in waveforms]
        attention_masks_padded = pad_sequence(attention_masks, batch_first=True)

        # Stack sample rates and emotions
        sample_rates = torch.stack(sample_rates)
        emotions = torch.stack(emotions)

        return waveforms_padded, sample_rates, emotions, attention_masks_padded

# Example: Load Emodb dataset
# emodb_dataset = EmodbDataset('/home/alanwuha/Documents/Projects/datasets/emodb/download')

# Example: Iterate through samples
# for i in range(len(emodb_dataset)):
#     sample = emodb_dataset[i]
#     print(i, sample)