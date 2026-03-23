import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset

from build_dataset.augmentation import SpectrogramAugmentation

def get_samples(root_dir, split_name):
    
    split_path = os.path.join(root_dir, split_name)
    samples = []

    genres = sorted([
        g for g in os.listdir(split_path)
        if os.path.isdir(os.path.join(split_path, g))
    ])
    class_map = {genre: idx for idx, genre in enumerate(genres)}

    for genre in genres:
        genre_path = os.path.join(split_path, genre)
        
        for file_name in os.listdir(genre_path):
            if file_name.endswith(".npy"):
                npy_path = os.path.join(genre_path, file_name)
                samples.append((npy_path, class_map[genre]))

    return samples, class_map


class MelNPYDataset(Dataset):
    def __init__(self, samples, train=True):
        self.samples = samples
        self.train = train
        self.augment = SpectrogramAugmentation() if train else None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        npy_path, label = self.samples[idx]

        mel = np.load(npy_path)

        mel = torch.tensor(mel, dtype=torch.float32)

        if mel.dim() == 2:
            mel = mel.unsqueeze(0)
        
        mel = (mel - mel.mean()) / (mel.std() + 1e-6)
        
        crop_size = 120
        T = mel.shape[-1]
        
        if T > crop_size:
            if self.train:
                start = random.randint(0, T - crop_size)
            else:
                start = (T - crop_size) // 2
        
            mel = mel[:, :, start:start + crop_size]
        
        if self.train:
            mel = self.augment(mel)

        return mel, label