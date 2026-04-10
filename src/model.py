import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn as nn

class GalaxyDataset(Dataset):

    def __init__(self, file_path, n_bins = 70, r_max = 10):

        data = np.load(file_path, allow_pickle=True)
        self.dataset = data["dataset"]
        self.n_bins = n_bins
        self.r_bins = np.linspace(0, r_max, n_bins + 1 )


    def compute_features(self, R, vlos):

        features = []

        for i in range(self.n_bins):
            mask = ((R >= self.r_bins[i]) & (R < self.r_bins[i+1]))
            if np.sum(mask) > 0:
                v = vlos[mask]

                features.append(np.mean(v))
                features.append(np.std(v))
                features.append(len(v))
            else:
                features += [0,0,0]

        return np.array(features)


    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):

        g = self.dataset[idx]
        R = g["R"]
        vlos = g["vlos"]

        x = self.compute_features(R, vlos)

        y = np.array([
            np.log10(g["rho_s"]),
            np.log10(g["r_s"]),
            np.log10(g["a"]),
            g["beta_inf"],
            np.log10(g["r_beta"])

        ])

        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32)
        )

class SimpleModel(nn.Module):

    def __init__(self, input_dim, output_dim = 5):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
            nn.Dropout(0.1),

        )

    def forward(self, x):
        return self.net(x)