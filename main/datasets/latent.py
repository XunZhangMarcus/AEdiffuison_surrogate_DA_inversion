import torch
from torch.utils.data import Dataset
from joblib import load
import scipy.io
import os
import numpy as np

class MyDatasetInitial(Dataset):
    def __init__(
        self,
        z_vae_size,
        z_ddpm_size,
        share_ddpm_latent=True,
    ):

        n_samples, *dims = z_ddpm_size

        self.z_vae = torch.randn(z_vae_size)
        self.share_ddpm_latent = share_ddpm_latent

        if self.share_ddpm_latent:
            self.z_ddpm = torch.randn(dims)
        else:
            self.z_ddpm = torch.randn(z_ddpm_size)

    def __getitem__(self, idx):
        if self.share_ddpm_latent:
            return self.z_ddpm, self.z_vae[idx]
        return self.z_ddpm[idx], self.z_vae[idx]

    def __len__(self):
        return int(self.z_vae.size(0))


class MyDatasetInversion(Dataset):
    def __init__(self, data_dir, Ne):
        self.data_dir = data_dir
        self.z_ddpm_file = "z_ddpm_128_128.mat"
        self.z_vae_file = "z_a.mat"
        self.Ne = Ne
        z_ddpm_file = os.path.join(self.data_dir, self.z_ddpm_file)
        z_ddpm_data = scipy.io.loadmat(z_ddpm_file)["z_ddpm"]

        z_ddpm_data = np.expand_dims(z_ddpm_data, axis=0)
        self.z_ddpm = torch.from_numpy(z_ddpm_data).float()

        # Load z_vae from mat file
        z_vae_file = os.path.join(self.data_dir, self.z_vae_file)
        z_vae_data = scipy.io.loadmat(z_vae_file)["z_a"]
        z_vae_data = np.split(z_vae_data, self.Ne, axis=1)
        z_vae_data = [np.expand_dims(arr, 2) for arr in z_vae_data]
        self.z_vae = [torch.from_numpy(arr).float() for arr in z_vae_data]

    def __len__(self):
        return len(self.z_vae)

    def __getitem__(self, idx):

        return self.z_ddpm, self.z_vae[idx]


class MyDatasetResults(Dataset):
    def __init__(self, data_dir, iter):
        self.data_dir = data_dir
        self.z_ddpm_file = "z_ddpm_128_128.mat"
        self.z_vae_file = "z_mean_results.mat"
        self.iter = iter
        # Load z_ddpm from mat file
        z_ddpm_file = os.path.join(self.data_dir, self.z_ddpm_file)
        z_ddpm_data = scipy.io.loadmat(z_ddpm_file)["z_ddpm"]

        z_ddpm_data = np.expand_dims(z_ddpm_data, axis=0)
        self.z_ddpm = torch.from_numpy(z_ddpm_data).float()

        # Load z_vae from mat file
        z_vae_file = os.path.join(self.data_dir, self.z_vae_file)
        z_vae_data = scipy.io.loadmat(z_vae_file)["z_mean_results"]
        z_vae_data = np.split(z_vae_data, self.iter+1, axis=1)
        z_vae_data = [np.expand_dims(arr, 2) for arr in z_vae_data]
        self.z_vae = [torch.from_numpy(arr).float() for arr in z_vae_data]

    def __len__(self):
        return len(self.z_vae)

    def __getitem__(self, idx):

        return self.z_ddpm, self.z_vae[idx]


class VaeMyDataset(Dataset):
    def __init__(self, data_dir, Ne):
        self.data_dir = data_dir
        self.z_vae_file = "z_a.mat"
        self.Ne = Ne
        # Load z_vae from mat file
        z_vae_file = os.path.join(self.data_dir, self.z_vae_file)
        z_vae_data = scipy.io.loadmat(z_vae_file)["z_a"]
        z_vae_data = np.split(z_vae_data, self.Ne, axis=1)
        z_vae_data = [np.expand_dims(arr, 2) for arr in z_vae_data]
        self.z_vae = [torch.from_numpy(arr).float() for arr in z_vae_data]

    def __len__(self):
        return len(self.z_vae)

    def __getitem__(self, idx):
        return self.z_vae[idx]


