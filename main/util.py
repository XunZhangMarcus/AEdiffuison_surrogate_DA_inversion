import logging

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

from datasets import (
    AFHQv2Dataset,
    CelebADataset,
    CelebAHQDataset,
    CelebAMaskHQDataset,
    CIFAR10Dataset,
    FFHQDataset,
)

logger = logging.getLogger(__name__)


def configure_device(device):
    if device.startswith("gpu"):
        gpu_id = device.split(":")[-1]
        if gpu_id == "":
            # Use all GPU's
            gpu_id = -1
        gpu_id = [int(id) for id in gpu_id.split(",")]
        return f"cuda:{gpu_id}", gpu_id
    return device


def space_timesteps(num_timesteps, desired_count, type="uniform"):
    """
    Create a list of timesteps to use from an original diffusion process,
    given the number of timesteps we want to take from equally-sized portions
    of the original process.
    :param num_timesteps: the number of diffusion steps in the original
                          process to divide up.
    :return: a set of diffusion steps from the original process to use.
    """
    if type == "uniform":
        for i in range(1, num_timesteps):
            if len(range(0, num_timesteps, i)) == desired_count:
                return range(0, num_timesteps, i)
        raise ValueError(
            f"cannot create exactly {desired_count} steps with an integer stride"
        )
    elif type == "quad":
        seq = np.linspace(0, np.sqrt(num_timesteps * 0.8), desired_count) ** 2
        seq = [int(s) for s in list(seq)]
        return seq
    else:
        raise NotImplementedError


def get_dataset(name, root, image_size, norm=True, flip=True, **kwargs):
    assert isinstance(norm, bool)

    if name == "celeba":
        dataset = CelebADataset(root, norm=norm, **kwargs)
    elif name == "TI":
        dataset = TI_ataset(root, norm=norm, **kwargs)
    else:
        raise NotImplementedError(
            f"The dataset {name} does not exist in our datastore."
        )
    return dataset


def convert_to_np(obj):
    obj = obj.permute(0, 2, 3, 1).contiguous()
    obj = obj.detach().cpu().numpy()

    obj_list = []
    for _, out in enumerate(obj):
        obj_list.append(out)
    return obj_list


def normalize(obj):
    B, C, H, W = obj.shape
    for i in range(1):
        channel_val = obj[:, i, :, :].view(B, -1)
        channel_val -= channel_val.min(1, keepdim=True)[0]
        channel_val /= (channel_val.max(1, keepdim=True)[0] - channel_val.min(1, keepdim=True)[0])
        channel_val = channel_val.view(B, H, W)
        obj[:, i, :, :] = channel_val
    return obj


def save_as_np(obj, file_name="output", denorm=True):
    if denorm:
        obj = normalize(obj)
    obj_list = convert_to_np(obj)

    for i, out in enumerate(obj_list):
        current_file_name = file_name + "_%d.npy" % i
        np.save(current_file_name, out)
