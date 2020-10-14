import torch
import os

DEVICE = torch.device("cuda")
#DEVICE = None

DEFAULT_DATA_DIR = "./data_download/"

SAVED_CHECKPOINTS = [
    32 * 1000,
    100 * 1000,
    150 * 1000,
    200 * 1000,
    300 * 1000,
    400 * 1000,
]
