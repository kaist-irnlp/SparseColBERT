from torch import nn
import torch
import math
import torch.nn.functional as F
import abc
from .nupic import *


class MinMaxLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        _min, _max = (
            input.min(1, keepdim=True).values,
            input.max(1, keepdim=True).values,
        )
        input -= _min
        input /= _max - _min
        return input


class WTAModel(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams

        # init weights
        self._init_layers()

    def forward(self, x):
        features = self.layers(x)
        if self.hparams.model.normalize_sparse:
            features = F.normalize(features, dim=-1)
        return features

    def on_epoch_end(self):
        self.apply(update_boost_strength)
        self.apply(rezero_weights)

    @property
    def entropy(self):
        return self.kwinners[-1].entropy()

    @property
    def max_entropy(self):
        return self.kwinners[-1].max_entropy()

    @property
    def boost_strength(self):
        return self.kwinners[-1].boost_strength

    def _init_layers(self):
        self.layers = nn.Sequential()
        self.kwinners = []  # for logging

        n = self.hparams.n
        k = self.hparams.k
        weight_sparsity = self.hparams.model.weight_sparsity
        normalize_weights = self.hparams.model.normalize_weights
        # dropout = self.hparams.model.dropout
        k_inference_factor = self.hparams.model.k_inference_factor
        boost_strength = self.hparams.model.boost_strength
        boost_strength_factor = self.hparams.model.boost_strength_factor
        next_input_size = self.hparams.model.dense_size
        for i in range(len(n)):
            linear = nn.Linear(next_input_size, n[i])
            if 0 < weight_sparsity < 1:
                linear = SparseWeights(linear, sparsity=weight_sparsity)
                if normalize_weights:
                    linear.apply(normalize_sparse_weights)
            self.layers.add_module(f"linear_{i+1}", linear)
            # self.layers.add_module(f"bn_{i+1}", nn.BatchNorm1d(n[i], affine=False))
            if self.hparams.model.use_activation:
                self.layers.add_module(f"selu_{i+1}", nn.SELU())
            if self.hparams.model.use_nonneg:  # minmax norm
                self.layers.add_module(f"minmax_{i+1}", MinMaxLayer())
            # add kwinner layer
            kwinner = KWinners(
                n=n[i],
                percent_on=k[i],
                k_inference_factor=k_inference_factor,
                boost_strength=boost_strength,
                boost_strength_factor=boost_strength_factor,
                break_ties=True,
                relu=False,
                inplace=False,
            )
            self.layers.add_module(f"kwinner_{i+1}", kwinner)
            self.kwinners.append(kwinner)
            next_input_size = n[i]
        # save output_size
        self.output_size = next_input_size
