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

    def forward(self, x, k_vec=None):
        x = self.pre_sparse(x)

        if k_vec is None:
            features = self.sparse(x)
        else:  # variable k
            features = []
            for xi, ki in zip(x, k_vec):
                fi = self.sparse(xi.unsqueeze(dim=0), ki)
                features.append(fi)
            features = torch.cat(features)

        if self.hparams.model.normalize_sparse:
            features = F.normalize(features, dim=-1)
        return features

    def on_epoch_end(self):
        self.apply(update_boost_strength)
        self.apply(rezero_weights)

    def _init_layers(self):
        # define params
        n = self.hparams.model.n
        k = self.hparams.model.k
        weight_sparsity = self.hparams.model.weight_sparsity
        normalize_weights = self.hparams.model.normalize_weights
        k_inference_factor = self.hparams.model.k_inference_factor
        boost_strength = self.hparams.model.boost_strength
        boost_strength_factor = self.hparams.model.boost_strength_factor
        input_size = self.hparams.model.dense_size

        # build pre-sparse
        self.pre_sparse = nn.Sequential()
        linear = nn.Linear(input_size, n)
        if 0 < weight_sparsity < 1:
            linear = SparseWeights(linear, sparsity=weight_sparsity)
            if normalize_weights:
                linear.apply(normalize_sparse_weights)
        self.pre_sparse.add_module(f"linear", linear)
        if self.hparams.model.use_nonneg:  # minmax norm
            self.pre_sparse.add_module(f"minmax", MinMaxLayer())

        # build sparse
        self.sparse = KWinners(
            n=n,
            percent_on=k,
            k_inference_factor=k_inference_factor,
            boost_strength=boost_strength,
            boost_strength_factor=boost_strength_factor,
            break_ties=True,
            relu=False,
            inplace=False,
        )

        # build layers = pre_sparse + sparse
        self.layers = nn.Sequential(self.pre_sparse, self.sparse)
