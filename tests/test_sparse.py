import pytest
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from src.wta.wta import MinMaxLayer
from src.wta import WTAModel


@pytest.fixture
def hparams():
    return OmegaConf.create(
        {
            "n": [2048],
            "k": [0.005],
            "model": {
                "weight_sparsity": 0.3,
                "normalize_weights": True,
                "k_inference_factor": 1.5,
                "boost_strength": 1.5,
                "boost_strength_factor": 0.9,
                "dense_size": 768,
                "normalize_sparse": True,
                "use_nonneg": False,
            },
        }
    )


def test_wta_nonneg(hparams):
    # normalized, does not allow negativces
    hparams.normalize_sparse = True
    hparams.use_nonneg = True
    # init
    wta = WTAModel(hparams)
    t = torch.randn(32, hparams.model.dense_size)
    t = wta(t)
    # check the output
    ## normalized
    t_norms = torch.norm(t, dim=-1)
    assert torch.allclose(t_norms, torch.ones_like(t_norms))
    ## nonneg
    assert (t < 0).sum() == 0


def test_wta_nomalized(hparams):
    # normalized, allow negatives
    hparams.normalize_sparse = True
    hparams.use_nonneg = False
    # init
    wta = WTAModel(hparams)
    t = torch.randn(32, hparams.model.dense_size)
    t = wta(t)
    # check the output
    t_norms = torch.norm(t, dim=-1)
    assert torch.allclose(t_norms, torch.ones_like(t_norms))