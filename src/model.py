from .wta import WTAModel
import string
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from omegaconf import ListConfig

from transformers import (
    BertPreTrainedModel,
    BertModel,
    BertTokenizer,
    BertTokenizerFast,
)
from src.parameters import DEVICE


class ColBERT(BertPreTrainedModel):
    def __init__(
        self, config, query_maxlen, doc_maxlen, dim=128, similarity_metric="cosine"
    ):
        super(ColBERT, self).__init__(config)

        self.query_maxlen = query_maxlen
        self.doc_maxlen = doc_maxlen
        self.similarity_metric = similarity_metric

        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.skiplist = {w: True for w in string.punctuation}

        self.bert = BertModel(config)
        self.linear = nn.Linear(config.hidden_size, dim, bias=False)
        self.config = config

        self.init_weights()

    def forward(self, Q, D):
        return self.score(self.query(Q), self.doc(D))

    def query(self, queries):
        queries = [["[unused0]"] + self._tokenize(q) for q in queries]

        input_ids, attention_mask = zip(
            *[self._encode(x, self.query_maxlen) for x in queries]
        )
        input_ids, attention_mask = self._tensorize(input_ids), self._tensorize(
            attention_mask
        )

        Q = self.bert(input_ids, attention_mask=attention_mask)[0]
        Q = self.linear(Q)

        return torch.nn.functional.normalize(Q, p=2, dim=-1)

    def doc(self, docs, return_mask=False):
        docs = [["[unused1]"] + self._tokenize(d)[: self.doc_maxlen - 3] for d in docs]

        lengths = [len(d) + 2 for d in docs]  # +2 for [CLS], [SEP]
        d_max_length = max(lengths)

        input_ids, attention_mask = zip(*[self._encode(x, d_max_length) for x in docs])
        input_ids, attention_mask = self._tensorize(input_ids), self._tensorize(
            attention_mask
        )

        D = self.bert(input_ids, attention_mask=attention_mask)[0]
        D = self.linear(D)

        # [CLS] .. d ... [SEP] [PAD] ... [PAD]
        mask = [
            [1]
            + [x not in self.skiplist for x in d]
            + [1]
            + [0] * (d_max_length - length)
            for d, length in zip(docs, lengths)
        ]

        D = D * torch.tensor(mask, device=DEVICE, dtype=torch.float32).unsqueeze(2)
        D = torch.nn.functional.normalize(D, p=2, dim=-1)

        return (D, mask) if return_mask else D

    def score(self, Q, D):
        if self.similarity_metric == "cosine":
            return (Q @ D.permute(0, 2, 1)).max(2).values.sum(1)

        assert self.similarity_metric == "l2"
        return (
            (-1.0 * ((Q.unsqueeze(2) - D.unsqueeze(1)) ** 2).sum(-1))
            .max(-1)
            .values.sum(-1)
        )

    def _tokenize(self, text):
        if type(text) == list:
            return text

        return self.tokenizer.tokenize(text)

    def _encode(self, x, max_length):
        input_ids = self.tokenizer.encode(
            x, add_special_tokens=True, max_length=max_length, truncation=True
        )

        padding_length = max_length - len(input_ids)
        attention_mask = [1] * len(input_ids) + [0] * padding_length
        input_ids = input_ids + [103] * padding_length

        return input_ids, attention_mask

    def _tensorize(self, l):
        return torch.tensor(l, dtype=torch.long, device=DEVICE)


class SparseColBERT(ColBERT):
    def __init__(
        self,
        config,
        query_maxlen,
        doc_maxlen,
        n,
        k,
        k_inference_factor=1.5,
        normalize_sparse=True,
        use_nonneg=False,
        use_ortho=False,
        similarity_metric="cosine",
    ):
        projection_dim_not_used = 128
        super().__init__(
            config, query_maxlen, doc_maxlen, projection_dim_not_used, similarity_metric
        )
        # modification
        n = n if type(n) in (ListConfig, list) else [n]
        k = k if type(k) in (ListConfig, list) else [k]
        self.n = n
        self.k = k
        self.dense_size = self.bert.embeddings.word_embeddings.weight.shape[1]
        wta_params = OmegaConf.create(
            {
                "n": n,
                "k": k,
                "model": {
                    "weight_sparsity": 0.3,
                    "normalize_weights": True,
                    "k_inference_factor": k_inference_factor,
                    "boost_strength": 1.5,
                    "boost_strength_factor": 0.9,
                    "dense_size": self.dense_size,
                    "normalize_sparse": normalize_sparse,
                    "use_nonneg": use_nonneg,
                    "use_ortho": use_ortho,
                },
            }
        )
        self.linear = nn.Identity()
        self.sparse = WTAModel(wta_params)
        self.is_sparse = True
        self.use_nonneg = use_nonneg
        self.use_ortho = use_ortho

    def forward(self, Q, D, return_embedding=False):
        Q, D = self.query(Q), self.doc(D)
        scores = self.score(Q, D)
        return (scores, Q, D) if return_embedding else scores

    def query(self, queries):
        Q = super().query(queries)
        return self._sparse_maxpool(Q)

    def doc(self, docs, return_mask=False):
        D, mask = None, None
        if return_mask:
            D, mask = super().doc(docs, return_mask)
        else:
            D = super().doc(docs, return_mask)
        D = self._sparse_maxpool(D)
        return (D, mask) if return_mask else D

    def _sparse_maxpool(self, T):
        T_sparse = []
        for t in torch.unbind(T):
            t_sparse = torch.max(self.sparse(t), dim=0).values
            T_sparse.append(t_sparse)
        T_sparse = torch.stack(T_sparse)
        return T_sparse

    def score(self, Q, D):
        if self.similarity_metric == "cosine":
            scores = (Q * D).sum(dim=-1)
            return scores

        assert self.similarity_metric == "l2"
        return F.mse_loss(Q, D, reduction="none")

    def ortho(self, T):
        ortho_loss = torch.mean(
            torch.norm(
                torch.matmul(T, T.T) - torch.eye(T.shape[0], device=DEVICE),
                p=2,
                dim=-1,
            )
        )
        return ortho_loss * 0.01  # lambda for ortho loss

    def ortho_all(self, tensors):
        return torch.mean(torch.tensor([self.ortho(t) for t in tensors])).to(DEVICE)
