from .wta import WTAModel
import string
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf

from transformers import BertPreTrainedModel, BertModel, BertTokenizer
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
        dim=128,
        similarity_metric="cosine",
    ):
        super().__init__(config, query_maxlen, doc_maxlen, dim, similarity_metric)
        # modification
        n = n if isinstance(n, list) else [n]
        k = k if isinstance(k, list) else [k]
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
                    "k_inference_factor": 1.5,
                    "boost_strength": 1.5,
                    "boost_strength_factor": 0.9,
                    "dense_size": self.dense_size,
                },
            }
        )
        self.linear = WTAModel(wta_params)

    def _sparse_maxpool(self, T):
        T_max = torch.max(T, dim=1).values
        return T_max

    # TODO: need changes
    def forward(self, Q, D):
        Q_sp, D_sp = self.query(Q), self.doc(D)
        Q_sp, D_sp = self._sparse_maxpool(Q_sp), self._sparse_maxpool(D_sp)
        return self.score(Q_sp, D_sp)

    # TODO: need changes
    def score(self, Q, D):
        if self.similarity_metric == "cosine":
            scores = (Q * D).sum(dim=-1)
            return scores

        assert self.similarity_metric == "l2"
        return F.mse_loss(Q, D, reduction="none")

    # def query(self, queries):
    #     queries = [["[unused0]"] + self._tokenize(q) for q in queries]

    #     input_ids, attention_mask = zip(
    #         *[self._encode(x, self.query_maxlen) for x in queries]
    #     )
    #     input_ids, attention_mask = self._tensorize(input_ids), self._tensorize(
    #         attention_mask
    #     )

    #     Q = self.bert(input_ids, attention_mask=attention_mask)[0]
    #     Q = self.linear(Q)

    #     return torch.nn.functional.normalize(Q, p=2, dim=2)

    # def doc(self, docs, return_mask=False):
    #     docs = [["[unused1]"] + self._tokenize(d)[: self.doc_maxlen - 3] for d in docs]

    #     lengths = [len(d) + 2 for d in docs]  # +2 for [CLS], [SEP]
    #     d_max_length = max(lengths)

    #     input_ids, attention_mask = zip(*[self._encode(x, d_max_length) for x in docs])
    #     input_ids, attention_mask = self._tensorize(input_ids), self._tensorize(
    #         attention_mask
    #     )

    #     D = self.bert(input_ids, attention_mask=attention_mask)[0]
    #     D = self.linear(D)

    #     # [CLS] .. d ... [SEP] [PAD] ... [PAD]
    #     mask = [
    #         [1]
    #         + [x not in self.skiplist for x in d]
    #         + [1]
    #         + [0] * (d_max_length - length)
    #         for d, length in zip(docs, lengths)
    #     ]

    #     D = D * torch.tensor(mask, device=DEVICE, dtype=torch.float32).unsqueeze(2)
    #     D = torch.nn.functional.normalize(D, p=2, dim=2)

    #     return (D, mask) if return_mask else D

    # def _tokenize(self, text):
    #     if type(text) == list:
    #         return text

    #     return self.tokenizer.tokenize(text)

    # def _encode(self, x, max_length):
    #     input_ids = self.tokenizer.encode(
    #         x, add_special_tokens=True, max_length=max_length
    #     )

    #     padding_length = max_length - len(input_ids)
    #     attention_mask = [1] * len(input_ids) + [0] * padding_length
    #     input_ids = input_ids + [103] * padding_length

    #     return input_ids, attention_mask

    # def _tensorize(self, l):
    #     return torch.tensor(l, dtype=torch.long, device=DEVICE)
