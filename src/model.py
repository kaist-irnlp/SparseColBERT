from torch._C import dtype
from .wta import WTAModel
import string
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from omegaconf import ListConfig

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
        k_inference_factor=1.0,
        normalize_sparse=True,
        use_nonneg=False,
        similarity_metric="cosine",
    ):
        dim_not_used = 128
        super().__init__(
            config, query_maxlen, doc_maxlen, dim_not_used, similarity_metric
        )
        self.n = n
        self.k = k
        self.dense_size = self.bert.embeddings.word_embeddings.weight.shape[1]
        wta_params = OmegaConf.create(
            {
                "model": {
                    "n": n,
                    "k": k,
                    "weight_sparsity": 0.3,
                    "normalize_weights": True,
                    "k_inference_factor": k_inference_factor,
                    "boost_strength": 1.5,
                    "boost_strength_factor": 0.9,
                    "dense_size": self.dense_size,
                    "normalize_sparse": normalize_sparse,
                    "use_nonneg": use_nonneg,
                },
            }
        )
        self.linear = nn.Identity()
        self.sparse = WTAModel(wta_params)
        self.is_sparse = True
        self.criterion = nn.CrossEntropyLoss()

    def forward(
        self,
        input_Q_ids,
        input_Q_att,
        input_D1_ids,
        input_D1_att,
        input_D1_mask,
        input_D2_ids,
        input_D2_att,
        input_D2_mask,
    ):
        # Q, D1, D2 = zip(*B)
        Q = self.query(
            torch.cat([input_Q_ids, input_Q_ids], dim=0),
            torch.cat([input_Q_att, input_Q_att], dim=0),
        )
        D = self.doc(
            torch.cat([input_D1_ids, input_D2_ids], dim=0),
            torch.cat([input_D1_att, input_D2_att], dim=0),
            torch.cat([input_D1_mask, input_D2_mask], dim=0),
        )
        colbert_out = self.score(Q, D)
        colbert_out1, colbert_out2 = (
            colbert_out[: len(input_Q_ids)],
            colbert_out[len(input_Q_ids) :],
        )

        out = torch.stack((colbert_out1, colbert_out2), dim=-1)
        positive_score, negative_score = (
            round(colbert_out1.mean().item(), 2),
            round(colbert_out2.mean().item(), 2),
        )
        labels = torch.zeros_like(out, dtype=torch.long)
        loss_contrast = self.criterion(out, labels[:, 0])

        # ortho
        loss_ortho = self.ortho_all([Q, D]) if self.use_ortho else 0
        loss = loss_contrast + loss_ortho
        return loss, out

    def query(self, input_ids, attention_mask):
        # Q = super().query(queries)
        Q = self.bert(input_ids, attention_mask=attention_mask)[0]
        Q = self.linear(Q)
        Q = torch.nn.functional.normalize(Q, p=2, dim=-1)
        return self._sparse_maxpool(Q)

    def doc(self, input_ids, attention_mask, mask, return_mask=False):
        # D = super().doc(docs, return_mask)

        D = self.bert(input_ids, attention_mask=attention_mask)[0]
        D = self.linear(D)

        D = D * mask.unsqueeze(2)
        D = torch.nn.functional.normalize(D, p=2, dim=-1)

        D = self._sparse_maxpool(D)
        return (D, mask) if return_mask else D

    def tokenize_and_query(self, queries):
        Q = super().query(queries)
        return self._sparse_maxpool(Q)

    def tokenize_and_doc(self, docs, return_mask=False):
        D, mask = None, None
        if return_mask:
            D, mask = super().doc(docs, return_mask)
        else:
            D = super().doc(docs, return_mask)
        D = self._sparse_maxpool(D)
        return (D, mask) if return_mask else D

    def _sparse_maxpool(self, T, k_mat=None):
        """
        k_mat.shape = (batch_size, num_tokens)
        """
        T_sparse = []
        if k_mat is None:  # static k
            for t in torch.unbind(T):
                t_sparse = torch.max(self.sparse(t), dim=0).values
                T_sparse.append(t_sparse)
        else:  # dynamic k
            for t, k_vec in zip(torch.unbind(T), k_mat):
                t_sparse = torch.max(self.sparse(t, k_vec), dim=0).values
                T_sparse.append(t_sparse)

        T_sparse = torch.stack(T_sparse)
        return T_sparse

    def score(self, Q, D):
        if self.similarity_metric == "cosine":
            scores = (Q * D).sum(dim=-1)
            return scores

        assert self.similarity_metric == "l2"
        return F.mse_loss(Q, D, reduction="none")
