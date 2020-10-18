import enum
import math
import os
import random
from argparse import ArgumentParser
from pathlib import Path
import string

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, IterableDataset
from transformers import BertPreTrainedModel, BertModel, BertTokenizer
from tqdm import tqdm
from transformers import AdamW, Trainer

from src.model import ColBERT, SparseColBERT
from src.parameters import DEVICE, SAVED_CHECKPOINTS
from src.utils import batch, print_message, save_checkpoint, load_checkpoint
from dataclasses import dataclass
from typing import List, Optional, Union


@dataclass
class InputExample:
    input_Q_ids: List[int]
    input_Q_att: List[int]
    input_D1_ids: List[int]
    input_D1_att: List[int]
    input_D1_mask: List[int]
    input_D2_ids: List[int]
    input_D2_att: List[int]
    input_D2_mask: List[int]


class TrainDataset(IterableDataset):
    def __init__(self, data_file):
        print_message("#> Training with the triples in", data_file, "...\n\n")
        if data_file.endswith(".parquet"):
            self.data = pd.read_parquet(data_file).values.tolist()
        else:
            self.data = pd.read_csv(
                data_file, sep="\t", names=["query", "pos", "neg"]
            ).values.tolist()

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        data = None
        if worker_info is None:  # single-process data loading, return the full iterator
            data = self.data
        else:  # in a worker process
            # split workload
            per_worker = int(math.ceil(len(self.data) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            start = worker_id * per_worker
            end = min(start + per_worker, len(self.data))
            data = self.data[start:end]
        return iter(data)


class TrainDatasetforTPU(Dataset):
    def __init__(self, data_file, query_maxlen, doc_maxlen, numins, startidx):
        print_message("#> Training with the triples in", data_file, "...\n\n")
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.reader = open(data_file, mode="r", encoding="utf-8")
        self.query_maxlen = query_maxlen
        self.doc_maxlen = doc_maxlen
        self.skiplist = {w: True for w in string.punctuation}
        self.numins = numins
        #self.data = self._getdata(numins)
        if not startidx == None:
            print("Start from: ", startidx)
            self._initalize_reader_from_startindex(startidx)

    def __len__(self):
        return self.numins
        #return len(self.data)

    def _getdata(self, numins):
        return [self.reader.readline().split("\t") for _ in range(numins)]
    
    def _initalize_reader_from_startindex(self, start_index):
        for _ in range(start_index):
            self.reader.readline()

    def _convert_raw_to_obj(self, raw_ex):
        Q, D1, D2 = raw_ex[0], raw_ex[1], raw_ex[2]
        Q_ids, Q_att = self._convert_query_to_ids(Q)
        D1_ids, D1_att, D1_mask = self._convert_doc_to_ids(D1)
        D2_ids, D2_att, D2_mask = self._convert_doc_to_ids(D2)
        return InputExample(
            input_Q_ids=Q_ids,
            input_Q_att=Q_att,
            input_D1_ids=D1_ids,
            input_D1_att=D1_att,
            input_D1_mask=D1_mask,
            input_D2_ids=D2_ids,
            input_D2_att=D2_att,
            input_D2_mask=D2_mask,
        )
        # return InputExample(Q=Q, D1=D1, D2=D2)
        # return InputExample(*raw_ex)

    def _convert_query_to_ids(self, query):
        query = ["[unused0]"] + self._tokenize(query)
        input_id, attention_mask = self._encode(query, self.query_maxlen)
        input_id, attention_mask = (
            torch.tensor(input_id, dtype=torch.long),
            torch.tensor(attention_mask, dtype=torch.long),
        )
        return input_id, attention_mask

    def _convert_doc_to_ids(self, doc):
        doc = ["[unused1]"] + self._tokenize(doc)[: self.doc_maxlen - 3]
        length = len(doc) + 2
        mask = (
            [1]
            + [x not in self.skiplist for x in doc]
            + [1]
            + [0] * (self.doc_maxlen - length)
        )
        mask = torch.tensor(mask, dtype=torch.float32)
        input_id, attention_mask = self._encode(doc, self.doc_maxlen)
        input_id, attention_mask = (
            torch.tensor(input_id, dtype=torch.long),
            torch.tensor(attention_mask, dtype=torch.long),
        )
        return input_id, attention_mask, mask

    def _encode(self, x, max_length):
        input_ids = self.tokenizer.encode(
            x, add_special_tokens=True, max_length=max_length, truncation=True
        )

        padding_length = max_length - len(input_ids)
        attention_mask = [1] * len(input_ids) + [0] * padding_length
        input_ids = input_ids + [103] * padding_length

        return input_ids, attention_mask

    def _tokenize(self, text):
        if type(text) == list:
            return text

        return self.tokenizer.tokenize(text)

    def __getitem__(self, i):
        return self._convert_raw_to_obj(self.reader.readline().split("\t"))
        #return self._convert_raw_to_obj(self.data[i])


class TrainReader:
    def __init__(self, data_file):
        print_message("#> Training with the triples in", data_file, "...\n\n")
        self.reader = open(data_file, mode="r", encoding="utf-8")

    def get_minibatch(self, bsize):
        return [self.reader.readline().split("\t") for _ in range(bsize)]


# checkpoint_dir = Path("checkpoints")
# checkpoint_dir.mkdir(parents=True, exist_ok=True)


def manage_checkpoints(colbert, optimizer, batch_idx, output_dir):
    config = colbert.config
    checkpoint_dir = Path(output_dir)
    model_desc = f"colbert_hidden={config.hidden_size}_qlen={colbert.query_maxlen}_dlen={colbert.doc_maxlen}"
    if hasattr(colbert, "sparse"):
        n = "-".join([str(n) for n in colbert.n])
        k = "-".join([str(k) for k in colbert.k])
        model_desc += f"_sparse_n={n}_k={k}"
    else:
        model_desc += f"_dense"

    if batch_idx % 50000 == 0:
        save_checkpoint(
            checkpoint_dir / f"{model_desc}.last.dnn", 0, batch_idx, colbert, optimizer
        )

    if batch_idx in SAVED_CHECKPOINTS:
        save_checkpoint(
            checkpoint_dir / (f"{model_desc}.{batch_idx}.dnn"),
            0,
            batch_idx,
            colbert,
            optimizer,
        )


def train(args, training_args):
    if args.use_dense:
        colbert = ColBERT.from_pretrained(
            "bert-base-uncased",
            query_maxlen=args.query_maxlen,
            doc_maxlen=args.doc_maxlen,
            dim=args.dim,
            similarity_metric=args.similarity,
        )
    else:
        colbert = SparseColBERT.from_pretrained(
            "bert-base-uncased",
            query_maxlen=args.query_maxlen,
            doc_maxlen=args.doc_maxlen,
            n=args.n,
            k=args.k,
            normalize_sparse=args.normalize_sparse,
            use_nonneg=args.use_nonneg,
            similarity_metric=args.similarity,
        )
    if not args.original_checkpoint == None:
        non_strict_load = False
        checkpoint = load_checkpoint(args.original_checkpoint, colbert, non_strict_load = non_strict_load)
        
    train_dataset = TrainDatasetforTPU(
        args.triples, 
        args.query_maxlen, 
        args.doc_maxlen, 
        numins=args.training_ins_num,
        startidx = args.training_ins_start_from,
    )
    trainer = Trainer(
        model=colbert,
        args=training_args,
        train_dataset=train_dataset,
    )

    # Training
    # if training_args.do_train:
    trainer.train()
    trainer.save_model()

    # colbert = colbert.to(DEVICE)
    # colbert.train()

    # criterion = nn.CrossEntropyLoss()
    # optimizer = AdamW(colbert.parameters(), lr=args.lr, eps=1e-8)

    # optimizer.zero_grad()
    # labels = torch.zeros(args.bsize, dtype=torch.long, device=DEVICE)

    # reader = TrainReader(args.triples)
    # # dset = TrainDataset(args.triples)
    # # loader = DataLoader(dset, batch_size=args.bsize, num_workers=0, pin_memory=True)
    # train_loss = 0.0

    # PRINT_PERIOD = 100

    # for batch_idx in tqdm(range(args.maxsteps)):
    #     Batch = reader.get_minibatch(args.bsize)
    #     # for batch_idx, Batch in enumerate(tqdm(loader)):
    #     #     if batch_idx > args.maxsteps:
    #     #         print_message("#> Finish training at", batch_idx, "...\n\n")
    #     #         break
    #     #     Batch = [[q, pos, neg] for (q, pos, neg) in zip(Batch[0], Batch[1], Batch[2])]
    #     Batch = sorted(Batch, key=lambda x: max(len(x[1]), len(x[2])))

    #     positive_score, negative_score = 0.0, 0.0
    #     for B_idx in range(args.accumsteps):
    #         size = args.bsize // args.accumsteps
    #         B = Batch[B_idx * size : (B_idx + 1) * size]
    #         Q, D1, D2 = zip(*B)

    #         colbert_out = colbert(Q + Q, D1 + D2)
    #         colbert_out1, colbert_out2 = colbert_out[: len(Q)], colbert_out[len(Q) :]

    #         out = torch.stack((colbert_out1, colbert_out2), dim=-1)

    #         positive_score, negative_score = (
    #             round(colbert_out1.mean().item(), 2),
    #             round(colbert_out2.mean().item(), 2),
    #         )

    #         # if (B_idx % PRINT_PERIOD) == 0:
    #         #     print(
    #         #         "#>>>   ",
    #         #         positive_score,
    #         #         negative_score,
    #         #         "\t\t|\t\t",
    #         #         positive_score - negative_score,
    #         #     )

    #         loss = criterion(out, labels[: out.size(0)])
    #         loss = loss / args.accumsteps
    #         loss.backward()

    #         train_loss += loss.item()

    #     torch.nn.utils.clip_grad_norm_(colbert.parameters(), 2.0)

    #     optimizer.step()
    #     optimizer.zero_grad()

    #     if (batch_idx % PRINT_PERIOD) == 0:
    #         # score
    #         print(
    #             "#>>>   ",
    #             positive_score,
    #             negative_score,
    #             "\t\t|\t\t",
    #             positive_score - negative_score,
    #         )

    #         # loss
    #         print_message(batch_idx, train_loss / (batch_idx + 1))

    #     manage_checkpoints(colbert, optimizer, batch_idx + 1, args.output_dir)
