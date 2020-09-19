import enum
import math
import os
import random
from argparse import ArgumentParser
from pathlib import Path
import neptune

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, IterableDataset
from tqdm import tqdm
from transformers import AdamW

from src.model import ColBERT, SparseColBERT
from src.parameters import DEVICE, SAVED_CHECKPOINTS
from src.utils import batch, print_message, save_checkpoint


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


class TrainReader:
    def __init__(self, data_file):
        print_message("#> Training with the triples in", data_file, "...\n\n")
        self.reader = open(data_file, mode="r", encoding="utf-8")

    def get_minibatch(self, bsize):
        return [self.reader.readline().split("\t") for _ in range(bsize)]


checkpoint_dir = Path("checkpoints")
checkpoint_dir.mkdir(parents=True, exist_ok=True)


def manage_checkpoints(colbert, optimizer, batch_idx):
    config = colbert.config
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


def train(args):
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
            use_binarization=args.use_binarization,
            dim=args.dim,
            similarity_metric=args.similarity,
        )
    colbert = colbert.to(DEVICE)
    colbert.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(colbert.parameters(), lr=args.lr, eps=1e-8)

    optimizer.zero_grad()
    labels = torch.zeros(args.bsize, dtype=torch.long, device=DEVICE)

    reader = TrainReader(args.triples)
    # dset = TrainDataset(args.triples)
    # loader = DataLoader(dset, batch_size=args.bsize, num_workers=0, pin_memory=True)
    train_loss = 0.0

    PRINT_PERIOD = 100

    for batch_idx in tqdm(range(args.maxsteps)):
        Batch = reader.get_minibatch(args.bsize)
        # for batch_idx, Batch in enumerate(tqdm(loader)):
        #     if batch_idx > args.maxsteps:
        #         print_message("#> Finish training at", batch_idx, "...\n\n")
        #         break
        #     Batch = [[q, pos, neg] for (q, pos, neg) in zip(Batch[0], Batch[1], Batch[2])]
        Batch = sorted(Batch, key=lambda x: max(len(x[1]), len(x[2])))

        positive_score, negative_score = 0.0, 0.0
        for B_idx in range(args.accumsteps):
            size = args.bsize // args.accumsteps
            B = Batch[B_idx * size : (B_idx + 1) * size]
            Q, D1, D2 = zip(*B)

            colbert_out = colbert(Q + Q, D1 + D2)
            colbert_out1, colbert_out2 = colbert_out[: len(Q)], colbert_out[len(Q) :]

            out = torch.stack((colbert_out1, colbert_out2), dim=-1)

            positive_score, negative_score = (
                round(colbert_out1.mean().item(), 2),
                round(colbert_out2.mean().item(), 2),
            )

            # if (B_idx % PRINT_PERIOD) == 0:
            #     print(
            #         "#>>>   ",
            #         positive_score,
            #         negative_score,
            #         "\t\t|\t\t",
            #         positive_score - negative_score,
            #     )

            loss = criterion(out, labels[: out.size(0)])
            loss = loss / args.accumsteps
            loss.backward()

            train_loss += loss.item()

        torch.nn.utils.clip_grad_norm_(colbert.parameters(), 2.0)

        optimizer.step()
        optimizer.zero_grad()

        if (batch_idx % PRINT_PERIOD) == 0:
            # score
            print(
                "#>>>   ",
                positive_score,
                negative_score,
                "\t\t|\t\t",
                positive_score - negative_score,
            )

            # loss
            print_message(batch_idx, train_loss / (batch_idx + 1))

            args.neptune.log_metric('positive_score', (batch_idx + 1), positive_score)
            args.neptune.log_metric('negative_score', (batch_idx + 1), negative_score)
            args.neptune.log_metric('margin', (batch_idx + 1), positive_score - negative_score)
            args.neptune.log_metric('loss', (batch_idx + 1), train_loss / (batch_idx + 1))

        manage_checkpoints(colbert, optimizer, batch_idx + 1)
