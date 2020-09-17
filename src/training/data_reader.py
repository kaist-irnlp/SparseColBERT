import os
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, IterableDataset
from tqdm import tqdm
import pandas as pd

from argparse import ArgumentParser
from transformers import AdamW

from src.parameters import DEVICE, SAVED_CHECKPOINTS

from src.model import ColBERT, SparseColBERT
from src.utils import batch, print_message, save_checkpoint


class TrainDataset(Dataset):
    def __init__(self, data_file):
        print_message("#> Training with the triples in", data_file, "...\n\n")
        self.data = pd.read_csv(data_file, sep="\t", names=["query", "pos", "neg"])

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        row = self.data.iloc[index]
        return row.values


class TrainReader:
    def __init__(self, data_file):
        print_message("#> Training with the triples in", data_file, "...\n\n")
        self.reader = open(data_file, mode="r", encoding="utf-8")

    def get_minibatch(self, bsize):
        return [self.reader.readline().split("\t") for _ in range(bsize)]


def manage_checkpoints(colbert, optimizer, batch_idx):
    if batch_idx % 2000 == 0:
        save_checkpoint("colbert.dnn", 0, batch_idx, colbert, optimizer)

    if batch_idx in SAVED_CHECKPOINTS:
        save_checkpoint(
            "colbert-" + str(batch_idx) + ".dnn", 0, batch_idx, colbert, optimizer
        )


def train(args):
    colbert = SparseColBERT.from_pretrained(
        "bert-base-uncased",
        query_maxlen=args.query_maxlen,
        doc_maxlen=args.doc_maxlen,
        n=args.n,
        k=args.k,
        dim=args.dim,
        similarity_metric=args.similarity,
    )
    colbert = colbert.to(DEVICE)
    colbert.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(colbert.parameters(), lr=args.lr, eps=1e-8)

    optimizer.zero_grad()
    labels = torch.zeros(args.bsize, dtype=torch.long, device=DEVICE)

    # reader = TrainReader(args.triples)
    dset = TrainDataset(args.triples)
    loader = DataLoader(dset, batch_size=args.bsize, num_workers=0, pin_memory=True)
    train_loss = 0.0

    PRINT_PERIOD = 100

    # for batch_idx in tqdm(range(args.maxsteps)):
    #     Batch = reader.get_minibatch(args.bsize)
    for batch_idx, Batch in tqdm(enumerate(loader)):
        if batch_idx > args.maxsteps:
            print_message("#> Finish training at", batch_idx, "...\n\n")
            break
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

        manage_checkpoints(colbert, optimizer, batch_idx + 1)
