from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd
import dask.dataframe as dd
from dask.diagnostics import ProgressBar
from scipy import sparse
from tqdm import tqdm
import math

from src.model import SparseColBERT
from src.parameters import DEVICE
from src.utils import load_checkpoint


def encode():
    pass


def load_model(args):
    args.model = SparseColBERT.from_pretrained(
        "bert-base-uncased",
        query_maxlen=args.query_maxlen,
        doc_maxlen=args.doc_maxlen,
        k=args.k,
        n=args.n,
        k_inference_factor=args.k_inference_factor,
    )
    args.model = args.model.to(DEVICE)
    checkpoint = load_checkpoint(args.checkpoint, args.model)
    args.model.eval()

    return args.model, checkpoint


def get_ids_and_embs(data_path, model, batch_size=32, is_query=False):
    # load
    data = pd.read_csv(data_path, sep="\t", names=["id", "text"], chunksize=batch_size)
    with ProgressBar():
        total = (
            dd.read_csv(data_path, sep="\t", names=["id", "text"]).shape[0].compute()
        )
        total_steps = math.ceil(total / batch_size)
        print("Total steps:", total_steps)

    # process
    ids = []
    embs = []
    encode = model.query if is_query else model.doc
    for chunk in tqdm(data, total=total_steps):
        # ids
        ids.append(chunk.id.values)
        # embs
        T = chunk.text.values
        e = sparse.csr_matrix(encode(T).detach().cpu())
        embs.append(e)
    ids = np.concatenate(ids)
    embs = sparse.vstack(embs)
    return ids, embs


def save_ids_and_embs(ids, embs, output_dir, postfix, is_query=False):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = "query" if is_query else "doc"
    # ids
    output_path = output_dir / f"{prefix}_ids"
    np.save(output_path, ids)
    # embs
    output_path = output_dir / f"{prefix}_{postfix}"
    sparse.save_npz(output_path, embs)


def main():
    parser = ArgumentParser(
        description="Exhaustive (non-index-based) evaluation of re-ranking with ColBERT."
    )

    # parser.add_argument("--index", dest="index", required=True)
    parser.add_argument("--checkpoint", dest="checkpoint", required=True)
    parser.add_argument("--query", default="queries.dev.small.tsv")
    parser.add_argument("--collection", dest="collection", default="collection.tsv")
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--output_dir", dest="output_dir", default="outputs.index/")
    parser.add_argument("--query_maxlen", dest="query_maxlen", default=32, type=int)
    parser.add_argument("--doc_maxlen", dest="doc_maxlen", default=180, type=int)
    parser.add_argument("--n", default=4096, type=int)
    parser.add_argument("--k", default=0.005, type=float)
    parser.add_argument("--k_inference_factor", default=1.5, type=float)

    args = parser.parse_args()

    model, meta = load_model(args)

    index_postfix = f"n={args.n}_k={args.k}_epoch={meta['epoch']}_step={meta['batch']}"
    output_dir = Path(args.output_dir)

    # query
    ## process
    ids, embs = get_ids_and_embs(
        args.query, model, batch_size=args.batch_size, is_query=True
    )
    ## save
    save_ids_and_embs(ids, embs, output_dir, index_postfix, is_query=True)

    # docs
    ## process
    ids, embs = get_ids_and_embs(
        args.collection,
        model,
        batch_size=args.batch_size,
    )
    ## save
    save_ids_and_embs(ids, embs, output_dir, index_postfix)


if __name__ == "__main__":
    main()
