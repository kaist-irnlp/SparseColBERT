from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
from scipy import sparse
from tqdm import tqdm

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

    # query
    ## load data
    data = pd.read_csv(
        args.query, sep="\t", names=["id", "text"], chunksize=args.batch_size
    )
    ## process
    embs = []
    for chunk in tqdm(data):
        Q = chunk.text.values
        e = sparse.csr_matrix(model.query(Q).detach().cpu())
        embs.append(e)
    embs = sparse.vstack(embs)
    ## save
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"query_{index_postfix}"
    sparse.save_npz(output_path, embs)

    # docs
    ## load data
    data = pd.read_csv(
        args.collection, sep="\t", names=["id", "text"], chunksize=args.batch_size
    )
    ## process
    embs = []
    for chunk in tqdm(data):
        D = chunk.text.values
        e = sparse.csr_matrix(model.doc(D).detach().cpu())
        embs.append(e)
    embs = sparse.vstack(embs)
    ## save
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"doc_{index_postfix}"
    sparse.save_npz(output_path, embs)


if __name__ == "__main__":
    main()
