import os
import random
import torch
from transformers import (
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)
from dataclasses import dataclass, field

from argparse import ArgumentParser

from src.parameters import DEFAULT_DATA_DIR
from src.training.data_reader import train
from src.utils import print_message, create_directory


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    lr: float = field(default=3e-06)
    maxsteps: int = field(default=400000)
    bsize: int = field(default=16)
    accumsteps: int = field(default=8)

    data_dir: str = field(default=DEFAULT_DATA_DIR)
    triples: str = field(default="triples.train.small.tsv")
    # output_dir = field(default="outputs.train/")

    similarity: str = field(default="cosine")
    query_maxlen: int = field(default=32)
    doc_maxlen: int = field(default=180)
    use_dense: bool = field(default=False)
    n: int = field(default=4096)
    k: float = field(default=0.005)
    normalize_sparse: bool = field(default=True)
    use_nonneg: bool = field(default=False)
    training_ins_num: int = field(default=10000)
    original_checkpoint: str = field(default=None)
    training_ins_start_from: int = field(default=None)
    static_out_k : bool = field(default = False)


def main():
    random.seed(12345)
    torch.manual_seed(1)

    # parser = ArgumentParser(
    #     description="Training ColBERT with <query, positive passage, negative passage> triples."
    # )

    # parser.add_argument("--lr", dest="lr", default=3e-06, type=float)
    # parser.add_argument("--maxsteps", dest="maxsteps", default=400000, type=int)
    # parser.add_argument("--bsize", dest="bsize", default=16, type=int)
    # parser.add_argument("--accum", dest="accumsteps", default=4, type=int)

    # parser.add_argument("--data_dir", dest="data_dir", default=DEFAULT_DATA_DIR)
    # parser.add_argument("--triples", dest="triples", default="triples.train.small.tsv")
    # #parser.add_argument("--output_dir", dest="output_dir", default="outputs.train/")

    # parser.add_argument(
    #     "--similarity", dest="similarity", default="cosine", choices=["cosine", "l2"]
    # )
    # parser.add_argument("--dim", dest="dim", default=128, type=int)
    # parser.add_argument("--query_maxlen", dest="query_maxlen", default=32, type=int)
    # parser.add_argument("--doc_maxlen", dest="doc_maxlen", default=180, type=int)
    # parser.add_argument("--use_dense", action="store_true")
    # parser.add_argument("--n", default=4096, type=int)
    # parser.add_argument("--k", default=0.005, type=float)
    # parser.add_argument(
    #     "--dont_normalize_sparse", dest="normalize_sparse", action="store_false"
    # )
    # parser.add_argument("--output_dir", default="./models/tpu")
    # parser.add_argument("--tpu_num_cores", default=8)
    # args = parser.parse_args()
    parser = HfArgumentParser((ModelArguments, TrainingArguments))

    args, training_args = parser.parse_args_into_dataclasses()
    args.input_arguments = args
    # args = model_args + training_args

    # TODO: Add resume functionality
    # TODO: Save the configuration to the checkpoint.
    # TODO: Extract common parser arguments/behavior into a class.

    create_directory(training_args.output_dir)

    assert args.bsize % args.accumsteps == 0, (
        (args.bsize, args.accumsteps),
        "The batch size must be divisible by the number of gradient accumulation steps.",
    )
    assert args.query_maxlen <= 512
    assert args.doc_maxlen <= 512

    args.triples = os.path.join(args.data_dir, args.triples)

    train(args, training_args)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
