"""Prepare TinyStories as a token-id memmap for RivaQuant pretraining.

TinyStories (Eldan & Li, 2023) is the standard corpus for training genuinely
small from-scratch LMs that still produce coherent output — picked here
specifically because it de-risks the one real unknown in this project
(whether ternary weights converge at all) by keeping everything else about
the training run boring and well-precedented.
"""
import os

import numpy as np
import tiktoken
from datasets import load_dataset

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def prepare() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    enc = tiktoken.get_encoding("gpt2")
    eot = enc.eot_token  # <|endoftext|>, used as a story separator

    ds = load_dataset("roneneldan/TinyStories")

    for split, hf_split in (("train", "train"), ("val", "validation")):
        out_path = os.path.join(OUT_DIR, f"{split}.bin")
        if os.path.exists(out_path):
            print(f"skip {split}: already prepared")
            continue

        ids = []
        for row in ds[hf_split]:
            ids.extend(enc.encode_ordinary(row["text"]))
            ids.append(eot)

        arr = np.array(ids, dtype=np.uint16)
        arr.tofile(out_path)
        print(f"{split}: {len(arr):,} tokens -> {out_path}")


if __name__ == "__main__":
    prepare()
