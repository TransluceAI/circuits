import random

import numpy as np
import torch


def get_batch(
    hf_model_id: str,
    dataset_name: str,
    batch_size: int,
    seq_len: int,
    bos_token_id: int,
):
    """
    Loads a batch of sequences from a dataset tokenized using Dami's `tokenize_dataset.py` script.
    """

    rng = random.Random(64)
    while True:
        # Reconstruct memory map with very batch to avoid a memory leak.
        data_path = f"data/{dataset_name}_{hf_model_id.replace('/', '_')}.bin"
        data = np.memmap(data_path, dtype=np.uint32, mode="r")

        # Loop until we have a batch of the desired size
        batch, start_ix = [], []
        while len(batch) < batch_size:
            start = rng.randint(0, len(data) - seq_len + 1)
            seq = data[start : start + seq_len]
            # Check whether sequence includes two documents.
            if bos_token_id in seq[1:]:
                continue
            batch.append(torch.from_numpy(seq.astype(np.int64)))
            start_ix.append(start)

        yield batch, start_ix
