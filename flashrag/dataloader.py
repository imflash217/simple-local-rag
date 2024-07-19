import random

import numpy as np
import pandas as pd
import torch


def load_embeddings(
    embeddings_path: str = "../text_chunks_and_embeddings_df.csv",
) -> torch.Tensor:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    text_chunks_and_embeddings_df = pd.read_csv(embeddings_path)
    text_chunks_and_embeddings_df["embedding"] = text_chunks_and_embeddings_df[
        "embedding"
    ].apply(lambda x: np.fromstring(x.strip("[]"), sep=" "))

    pages_and_chunks = text_chunks_and_embeddings_df.to_dict(orient="records")

    embeddings = torch.tensor(
        np.array(text_chunks_and_embeddings_df["embedding"].to_list()),
        dtype=torch.float32,
    ).to(device=device)

    print(f"embeddings.shape = {embeddings.shape}")
    print(f"embeddings.dtype = {embeddings.dtype}")
    return embeddings
