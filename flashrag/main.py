from time import perf_counter as timer

import dataloader as dl
import torch
import utils as utils
from sentence_transformers import util


def get_similarity_score(query_embedding, reference_embeddings):
    start_time = timer()
    dot_scores = util.dot_score(query_embedding, reference_embeddings)[0]
    end_time = timer()

    print(
        f"Time taken to get scores on {len(reference_embeddings)} reference_embeddings: {end_time-start_time:.5f} seconds."
    )

    top_results_dot_product = torch.topk(dot_scores, k=5)
    return top_results_dot_product
