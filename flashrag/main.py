import os
import pprint
import re
import textwrap
from time import perf_counter as timer

import dataloader as dl
import fitz  # load the pymupdf package
import pandas as pd
import requests
import torch
import utils as ux
from sentence_transformers import SentenceTransformer, util
from spacy.lang.en import English
from tqdm.auto import tqdm


if __name__ == "__main__":
    query = (
        ux.get_random_query()
    )  # "symptoms of pellagra"  # "macronutrients functions"
    pdf_path = "human-nutrition-text.pdf"
    embeddings_df_save_path = "text_chunks_and_embeddings_df.csv"
    llm_model_id = "google/gemma-2b-it"
    config_use_quantization = False
    num_sentence_chunk_size = 10
    min_token_len_per_chunk = 30
    pages_and_chunks = []

    pages_and_texts = ux.open_and_read_pdf(pdf_path=pdf_path)

    # extracting sentences from the text
    nlp = English()
    nlp.add_pipe("sentencizer")
    for item in tqdm(pages_and_texts):
        item["sentences"] = list(nlp(item["text"]).sents)
        item["sentences"] = [str(sentence) for sentence in item["sentences"]]
        item["page_sentence_count_spacy"] = len(item["sentences"])

        # chunking our sentences together
        item["sentence_chunks"] = ux.split_list(
            input_list=item["sentences"], slice_size=num_sentence_chunk_size
        )
        item["num_chunks"] = len(item["sentence_chunks"])

        # split each chunk into its own item
        for sentence_chunk in item["sentence_chunks"]:
            chunk_dict = {}
            chunk_dict["page_number"] = item["page_number"]

            # join the sentences together into a paragraph-like structure
            joined_sentence_chunk = "".join(sentence_chunk).replace("  ", " ").strip()
            joined_sentence_chunk = re.sub(r"\.([A-Z])", r". \1", joined_sentence_chunk)
            chunk_dict["sentence_chunk"] = joined_sentence_chunk

            # get stats about the chunk
            chunk_dict["chunk_char_count"] = len(joined_sentence_chunk)
            chunk_dict["chunk_word_count"] = len(
                [word for word in joined_sentence_chunk.split(" ")]
            )
            chunk_dict["chunk_token_count"] = (
                len(joined_sentence_chunk) // 4
            )  # 1 token ~= 4 chars
            pages_and_chunks.append(chunk_dict)

    df = pd.DataFrame(pages_and_chunks)
    pages_and_chunks_over_min_token_len = df[
        df["chunk_token_count"] > min_token_len_per_chunk
    ].to_dict(orient="records")

    embedding_model = ux.get_embedding_model(
        model_name_or_path="all-mpnet-base-v2", device="cpu"
    )

    # NOTE
    # for item in tqdm(pages_and_chunks_over_min_token_len):
    # item["embedding"] = embedding_model.encode(item["sentence_chunk"])

    # advanced/faster batching approach to compute embeddings
    # text_chunks = [
    #     item["sentence_chunk"] for item in pages_and_chunks_over_min_token_len
    # ]
    # text_chunk_embeddings = embedding_model.encode(
    #     text_chunks,
    #     batch_size=2,  # you can use different batch sizes here for speed/performance, I found 32 works well for this use case
    #     convert_to_tensor=True,
    # )  # optional to return embeddings as tensor instead of array

    # pprint.pprint(pages_and_chunks_over_min_token_len[:2])
    # # NOTE:
    # text_chunks_and_embeddings_df = pd.DataFrame(pages_and_chunks_over_min_token_len)
    # text_chunks_and_embeddings_df.to_csv(embeddings_df_save_path, index=False)

    # sanity check the saved embeddings
    # text_chunks_and_embedding_df_load = pd.read_csv(embeddings_df_save_path)
    # print(text_chunks_and_embedding_df_load.head())
    reference_embeddings_reloaded, pages_and_chunks_reloaded = dl.load_embeddings(
        embeddings_path=embeddings_df_save_path
    )
    pprint.pprint(reference_embeddings_reloaded[:5])

    print("###" * 10)
    print(f"💬 Query: {query}")
    print(f"🤖 Result: ")

    query_embedding = ux.get_embeddings(
        input_text=query, embedding_model=embedding_model
    )
    print(f"query_embedding.shape = {query_embedding.shape}")
    scores, indices = ux.get_similarity_score(
        query_embedding=query_embedding,
        reference_embeddings=reference_embeddings_reloaded,
        n_topk=5,
    )

    for score, idx in zip(scores, indices):
        print(f"Score: {score:.4f}")
        print(f"Page # {pages_and_chunks_reloaded[idx]["page_number"]}")
        print(
            f"Text: {ux.print_textwrapped(pages_and_chunks_reloaded[idx]["sentence_chunk"])}"
        )
        print("........" * 10)

    llm_model, tokenizer = ux.get_llm_model(llm_model_id=llm_model_id)
    print("----" * 10)
    pprint.pprint(llm_model)
    print("----" * 10)

    # prompt engineering
    # user_prompt = query  # "What are macronutrients, and what roles do they play in the human body?"

    ####################
    context_items = ux.get_context_items(
        query=query,
        reference_embeddings=reference_embeddings_reloaded,
        pages_and_chunks=pages_and_chunks_reloaded,
    )
    prompt = ux.prompt_formatter(
        query=query, context_items=context_items, tokenizer=tokenizer
    )

    print("===========" * 10)
    print(prompt)

    model_response = ux.generate_llm_response(
        prompt=prompt, llm_model=llm_model, tokenizer=tokenizer
    )
    print("DONE")
