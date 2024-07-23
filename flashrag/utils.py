import os
import pprint
import re
import textwrap
from time import perf_counter as timer

import fitz  # load the pymupdf package
import pandas as pd
import requests
import torch
from sentence_transformers import SentenceTransformer, util
from spacy.lang.en import English
from tqdm.auto import tqdm


def download_pdf(
    pdf_path: str = "human-nutrition-text.pdf",
    url: str = "https://pressbooks.oer.hawaii.edu/humannutrition2/open/download?type=pdf",
):
    """
    Downloads a PDF file from a given URL and saves it to a specified path.

    Args:
        pdf_path (str): The path where the PDF will be saved. Defaults to "human-nutrition-text.pdf".
        url (str): The URL from which to download the PDF. Defaults to a specific human nutrition text PDF.

    Returns:
        None

    Raises:
        None

    Notes:
        If the file already exists at the specified path, the function will not download the file again.
    """
    if not os.path.exists(pdf_path):
        # The URL of the PDF you want to download
        filename = pdf_path
        response = requests.get(url=url)

        if response.status_code == 200:
            with open(filename, "wb") as f:
                f.write(response.content)
            print(f"The file is downloaded and saved as {filename}")
        else:
            print(
                f"Failed to download the file: {url}\nStatus Code: {response.status_code}"
            )
    else:
        print(f"The file ({pdf_path}) already exists.")


def text_formatter(text: str) -> str:
    """
    Performs minor formatting of the given text by replacing newline characters with spaces
    and stripping leading and trailing whitespace.

    Args:
        text (str): The input text to be formatted.

    Returns:
        str: The formatted text with newline characters replaced by spaces and leading/trailing whitespace removed.
    """
    cleaned_text = text.replace("\n", " ").strip()
    return cleaned_text


def open_and_read_pdf(pdf_path: str) -> list[dict]:
    """
    Opens a PDF file located at the specified path and reads its content.

    Args:
        pdf_path (str): The path to the PDF file to be read.

    Returns:
        list[dict]: A list of dictionaries representing various statistics about the PDF file.
    """
    doc = fitz.open(pdf_path)
    pages_and_texts = []
    for page_number, page in tqdm(enumerate(doc)):
        text = page.get_text()
        text = text_formatter(text)
        pages_and_texts.append(
            {
                "page_number": page_number - 41,
                "page_char_count": len(text),
                "page_word_count": len(text.split(" ")),
                "page_sentence_count_raw": len(text.split(". ")),
                "page_token_count": len(text) // 4,
                "text": text,
            }
        )
    return pages_and_texts


def split_list(input_list: list, slice_size: int) -> list[list[str]]:
    """
    Splits the input_list into sublists of size slice_size (or as close as possible)
    For example: a list of 17 sentences will be split into two lists of [[10],[7]]
    """
    return [
        input_list[i : i + slice_size] for i in range(0, len(input_list), slice_size)
    ]


def get_embedding_model(
    model_name_or_path: str = "all-mpnet-base-v2", device: str = "cpu"
):
    # embedding the sentences
    embedding_model = SentenceTransformer(
        model_name_or_path=model_name_or_path, device=device
    )

    embedding_model.to(device)
    return embedding_model


def print_textwrapped(text, wrap_length=90):
    wrapped_text = textwrap.fill(text=text, width=wrap_length)
    print(wrapped_text)
    return wrapped_text


def get_similarity_score(query_embedding, reference_embeddings, n_topk=5):
    start_time = timer()
    dot_scores = util.dot_score(query_embedding, reference_embeddings)[0]
    end_time = timer()

    print(
        f"Time taken to get scores on {len(reference_embeddings)} reference_embeddings: {end_time-start_time:.5f} seconds."
    )

    scores, indices = torch.topk(dot_scores, k=n_topk)
    return scores, indices


def get_embeddings(
    input_text: list, embedding_model: SentenceTransformer
) -> torch.tensor:
    assert isinstance(input_text, list) or isinstance(input_text, str)
    assert isinstance(embedding_model, SentenceTransformer)
    embeddings = embedding_model.encode(input_text, convert_to_tensor=True)
    return embeddings
