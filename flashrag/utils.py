import os
import pprint
import re
import textwrap

import fitz  # load the pymupdf package
import pandas as pd
import requests
from sentence_transformers import SentenceTransformer
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


if __name__ == "__main__":
    pdf_path = "human-nutrition-text.pdf"
    num_sentence_chunk_size = 10
    min_token_len_per_chunk = 30
    pages_and_texts = open_and_read_pdf(pdf_path=pdf_path)
    pages_and_chunks = []

    # extracting sentences from the text
    nlp = English()
    nlp.add_pipe("sentencizer")
    for item in tqdm(pages_and_texts):
        item["sentences"] = list(nlp(item["text"]).sents)
        item["sentences"] = [str(sentence) for sentence in item["sentences"]]
        item["page_sentence_count_spacy"] = len(item["sentences"])

        # chunking our sentences together
        item["sentence_chunks"] = split_list(
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

    embedding_model = get_embedding_model(
        model_name_or_path="all-mpnet-base-v2", device="cpu"
    )

    for item in tqdm(pages_and_chunks_over_min_token_len):
        item["embedding"] = embedding_model.encode(item["sentence_chunk"])

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
    text_chunks_and_embeddings_df = pd.DataFrame(pages_and_chunks_over_min_token_len)
    embeddings_df_save_path = "text_chunks_and_embeddings_df.csv"
    text_chunks_and_embeddings_df.to_csv(embeddings_df_save_path, index=False)

    # sanity check the saved embeddings
    text_chunks_and_embedding_df_load = pd.read_csv(embeddings_df_save_path)
    print(text_chunks_and_embedding_df_load.head())
