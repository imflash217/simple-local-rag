import os

import fitz  # load the pymupdf package
import requests
from tqdm.auto import tqdm


def download_pdf(
    pdf_path: str = "human-nutrition-text.pdf",
    url: str = "https://pressbooks.oer.hawaii.edu/humannutrition2/open/download?type=pdf",
):
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
    """Performs minor formatting of text"""
    cleaned_text = text.replace("\n", " ").strip()
    return cleaned_text


def open_and_read_pdf(pdf_path: str) -> list[dict]:
    """"""



















