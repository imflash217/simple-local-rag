import os
import pprint
import re
import textwrap
from time import perf_counter as timer
import random
import fitz  # load the pymupdf package
import pandas as pd
import requests
import torch
from sentence_transformers import SentenceTransformer, util
from spacy.lang.en import English
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import is_flash_attn_2_available


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


def get_llm_model(llm_model_id, print_model_details=True):
    attn_implementation = "sdpa"  # scaled dot product attention
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=llm_model_id
    )
    llm_model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=llm_model_id,
        torch_dtype=torch.float16,
        quantization_config=None,
        low_cpu_mem_usage=False,
        attn_implementation=attn_implementation,
    )

    if print_model_details:
        llm_num_params = sum([param.numel() for param in llm_model.parameters()])
        llm_mem_params = sum(
            [
                param.nelement() * param.element_size()
                for param in llm_model.parameters()
            ]
        )
        llm_mem_buffers = sum(
            [buf.nelement() * buf.element_size() for buf in llm_model.buffers()]
        )

        # calculate various model sizes
        llm_mem_bytes = llm_mem_params + llm_mem_buffers  # in bytes
        llm_mem_mb = llm_mem_bytes / (1024**2)  # in megabytes
        llm_mem_gb = llm_mem_bytes / (1024**3)  # in gigabytes

        print("----" * 10)
        pprint.pprint(
            {
                "llm_num_parameters": llm_num_params,
                "llm_mem_bytes": llm_mem_bytes,
                "llm_mem_mb": round(llm_mem_mb, 2),
                "llm_mem_gb": round(llm_mem_gb, 2),
            }
        )
        print("----" * 10)

    return llm_model, tokenizer


def get_random_query():
    # Nutrition-style questions generated with GPT4
    gpt4_questions = [
        "What are the macronutrients, and what roles do they play in the human body?",
        "How do vitamins and minerals differ in their roles and importance for health?",
        "Describe the process of digestion and absorption of nutrients in the human body.",
        "What role does fibre play in digestion? Name five fibre containing foods.",
        "Explain the concept of energy balance and its importance in weight management.",
    ]

    # Manually created question list
    manual_questions = [
        "How often should infants be breastfed?",
        "What are symptoms of pellagra?",
        "How does saliva help with digestion?",
        "What is the RDI for protein per day?",
        "water soluble vitamins",
    ]

    query_list = gpt4_questions + manual_questions
    query = random.choice(query_list)
    return query


def get_context_items(query: str, reference_embeddings, pages_and_chunks) -> list:
    embedding_model = get_embedding_model()
    query_embedding = get_embeddings(input_text=query, embedding_model=embedding_model)
    scores, indices = get_similarity_score(
        query_embedding=query_embedding, reference_embeddings=reference_embeddings
    )

    context_items = [pages_and_chunks[i] for i in indices]

    return context_items


def apply_prompt_template(query: str, tokenizer: AutoTokenizer):
    dialogue_template = [{"role": "user", "content": query}]
    prompt = tokenizer.apply_chat_template(
        conversation=dialogue_template, tokenize=False, add_generation_prompt=True
    )
    return prompt


def prompt_formatter(
    query: str, context_items: list[dict], tokenizer: AutoTokenizer
) -> str:
    context = "- " + "\n-".join([item["sentence_chunk"] for item in context_items])
    base_prompt = f"""Based on the following context items, please answer the query.
Give yourself room to think by extracting relevant passages from the context before answering the query.
Don't return the thinking, only return the answer.
Make sure your answers are as explanatory as possible.
Use the following examples as reference for the ideal answer style.
\nExample 1:
Query: What are the fat-soluble vitamins?
Answer: The fat-soluble vitamins include Vitamin A, Vitamin D, Vitamin E, and Vitamin K. These vitamins are absorbed along with fats in the diet and can be stored in the body's fatty tissue and liver for later use. Vitamin A is important for vision, immune function, and skin health. Vitamin D plays a critical role in calcium absorption and bone health. Vitamin E acts as an antioxidant, protecting cells from damage. Vitamin K is essential for blood clotting and bone metabolism.
\nExample 2:
Query: What are the causes of type 2 diabetes?
Answer: Type 2 diabetes is often associated with overnutrition, particularly the overconsumption of calories leading to obesity. Factors include a diet high in refined sugars and saturated fats, which can lead to insulin resistance, a condition where the body's cells do not respond effectively to insulin. Over time, the pancreas cannot produce enough insulin to manage blood sugar levels, resulting in type 2 diabetes. Additionally, excessive caloric intake without sufficient physical activity exacerbates the risk by promoting weight gain and fat accumulation, particularly around the abdomen, further contributing to insulin resistance.
\nExample 3:
Query: What is the importance of hydration for physical performance?
Answer: Hydration is crucial for physical performance because water plays key roles in maintaining blood volume, regulating body temperature, and ensuring the transport of nutrients and oxygen to cells. Adequate hydration is essential for optimal muscle function, endurance, and recovery. Dehydration can lead to decreased performance, fatigue, and increased risk of heat-related illnesses, such as heat stroke. Drinking sufficient water before, during, and after exercise helps ensure peak physical performance and recovery.
\nNow use the following context items to answer the user query:
{context}
\nRelevant passages: <extract relevant passages from the context here>
User query: {query}
Answer:"""

    prompt = apply_prompt_template(query=base_prompt, tokenizer=tokenizer)

    return prompt


def generate_llm_response(prompt, llm_model, tokenizer):
    # tokenize the input text
    input_ids = tokenizer(prompt, return_tensors="pt")
    # pprint.pprint(input_ids)

    # generate from the LLM
    output_ids = llm_model.generate(
        **input_ids, temperature=0.7, do_sample=True, max_new_tokens=256
    )
    print("----" * 10)
    print(f"Model Output token ids:\n{output_ids}")

    # decode the LLM-generate token ids into text-strings
    output_decoded = tokenizer.decode(output_ids[0])
    print("----" * 10)
    print(f"Model output (decoded):\n{output_decoded}")

    # print()
    # format the model output
    output_formatted = (
        output_decoded.replace(prompt, "").replace("<bos>", "").replace("<eos>", "")
    )
    print("----" * 10)
    print(output_formatted)
    return output_formatted
