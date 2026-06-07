from pypdf import PdfReader
import numpy as np
from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def extract_text(pdf_file):
    reader = PdfReader(pdf_file)

    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"

    return text


def chunk_text(text, chunk_size=800):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]


def get_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding


def get_top_chunks(chunks, question, top_k=3):
    question_embedding = get_embedding(question)

    scored_chunks = []

    for chunk in chunks:
        chunk_embedding = get_embedding(chunk)

        score = np.dot(question_embedding, chunk_embedding)
        scored_chunks.append((chunk, score))

    # sort by relevance
    scored_chunks.sort(key=lambda x: x[1], reverse=True)

    # return top_k chunks
    return [c[0] for c in scored_chunks[:top_k]]