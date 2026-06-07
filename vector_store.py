import faiss
import numpy as np
from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class VectorStore:
    def __init__(self):
        self.index = None
        self.chunks = []

    def get_embedding(self, text):
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return np.array(response.data[0].embedding, dtype=np.float32)

    def build_index(self, chunks):
        self.chunks = chunks

        embeddings = [self.get_embedding(c) for c in chunks]
        embeddings = np.array(embeddings)

        self.index = faiss.IndexFlatL2(len(embeddings[0]))
        self.index.add(embeddings)

    def search(self, query, top_k=3):
        query_vec = self.get_embedding(query)

        distances, indices = self.index.search(
            np.array([query_vec]),
            top_k
        )

        results = [self.chunks[i] for i in indices[0]]

        return results