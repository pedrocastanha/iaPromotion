import os
import json
from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer
from faiss import IndexFlatL2, write_index, read_index

#Melhorar: Criar mais chunks a partir do documento, e melhorar a lógica de embbendings, pra economia de tokens

class EmbeddingManager:
    def __init__(
        self,
        model_name: str = "paraphrase-MiniLM-L6-v2",
        vector_store_path: str = "src/data/faiss_index.bin",
        texts_store_path: str = "src/data/faiss_texts.json",
    ):
        self.model_name: str = model_name
        self.model: SentenceTransformer = SentenceTransformer(model_name)

        self.vector_store_path = vector_store_path
        self.texts_store_path = texts_store_path

        self.index = None
        self.texts: List[str] = []

        self._ensure_dirs()
        self._load_or_create_vector_store()
        self._load_texts()

    def _ensure_dirs(self):
        os.makedirs(os.path.dirname(self.vector_store_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.texts_store_path), exist_ok=True)

    def _load_or_create_vector_store(self):
        if os.path.exists(self.vector_store_path):
            print(f"loading vector store from {self.vector_store_path}")
            self.index = read_index(self.vector_store_path)
        else:
            print("creating vector store")
            dim = self.model.get_sentence_embedding_dimension()
            self.index = IndexFlatL2(dim)

    def _load_texts(self):
        if os.path.exists(self.texts_store_path):
            try:
                with open(self.texts_store_path, "r", encoding="utf-8") as f:
                    self.texts = json.load(f)
            except Exception:
                self.texts = []
        else:
            self.texts = []

    def _save_texts(self):
        with open(self.texts_store_path, "w", encoding="utf-8") as f:
            json.dump(self.texts, f, ensure_ascii=False, indent=2)

    def create_embeddings(self, texts: List[str]) -> np.ndarray:
        print(f"creating embeddings from {len(texts)} texts")
        embeddings = self.model.encode(
            texts, convert_to_numpy=True, show_progress_bar=True
        ).astype(np.float32)
        print("embeddings created")
        return embeddings

    def add_embedding(self, text: str):
        if not text:
            return
        self.add_embeddings([text])

    def add_embeddings(self, texts: List[str]):
        if not texts:
            return

        embeddings = self.create_embeddings(texts)
        self.index.add(embeddings)
        self.texts.extend(texts)

        write_index(self.index, self.vector_store_path)
        self._save_texts()

        print(f"{len(texts)} texts added to vector store")
        print(f"vector store saved in {self.vector_store_path}")
        print(f"texts saved in {self.texts_store_path}")

    def search_vector_store(self, query: str, k: int = 5) -> List[str]:
        if self.index is None or self.index.ntotal == 0:
            print("vector store is empty")
            return []

        k = max(1, min(k, self.index.ntotal))
        query_embedding = self.model.encode(
            query, convert_to_numpy=True, show_progress_bar=False
        ).astype(np.float32).reshape(1, -1)

        distances, indices = self.index.search(query_embedding, k)

        retrieved_texts = [
            self.texts[i] for i in indices[0] if 0 <= i < len(self.texts)
        ]
        print(f"search for '{query}' returned {len(retrieved_texts)} results")
        return retrieved_texts