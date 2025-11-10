from typing import List, Iterable
import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingService:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()

    def embed(self, texts: Iterable[str], normalize: bool = True, batch_size: int = 32) -> List[List[float]]:
        vecs = self.model.encode(list(texts), batch_size=batch_size, show_progress_bar=True)
        if normalize:
            norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
            vecs = vecs / norms
        return vecs.tolist()

    def embed_one(self, text: str, normalize: bool = True) -> List[float]:
        return self.embed([text], normalize=normalize)[0]