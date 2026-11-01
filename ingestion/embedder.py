import hashlib
import math

try:
    from fastembed import TextEmbedding
    HAS_FASTEMBED = True
except ImportError:
    HAS_FASTEMBED = False
    TextEmbedding = None


class VectorEmbedder:
    """Generates 384-dimensional dense embeddings normalized to unit L2 length."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        batch_size: int = 32,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.embedding_dim = 384
        self._model = None

        if HAS_FASTEMBED and TextEmbedding is not None:
            try:
                self._model = TextEmbedding(model_name=self.model_name)
            except Exception:
                self._model = None

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        all_embeddings: list[list[float]] = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            batch_vectors = self._embed_batch(batch)
            for vec in batch_vectors:
                norm = math.sqrt(sum(v * v for v in vec))
                if norm == 0.0:
                    normalized = [1.0 / math.sqrt(self.embedding_dim)] * self.embedding_dim
                else:
                    normalized = [v / norm for v in vec]
                all_embeddings.append(normalized)

        return all_embeddings

    def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        if self._model is not None:
            try:
                embeddings_generator = self._model.embed(batch)
                return [list(vec) for vec in embeddings_generator]
            except Exception:
                pass

        return [self._deterministic_fallback_vector(text) for text in batch]

    def _deterministic_fallback_vector(self, text: str) -> list[float]:
        import re

        words = re.findall(r"\b[a-zA-Z0-9_-]+\b", text.lower())
        vector = [0.0] * self.embedding_dim

        if not words:
            return [1.0 / math.sqrt(self.embedding_dim)] * self.embedding_dim

        # Unigram feature hashing
        for w in words:
            h = int(hashlib.sha256(w.encode("utf-8")).hexdigest(), 16)
            idx = h % self.embedding_dim
            sign = 1.0 if ((h >> 16) & 1) else -1.0
            vector[idx] += sign

        # Bigram feature hashing for phrase alignment
        for i in range(len(words) - 1):
            bigram = f"{words[i]}_{words[i+1]}"
            h = int(hashlib.sha256(bigram.encode("utf-8")).hexdigest(), 16)
            idx = h % self.embedding_dim
            sign = 1.5 if ((h >> 16) & 1) else -1.5
            vector[idx] += sign

        norm = math.sqrt(sum(x * x for x in vector))
        if norm == 0.0:
            return [1.0 / math.sqrt(self.embedding_dim)] * self.embedding_dim
        return [x / norm for x in vector]
