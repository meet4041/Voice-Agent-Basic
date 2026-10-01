"""Ollama embedding client for free, local semantic search."""

import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class EmbeddingError(RuntimeError):
    """Raised when Ollama cannot create a local embedding."""


class OllamaEmbeddingProvider:
    def __init__(self, *, model: str, base_url: str, timeout_seconds: float) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        payload = json.dumps({"model": self.model, "input": texts}).encode("utf-8")
        request = Request(
            f"{self.base_url}/api/embed",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                result = json.loads(response.read().decode("utf-8"))
        except HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            raise EmbeddingError(f"Ollama could not use embedding model '{self.model}': {detail}") from error
        except URLError as error:
            raise EmbeddingError(
                "VOCA could not reach Ollama. Open Ollama and make sure the embedding model is downloaded."
            ) from error
        except (TimeoutError, json.JSONDecodeError) as error:
            raise EmbeddingError("Ollama did not return usable search embeddings.") from error

        embeddings = result.get("embeddings", [])
        if len(embeddings) != len(texts):
            raise EmbeddingError("Ollama returned an unexpected number of search embeddings.")
        return [[float(value) for value in embedding] for embedding in embeddings]
