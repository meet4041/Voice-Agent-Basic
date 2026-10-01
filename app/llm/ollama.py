"""Free local LLM provider backed by an Ollama server on this computer."""

import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from app.llm.base import Message


class LLMError(RuntimeError):
    """Raised when the local Ollama model cannot produce a response."""


class OllamaProvider:
    def __init__(
        self,
        *,
        model: str,
        base_url: str = "http://127.0.0.1:11434",
        timeout_seconds: float = 120,
        max_tokens: int = 90,
        keep_alive: str = "30m",
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.max_tokens = max_tokens
        self.keep_alive = keep_alive

    def generate(self, messages: list[Message]) -> str:
        if not messages:
            raise ValueError("VOCA needs at least one message before it can ask the LLM.")

        payload = json.dumps(
            {
                "model": self.model,
                "messages": messages,
                "stream": False,
                "keep_alive": self.keep_alive,
                "options": {"num_predict": self.max_tokens},
            }
        ).encode("utf-8")
        request = Request(
            f"{self.base_url}/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                result = json.loads(response.read().decode("utf-8"))
        except HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            raise LLMError(f"Ollama could not use model '{self.model}': {detail}") from error
        except URLError as error:
            raise LLMError(
                "VOCA could not reach Ollama. Open the Ollama app, then make sure the configured model is downloaded."
            ) from error
        except (TimeoutError, json.JSONDecodeError) as error:
            raise LLMError("Ollama did not return a usable response. Please try again.") from error

        content = str(result.get("message", {}).get("content", "")).strip()
        if not content:
            raise LLMError("Ollama returned an empty response.")
        return content

    def warm(self) -> None:
        """Ask Ollama to load the configured model without generating a reply."""
        payload = json.dumps(
            {"model": self.model, "prompt": "", "keep_alive": self.keep_alive}
        ).encode("utf-8")
        request = Request(
            f"{self.base_url}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds):
                pass
        except (HTTPError, URLError, TimeoutError):
            # Startup remains usable if Ollama is temporarily unavailable.
            pass
