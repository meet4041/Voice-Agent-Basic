"""Contracts shared by language-model providers."""

from typing import Protocol, TypedDict


class Message(TypedDict):
    role: str
    content: str


class LLMProvider(Protocol):
    def generate(self, messages: list[Message]) -> str:
        """Return one complete assistant response for the supplied conversation."""
