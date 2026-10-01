"""Selective memory extraction with no model-generated facts."""

from app.memory.store import Memory, MemoryStore


class MemoryService:
    def __init__(self, store: MemoryStore) -> None:
        self.store = store

    def consider_user_message(self, text: str) -> Memory | None:
        """Store only clear, direct statements that may be useful in future chats."""
        if not self.store.is_enabled():
            return None
        cleaned_text = text.strip()
        category = _category_for(cleaned_text.lower())
        if category is None:
            return None
        return self.store.add(cleaned_text, category=category)


def _category_for(text: str) -> str | None:
    project_phrases = ("i am building", "i'm building", "i am working on", "i'm working on", "i am developing", "i'm developing")
    goal_phrases = ("my goal is", "i want to learn", "i'm learning", "i am learning")
    preference_phrases = ("i prefer", "i like", "i don't like", "i do not like")
    if text.startswith(project_phrases):
        return "project"
    if text.startswith(goal_phrases):
        return "goal"
    if text.startswith(preference_phrases):
        return "preference"
    return None
