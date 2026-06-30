"""
Conversation memory.
Stores (role, content) pairs per session.
Compresses old turns to stay within context window.
"""
from collections import deque
from dataclasses import dataclass, field


@dataclass
class Turn:
    role: str   # "user" or "assistant"
    content: str


class ConversationMemory:
    def __init__(self, max_turns: int = 10, compress_after: int = 6):
        self.sessions: dict[str, deque[Turn]] = {}
        self.max_turns = max_turns
        self.compress_after = compress_after  # summarize turns older than this

    def add(self, session_id: str, role: str, content: str) -> None:
        if session_id not in self.sessions:
            self.sessions[session_id] = deque(maxlen=self.max_turns)
        self.sessions[session_id].append(Turn(role=role, content=content))

    def get_history(self, session_id: str) -> list[Turn]:
        return list(self.sessions.get(session_id, []))

    def format_for_prompt(self, session_id: str) -> str:
        """Returns conversation history as a string block for the LLM prompt."""
        turns = self.get_history(session_id)
        if not turns:
            return ""
        lines = []
        for t in turns:
            prefix = "User" if t.role == "user" else "Assistant"
            lines.append(f"{prefix}: {t.content}")
        return "\n".join(lines)

    def clear(self, session_id: str) -> None:
        self.sessions.pop(session_id, None)
