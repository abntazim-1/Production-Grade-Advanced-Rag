"""
Conversation memory — port from mini_rag.py.

Matches mini_rag.py's memory implementation:
  - Bounded deque per session (maxlen=memory_maxlen from config)
  - Thread-safe with a lock
  - get_history() returns last 6 turns as a formatted string (matching mini_rag.py)
  - save_turn() / add() both supported
"""
import threading
from collections import deque
from dataclasses import dataclass
from config import get_settings

settings = get_settings()


@dataclass
class Turn:
    role: str    # "user" or "assistant"
    content: str


class ConversationMemory:
    def __init__(self, max_turns: int = None):
        self.max_turns = max_turns or settings.memory_maxlen
        self._lock: threading.Lock = threading.Lock()
        self.sessions: dict[str, deque] = {}

    def add(self, session_id: str, role: str, content: str) -> None:
        """Thread-safe turn save. Matches mini_rag.py's save_turn()."""
        with self._lock:
            if session_id not in self.sessions:
                self.sessions[session_id] = deque(maxlen=self.max_turns)
            self.sessions[session_id].append({"role": role, "content": content})

    # Alias used by mini_rag.py calling pattern
    def save_turn(self, session_id: str, role: str, content: str) -> None:
        self.add(session_id, role, content)

    def get_history(self, session_id: str, last_n: int = 6) -> str:
        """
        Returns the last N turns as a formatted string.
        Matches mini_rag.py's get_history() exactly:
          'User: ...\nAssistant: ...'
        """
        with self._lock:
            turns = list(self.sessions.get(session_id, deque()))[-last_n:]
        return "\n".join(
            f"{t['role'].title()}: {t['content']}" for t in turns
        )

    def format_for_prompt(self, session_id: str) -> str:
        """Alias for get_history() — used by agent/graph.py."""
        return self.get_history(session_id)

    def get_raw(self, session_id: str) -> list[dict]:
        """Returns raw list of {'role': str, 'content': str} dicts."""
        with self._lock:
            return list(self.sessions.get(session_id, deque()))

    def clear(self, session_id: str) -> None:
        with self._lock:
            self.sessions.pop(session_id, None)

    @property
    def active_sessions(self) -> int:
        return len(self.sessions)
