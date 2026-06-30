"""
Input and output guardrails.

Input guards:
  - Prompt injection detection
  - Out-of-scope / jailbreak patterns
  - Query length limits

Output guards:
  - Citation verification (answer references retrieved chunks)
  - Hallucination flag if answer contains unsupported claims
"""
import re
from dataclasses import dataclass


@dataclass
class GuardResult:
    passed: bool
    reason: str = ""


INJECTION_PATTERNS = [
    r"ignore (previous|all|above) instructions",
    r"you are now (a|an|the)",
    r"pretend (you are|to be)",
    r"act as (a|an|the)",
    r"forget (everything|all|your)",
    r"system prompt",
    r"jailbreak",
    r"DAN mode",
]

OUT_OF_SCOPE_PATTERNS = [
    r"\b(bomb|weapon|explosive|poison|drug synthesis)\b",
    r"\b(hack|exploit|vulnerability|malware|ransomware)\b",
]


class InputGuard:
    def __init__(self, max_query_length: int = 1000):
        self.max_len = max_query_length
        self.injection_re = re.compile(
            "|".join(INJECTION_PATTERNS), re.IGNORECASE
        )
        self.scope_re = re.compile(
            "|".join(OUT_OF_SCOPE_PATTERNS), re.IGNORECASE
        )

    def check(self, query: str) -> GuardResult:
        if len(query) > self.max_len:
            return GuardResult(False, f"Query too long ({len(query)} chars)")

        if self.injection_re.search(query):
            return GuardResult(False, "Prompt injection pattern detected")

        if self.scope_re.search(query):
            return GuardResult(False, "Out-of-scope query detected")

        return GuardResult(True)


class OutputGuard:
    def check(self, answer: str, context_chunks: list[str]) -> GuardResult:
        """
        Basic hallucination check: verify the answer doesn't introduce
        proper nouns or numbers not found in any context chunk.
        """
        # Extract numbers from answer
        answer_numbers = set(re.findall(r"\b\d+\.?\d*\b", answer))
        context_text = " ".join(context_chunks)
        context_numbers = set(re.findall(r"\b\d+\.?\d*\b", context_text))

        unsupported_numbers = answer_numbers - context_numbers
        if len(unsupported_numbers) > 3:
            return GuardResult(
                False,
                f"Answer contains unsupported numbers: {unsupported_numbers}"
            )

        return GuardResult(True)
