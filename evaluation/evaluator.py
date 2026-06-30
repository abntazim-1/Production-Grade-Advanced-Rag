"""
Embedding-based evaluator — port from mini_rag.py.

Replaces the RAGAS LLM-judge approach which causes RagasOutputParserException
with sub-7B models. This is the exact fix from mini_rag.py:

  Faithfulness     = mean cosine similarity(answer, each context chunk)
                     → "Is the answer grounded in the retrieved documents?"
  Answer Relevancy = cosine similarity(question, answer)
                     → "Does the answer actually address the question?"

Both metrics return float in [0.0, 1.0]. Deterministic, fast (~10ms), 100%
reliable regardless of LLM output formatting.
"""
from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class EvalResult:
    faithfulness: Optional[float] = None
    answer_relevancy: Optional[float] = None

    def summary(self) -> str:
        lines = []
        for k, v in self.__dict__.items():
            val = f"{v:.3f}" if v is not None else "N/A"
            lines.append(f"  {k}: {val}")
        return "\n".join(lines)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    """
    Cosine similarity between two 1-D vectors.
    Matches mini_rag.py's _cosine() exactly.
    """
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


class EmbeddingEvaluator:
    """
    Embedding-based RAG evaluator — exact port of mini_rag.py's
    _embedding_eval() / evaluate_single_response() / evaluate_rag().

    Uses the already-loaded embedder — no extra model, no parser, no failures.
    """

    def __init__(self, embedder=None):
        self._embedder = embedder  # injected at startup from api/app.py

    @property
    def embedder(self):
        return self._embedder

    @embedder.setter
    def embedder(self, value):
        self._embedder = value

    def _embedding_eval(
        self,
        question: str,
        answer: str,
        contexts: list[str],
    ) -> dict:
        """
        Core metric computation.
        Exact port of mini_rag.py's _embedding_eval().
        """
        if not answer.strip():
            return {"faithfulness": 0.0, "answer_relevancy": 0.0}

        texts = [question, answer] + contexts
        vecs  = self._embedder.embed(texts)

        q_vec    = vecs[0]
        ans_vec  = vecs[1]
        ctx_vecs = vecs[2:]

        answer_relevancy = max(0.0, _cosine(q_vec, ans_vec))

        if ctx_vecs.shape[0] == 0:
            faithfulness = 0.0
        else:
            sims = [max(0.0, _cosine(ans_vec, cv)) for cv in ctx_vecs]
            faithfulness = float(np.mean(sims))

        return {
            "faithfulness":     round(faithfulness,     4),
            "answer_relevancy": round(answer_relevancy, 4),
        }

    def evaluate_single(
        self,
        question: str,
        answer: str,
        contexts: list[str],
        ground_truth: str = "",
    ) -> EvalResult:
        """
        Evaluate one chat turn.
        Matches mini_rag.py's evaluate_single_response().
        """
        if self._embedder is None:
            return EvalResult()
        scores = self._embedding_eval(question, answer, contexts)
        print(
            f"[EVAL] faithfulness={scores['faithfulness']:.3f}  "
            f"answer_relevancy={scores['answer_relevancy']:.3f}"
        )
        return EvalResult(
            faithfulness     = scores["faithfulness"],
            answer_relevancy = scores["answer_relevancy"],
        )

    def evaluate_batch(
        self,
        questions: list[str],
        answers: list[str],
        contexts_list: list[list[str]],
        ground_truths: list[str] = None,
    ) -> list[EvalResult]:
        """Batch evaluate — one EvalResult per question."""
        return [
            self.evaluate_single(q, a, c, gt or "")
            for q, a, c, gt in zip(
                questions,
                answers,
                contexts_list,
                ground_truths or [""] * len(questions),
            )
        ]

    def evaluate_rag(
        self,
        run_fn,
        questions: list[str],
        ground_truths: list[str] = None,
    ) -> dict:
        """
        Full pipeline evaluation over a question list.
        Matches mini_rag.py's evaluate_rag() exactly.
        run_fn(question) should return {"answer": str, "sources": [{"content": str}]}
        """
        import uuid
        print(f"[EVAL] Evaluating {len(questions)} queries (embedding-based, LLM-free)...")
        all_faith, all_relevancy = [], []

        for i, q in enumerate(questions):
            res      = run_fn(q)
            answer   = res["answer"]
            contexts = [s["content"] for s in res.get("sources", [])]
            scores   = self._embedding_eval(q, answer, contexts)
            all_faith.append(scores["faithfulness"])
            all_relevancy.append(scores["answer_relevancy"])
            print(
                f"  Q{i+1}: faithfulness={scores['faithfulness']:.3f}  "
                f"answer_relevancy={scores['answer_relevancy']:.3f}"
            )

        result = {
            "faithfulness":     round(float(np.mean(all_faith)),     4) if all_faith     else 0.0,
            "answer_relevancy": round(float(np.mean(all_relevancy)), 4) if all_relevancy else 0.0,
        }
        print(f"[EVAL] Aggregate → {result}")
        return result
