"""
RAGAS evaluation.
Runs async after each query (or in batch on a test set).
Metrics: faithfulness, context_precision, context_recall, answer_relevancy.
"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class EvalResult:
    faithfulness: Optional[float] = None
    context_precision: Optional[float] = None
    context_recall: Optional[float] = None
    answer_relevancy: Optional[float] = None

    def summary(self) -> str:
        lines = []
        for k, v in self.__dict__.items():
            val = f"{v:.3f}" if v is not None else "N/A"
            lines.append(f"  {k}: {val}")
        return "\n".join(lines)


class RAGASEvaluator:
    """
    Wrapper around the ragas library.
    Requires: pip install ragas datasets
    """

    def __init__(self, llm=None, embeddings=None):
        try:
            from ragas import evaluate
            from ragas.metrics import (
                faithfulness,
                context_precision,
                context_recall,
                answer_relevancy,
            )
            self.evaluate = evaluate
            self.metrics = [
                faithfulness,
                context_precision,
                context_recall,
                answer_relevancy,
            ]
            self.available = True
        except ImportError:
            print("[Evaluator] ragas not installed. Skipping evaluation.")
            self.available = False

    def evaluate_single(
        self,
        question: str,
        answer: str,
        contexts: list[str],
        ground_truth: str = "",
    ) -> EvalResult:
        if not self.available:
            return EvalResult()

        from datasets import Dataset

        data = {
            "question": [question],
            "answer": [answer],
            "contexts": [contexts],
            "ground_truth": [ground_truth or answer],
        }
        dataset = Dataset.from_dict(data)

        try:
            result = self.evaluate(dataset, metrics=self.metrics)
            df = result.to_pandas()
            row = df.iloc[0]
            return EvalResult(
                faithfulness=row.get("faithfulness"),
                context_precision=row.get("context_precision"),
                context_recall=row.get("context_recall"),
                answer_relevancy=row.get("answer_relevancy"),
            )
        except Exception as e:
            print(f"[Evaluator] RAGAS error: {e}")
            return EvalResult()

    def evaluate_batch(
        self,
        questions: list[str],
        answers: list[str],
        contexts_list: list[list[str]],
        ground_truths: list[str] = None,
    ) -> list[EvalResult]:
        return [
            self.evaluate_single(q, a, c, gt or "")
            for q, a, c, gt in zip(
                questions,
                answers,
                contexts_list,
                ground_truths or [""] * len(questions),
            )
        ]
