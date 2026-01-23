"""ARES integration for automatic RAG evaluation.

Stanford ARES provides LLM-as-judge evaluation with 95% confidence intervals.
Reference: https://github.com/stanford-futuredata/ARES
"""

from scripts.evaluation.ares.convert_to_ares import convert_gold_standard_to_ares
from scripts.evaluation.ares.generate_few_shot import generate_few_shot_examples
from scripts.evaluation.ares.run_evaluation import run_context_relevance_evaluation
from scripts.evaluation.ares.report import generate_evaluation_report

__all__ = [
    "convert_gold_standard_to_ares",
    "generate_few_shot_examples",
    "run_context_relevance_evaluation",
    "generate_evaluation_report",
]
