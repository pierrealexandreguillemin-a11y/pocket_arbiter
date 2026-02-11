# Gold Standard Validation by LLM-as-Judge

## Why validate the GS?

The Gold Standard (614 items) must be validated **before** building the RAG retriever. An LLM judge provides independent, item-level quality assurance with justifications for each verdict.

## The 3 criteria

Each GS item `(question, chunk, expected_answer)` is evaluated independently:

| Criterion | What it checks | Detects |
|-----------|---------------|---------|
| **Context Relevance** | Is the chunk relevant for this question? | Bad chunk mapping |
| **Answer Faithfulness** | Is the answer grounded in the chunk? | Hallucinated answers |
| **Answer Relevance** | Does the answer address the question? | Question-answer misalignment |

3 separate LLM calls per item ensure criteria independence.

## Reading the results

- **CSV** (`validation_results_{corpus}.csv`): Per-item verdicts with justifications
- **JSON** (`validation_report_{corpus}.json`): Aggregate pass rates, agreement metrics, flagged items

### Agreement metrics

- **Raw agreement**: Simple % of GS-creator vs judge agreement
- **Cohen's Kappa**: Chance-corrected agreement (biased by prevalence paradox when most items pass)
- **Gwet's AC1**: Robust to prevalence paradox (binary nominal) - **primary metric**

## Usage

```bash
# Mock mode (no LLM, for testing)
python -m scripts.evaluation.gs_validate --corpus fr --backend mock

# With Ollama (local)
python -m scripts.evaluation.gs_validate --corpus fr --backend ollama --model mistral:latest

# Quick test (10 items)
python -m scripts.evaluation.gs_validate --corpus fr --backend ollama --model mistral:latest --max-items 10
```

## Workflow for corrections

1. Run validation, review flagged items in the CSV
2. For each FAIL: check the justification and the source chunk
3. Fix the GS item (update chunk mapping, answer, or question)
4. Re-run validation to confirm fixes

## References

- "A Survey on LLM-as-a-Judge" (arXiv:2411.15594)
- "LLM Judge for Legal RAG with Gwet's AC" (arXiv:2509.12382)
- ARES (arXiv:2311.09476) Section 3.2
- ISO 29119: exhaustive evaluation when N <= 1000
