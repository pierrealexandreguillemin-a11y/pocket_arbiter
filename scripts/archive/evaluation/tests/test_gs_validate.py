"""Tests for GS validation by LLM-as-judge.

ISO Reference: ISO 29119 - Test design
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts.evaluation.gs_validate import (
    CRITERIA,
    _build_judge_prompt,
    _call_llm,
    _compute_agreement,
    _generate_report,
    _judge_item,
    _load_gs_items,
    _parse_verdict,
    validate_gs,
)

# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture()
def sample_results() -> list[dict]:
    """Sample results for report generation."""
    return [
        {
            "gs_id": "gs:001",
            "question": "Q1?",
            "chunk_id": "chunk-001",
            "cr_pass": True,
            "cr_reason": "Chunk relevant",
            "af_pass": True,
            "af_reason": "Answer grounded",
            "ar_pass": True,
            "ar_reason": "Answer addresses question",
        },
        {
            "gs_id": "gs:002",
            "question": "Q2?",
            "chunk_id": "chunk-002",
            "cr_pass": True,
            "cr_reason": "Chunk relevant",
            "af_pass": False,
            "af_reason": "Answer adds unsupported claim",
            "ar_pass": True,
            "ar_reason": "Answer relevant",
        },
        {
            "gs_id": "gs:003",
            "question": "Q3?",
            "chunk_id": "chunk-003",
            "cr_pass": False,
            "cr_reason": "Chunk about different topic",
            "af_pass": True,
            "af_reason": "Answer in chunk",
            "ar_pass": False,
            "ar_reason": "Answer does not address question",
        },
    ]


# ── _parse_verdict ────────────────────────────────────────────────────


class TestParseVerdict:
    """Tests for _parse_verdict."""

    def test_pass_verdict(self) -> None:
        is_pass, reason = _parse_verdict("[[PASS]] The chunk is relevant.")
        assert is_pass is True
        assert "relevant" in reason

    def test_fail_verdict(self) -> None:
        is_pass, reason = _parse_verdict(
            "[[FAIL]] The chunk is about a different topic."
        )
        assert is_pass is False
        assert "different topic" in reason

    def test_pass_case_insensitive(self) -> None:
        is_pass, _ = _parse_verdict("[[pass]] ok")
        assert is_pass is True

    def test_fail_takes_precedence(self) -> None:
        is_pass, _ = _parse_verdict("[[PASS]] but actually [[FAIL]]")
        assert is_pass is False

    def test_fallback_to_keyword(self) -> None:
        is_pass, _ = _parse_verdict("The answer passes the test.")
        assert is_pass is True

    def test_no_verdict_defaults_false(self) -> None:
        is_pass, _ = _parse_verdict("I don't know")
        assert is_pass is False

    def test_reason_truncation(self) -> None:
        long_reason = "x" * 300
        _, reason = _parse_verdict(f"[[PASS]] {long_reason}")
        assert len(reason) <= 200

    def test_mock_response(self) -> None:
        is_pass, reason = _parse_verdict("[[PASS]] Mock evaluation - no LLM call made.")
        assert is_pass is True
        assert "Mock" in reason

    def test_mixed_case_reason_extracted(self) -> None:
        """CRITICAL-2 fix: mixed-case markers must be stripped from reason."""
        is_pass, reason = _parse_verdict("[[Pass]] The chunk is relevant.")
        assert is_pass is True
        assert "[[" not in reason
        assert "relevant" in reason

    def test_mixed_case_fail_reason(self) -> None:
        is_pass, reason = _parse_verdict("[[Fail]] Not supported by chunk.")
        assert is_pass is False
        assert "[[" not in reason


# ── _build_judge_prompt ───────────────────────────────────────────────


class TestBuildJudgePrompt:
    """Tests for prompt construction."""

    def test_context_relevance_prompt(self) -> None:
        prompt = _build_judge_prompt(
            "context_relevance", "What is X?", "Chunk about X", "X is Y"
        )
        assert "QUESTION:" in prompt
        assert "CHUNK:" in prompt
        assert "EXPECTED ANSWER" not in prompt

    def test_answer_faithfulness_prompt(self) -> None:
        prompt = _build_judge_prompt(
            "answer_faithfulness", "What is X?", "Chunk about X", "X is Y"
        )
        assert "CHUNK:" in prompt
        assert "EXPECTED ANSWER:" in prompt
        assert "QUESTION" not in prompt

    def test_answer_relevance_prompt(self) -> None:
        prompt = _build_judge_prompt(
            "answer_relevance", "What is X?", "Chunk about X", "X is Y"
        )
        assert "QUESTION:" in prompt
        assert "EXPECTED ANSWER:" in prompt
        assert "CHUNK" not in prompt


# ── _judge_item ───────────────────────────────────────────────────────


class TestJudgeItem:
    """Tests for _judge_item with mock backend."""

    def test_mock_returns_pass(self) -> None:
        is_pass, reason = _judge_item(
            "Question?", "Chunk text", "Answer", "context_relevance", "mock"
        )
        assert is_pass is True
        assert "Mock" in reason

    def test_all_criteria_work(self) -> None:
        for criterion in CRITERIA:
            is_pass, reason = _judge_item("Q?", "Chunk", "Answer", criterion, "mock")
            assert is_pass is True
            assert isinstance(reason, str)

    def test_invalid_criterion_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown criterion"):
            _judge_item("Q?", "C", "A", "invalid_metric", "mock")

    def test_invalid_backend_raises(self) -> None:
        with pytest.raises(ValueError, match="Unsupported backend"):
            _judge_item("Q?", "C", "A", "context_relevance", "invalid_backend")


class TestCallLlmAnthropic:
    """Tests for _call_llm with anthropic backend."""

    def test_call_llm_anthropic_ok(self) -> None:
        """Anthropic backend returns response text and passes correct SDK params."""
        mock_block = type("TextBlock", (), {"text": " [[PASS]] Looks good. "})()
        mock_message = type("Message", (), {"content": [mock_block]})()

        captured: dict = {}

        def mock_create(**kw: object) -> object:
            captured.update(kw)
            return mock_message

        mock_messages = type("Messages", (), {"create": staticmethod(mock_create)})()

        class MockAnthropic:
            def __init__(self, **kw: object) -> None:
                captured["_api_key"] = kw.get("api_key")
                self.messages = mock_messages

        mock_anthropic = type("module", (), {"Anthropic": MockAnthropic})()

        with (
            patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}),
            patch.dict("sys.modules", {"anthropic": mock_anthropic}),
        ):
            result = _call_llm(
                "sys prompt", "usr prompt", "anthropic", "claude-sonnet-4-5-20250929"
            )

        # Verify .strip() applied
        assert result == "[[PASS]] Looks good."
        # Verify SDK parameters (ISO 29119: all inputs to external APIs)
        assert captured["_api_key"] == "test-key"
        assert captured["model"] == "claude-sonnet-4-5-20250929"
        assert captured["system"] == "sys prompt"
        assert captured["messages"] == [{"role": "user", "content": "usr prompt"}]
        assert captured["max_tokens"] == 250
        assert captured["temperature"] == 0
        assert captured["timeout"] == 120

    def test_call_llm_anthropic_no_key(self) -> None:
        """Anthropic backend raises OSError when API key is missing."""
        mock_anthropic = type("module", (), {"Anthropic": None})()

        with (
            patch.dict("os.environ", {}, clear=True),
            patch.dict("sys.modules", {"anthropic": mock_anthropic}),
        ):
            with pytest.raises(OSError, match="ANTHROPIC_API_KEY not set"):
                _call_llm("system", "user", "anthropic", "claude-sonnet-4-5-20250929")

    def test_call_llm_anthropic_empty_content(self) -> None:
        """Anthropic backend returns empty string when content list is empty."""
        mock_message = type("Message", (), {"content": []})()

        def mock_create(**kw: object) -> object:
            return mock_message

        mock_messages = type("Messages", (), {"create": staticmethod(mock_create)})()

        class MockAnthropic:
            def __init__(self, **kw: object) -> None:
                self.messages = mock_messages

        mock_anthropic = type("module", (), {"Anthropic": MockAnthropic})()

        with (
            patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}),
            patch.dict("sys.modules", {"anthropic": mock_anthropic}),
        ):
            result = _call_llm("sys", "usr", "anthropic", "model")

        assert result == ""


class TestCallLlmGroqEdge:
    """Edge-case tests for _call_llm groq backend."""

    def test_call_llm_groq_ok(self) -> None:
        """Groq backend returns response text and passes correct SDK params."""
        mock_msg = type("Msg", (), {"content": " [[PASS]] Groq ok. "})()
        mock_choice = type("Choice", (), {"message": mock_msg})()
        mock_response = type("Response", (), {"choices": [mock_choice]})()

        captured: dict = {}

        def mock_create(**kw: object) -> object:
            captured.update(kw)
            return mock_response

        class MockOpenAI:
            def __init__(self, **kw: object) -> None:
                captured["_api_key"] = kw.get("api_key")
                captured["_base_url"] = kw.get("base_url")
                self.chat = type(
                    "Chat",
                    (),
                    {
                        "completions": type(
                            "Completions",
                            (),
                            {"create": staticmethod(mock_create)},
                        )()
                    },
                )()

        mock_openai_mod = type("module", (), {"OpenAI": MockOpenAI})()

        with (
            patch.dict("os.environ", {"GROQ_API_KEY": "groq-test-key"}),
            patch.dict("sys.modules", {"openai": mock_openai_mod}),
        ):
            result = _call_llm(
                "sys prompt", "usr prompt", "groq", "llama-3.3-70b-versatile"
            )

        # Verify .strip() applied
        assert result == "[[PASS]] Groq ok."
        # Verify SDK parameters (ISO 29119: all inputs to external APIs)
        assert captured["_api_key"] == "groq-test-key"
        assert captured["_base_url"] == "https://api.groq.com/openai/v1"
        assert captured["model"] == "llama-3.3-70b-versatile"
        assert captured["messages"] == [
            {"role": "system", "content": "sys prompt"},
            {"role": "user", "content": "usr prompt"},
        ]
        assert captured["max_tokens"] == 250
        assert captured["temperature"] == 0

    def test_call_llm_groq_no_key(self) -> None:
        """Groq backend raises OSError when API key is missing."""
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(OSError, match="GROQ_API_KEY not set"):
                _call_llm("sys", "usr", "groq", "model")

    def test_call_llm_groq_empty_choices(self) -> None:
        """Groq backend returns empty string when choices list is empty."""
        mock_response = type("Response", (), {"choices": []})()

        class MockOpenAI:
            def __init__(self, **kw: object) -> None:
                self.chat = type(
                    "Chat",
                    (),
                    {
                        "completions": type(
                            "Completions",
                            (),
                            {"create": staticmethod(lambda **kw: mock_response)},
                        )()
                    },
                )()

        mock_openai_mod = type("module", (), {"OpenAI": MockOpenAI})()

        with (
            patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}),
            patch.dict("sys.modules", {"openai": mock_openai_mod}),
        ):
            result = _call_llm("sys", "usr", "groq", "model")

        assert result == ""


# ── _compute_agreement ────────────────────────────────────────────────


class TestComputeAgreement:
    """Tests for inter-annotator agreement."""

    def test_perfect_agreement(self) -> None:
        labels = [True] * 100
        result = _compute_agreement(labels)
        assert result["raw_agreement"] == 1.0
        assert result["cohens_kappa"] == 1.0
        assert result["gwets_ac1"] == 1.0

    def test_partial_agreement(self) -> None:
        labels = [True] * 80 + [False] * 20
        result = _compute_agreement(labels)
        assert result["raw_agreement"] == 0.8
        # Kappa should be lower due to high prevalence
        assert result["cohens_kappa"] < result["raw_agreement"]
        # AC2 should be more robust
        assert isinstance(result["gwets_ac1"], float)

    def test_all_disagree(self) -> None:
        labels = [False] * 100
        result = _compute_agreement(labels)
        assert result["raw_agreement"] == 0.0

    def test_empty_labels(self) -> None:
        result = _compute_agreement([])
        assert result["raw_agreement"] == 0.0
        assert result["cohens_kappa"] == 0.0
        assert result["gwets_ac1"] == 0.0

    def test_single_item(self) -> None:
        result = _compute_agreement([True])
        assert result["raw_agreement"] == 1.0

    def test_kappa_prevalence_paradox(self) -> None:
        """When prevalence is high, Kappa is low but AC1 is high."""
        labels = [True] * 95 + [False] * 5
        result = _compute_agreement(labels)
        # With 95% pass, kappa suffers from prevalence paradox
        assert result["gwets_ac1"] > result["cohens_kappa"]


# ── _generate_report ──────────────────────────────────────────────────


class TestGenerateReport:
    """Tests for CSV + JSON report generation."""

    def test_csv_structure(self, sample_results: list[dict], tmp_path: Path) -> None:
        csv_path, _, _ = _generate_report(
            sample_results, "fr", "mock", "test", tmp_path
        )
        assert csv_path.exists()

        with open(csv_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 3
        assert "gs_id" in rows[0]
        assert "cr_pass" in rows[0]
        assert "af_pass" in rows[0]
        assert "ar_pass" in rows[0]

    def test_json_structure(self, sample_results: list[dict], tmp_path: Path) -> None:
        _, json_path, _ = _generate_report(
            sample_results, "fr", "mock", "test", tmp_path
        )
        assert json_path.exists()

        with open(json_path, encoding="utf-8") as f:
            report = json.load(f)

        assert report["corpus"] == "fr"
        assert report["n_items"] == 3
        assert "metrics" in report
        assert "context_relevance" in report["metrics"]
        assert "answer_faithfulness" in report["metrics"]
        assert "answer_relevance" in report["metrics"]
        assert "overall_pass_rate" in report
        assert "agreement" in report
        assert "flagged_items" in report

    def test_pass_rates_correct(
        self, sample_results: list[dict], tmp_path: Path
    ) -> None:
        _, _, report = _generate_report(sample_results, "fr", "mock", "test", tmp_path)

        # CR: 2/3 pass, AF: 2/3 pass, AR: 2/3 pass
        assert abs(report["metrics"]["context_relevance"]["pass_rate"] - 2 / 3) < 0.01
        assert report["metrics"]["context_relevance"]["n_fail"] == 1

    def test_flagged_items(self, sample_results: list[dict], tmp_path: Path) -> None:
        _, _, report = _generate_report(sample_results, "fr", "mock", "test", tmp_path)

        # gs:002 fails AF, gs:003 fails CR+AR
        assert "gs:002" in report["flagged_items"]
        assert "gs:003" in report["flagged_items"]
        assert "gs:001" not in report["flagged_items"]


# ── Three metrics independent ─────────────────────────────────────────


class TestMetricsIndependence:
    """Verify the 3 criteria can diverge independently."""

    def test_all_three_can_diverge(self) -> None:
        """Mock different outcomes per criterion by patching _call_llm."""
        # Map distinctive keywords from each prompt to responses
        keyword_map = {
            "chunk contains information relevant": "[[FAIL]] Chunk not relevant",
            "faithfully grounded": "[[PASS]] Answer grounded",
            "adequately addresses": "[[PASS]] Answer relevant",
        }

        def mock_call(
            system_prompt: str,
            user_prompt: str,
            backend: str,
            model: str,
        ) -> str:
            lower = system_prompt.lower()
            for keyword, resp in keyword_map.items():
                if keyword in lower:
                    return resp
            return "[[PASS]] default"

        with patch("scripts.evaluation.gs_validate._call_llm", side_effect=mock_call):
            cr_pass, _ = _judge_item("Q?", "C", "A", "context_relevance", "ollama")
            af_pass, _ = _judge_item("Q?", "C", "A", "answer_faithfulness", "ollama")
            ar_pass, _ = _judge_item("Q?", "C", "A", "answer_relevance", "ollama")

        assert cr_pass is False
        assert af_pass is True
        assert ar_pass is True


# ── validate_gs (integration with mock) ───────────────────────────────


class TestValidateGsMock:
    """Integration test with mock backend (no LLM calls)."""

    def test_mock_backend_runs(self, tmp_path: Path) -> None:
        """Mock backend produces valid report structure."""
        # Create minimal GS + chunks for test
        gs_data = {
            "version": "1.1",
            "schema": "GS_SCHEMA_V2",
            "questions": [
                {
                    "id": "gs:test:answerable:0001:aaa",
                    "content": {
                        "question": "What is the time limit?",
                        "expected_answer": "5 minutes per player.",
                        "is_impossible": False,
                    },
                    "provenance": {
                        "chunk_id": "test.pdf-p001-parent001-child00",
                        "docs": ["test.pdf"],
                        "pages": [1],
                    },
                    "classification": {"hard_type": "ANSWERABLE"},
                    "validation": {"status": "VALIDATED"},
                },
                {
                    "id": "gs:test:unanswerable:0002:bbb",
                    "content": {
                        "question": "What is the salary?",
                        "expected_answer": "",
                        "is_impossible": True,
                    },
                    "provenance": {"chunk_id": "test.pdf-p001-parent001-child00"},
                    "classification": {"hard_type": "OUT_OF_DATABASE"},
                    "validation": {"status": "VALIDATED"},
                },
            ],
        }
        gs_path = tmp_path / "gs_test.json"
        gs_path.write_text(json.dumps(gs_data), encoding="utf-8")

        chunks_dir = tmp_path / "corpus" / "processed"
        chunks_dir.mkdir(parents=True)
        chunks_data = {
            "chunks": [
                {
                    "id": "test.pdf-p001-parent001-child00",
                    "text": "Each player has 5 minutes per game.",
                }
            ]
        }
        chunks_path = chunks_dir / "chunks_mode_b_fr.json"
        chunks_path.write_text(json.dumps(chunks_data), encoding="utf-8")

        with patch(
            "scripts.evaluation.gs_validate.CORPUS_DIR",
            chunks_dir,
        ):
            report = validate_gs(
                corpus="fr",
                llm_backend="mock",
                max_items=0,
                output_dir=tmp_path / "output",
                gs_path=gs_path,
            )

        assert report["n_items"] == 1  # Only 1 answerable
        assert report["overall_pass_rate"] == 1.0  # Mock always passes
        assert len(report["flagged_items"]) == 0


class TestLoadGsItems:
    """Tests for _load_gs_items filtering logic."""

    def test_filters_unanswerable(self, tmp_path: Path) -> None:
        gs_data = {
            "questions": [
                {
                    "id": "gs:001",
                    "content": {
                        "question": "Q1?",
                        "expected_answer": "A1",
                        "is_impossible": False,
                    },
                    "provenance": {"chunk_id": "c1"},
                    "classification": {"hard_type": "ANSWERABLE"},
                    "validation": {"status": "VALIDATED"},
                },
                {
                    "id": "gs:002",
                    "content": {
                        "question": "Q2?",
                        "expected_answer": "",
                        "is_impossible": True,
                    },
                    "provenance": {"chunk_id": "c1"},
                    "classification": {"hard_type": "OUT_OF_DATABASE"},
                    "validation": {"status": "VALIDATED"},
                },
            ]
        }
        gs_path = tmp_path / "gs.json"
        gs_path.write_text(json.dumps(gs_data), encoding="utf-8")

        chunks_dir = tmp_path / "corpus"
        chunks_dir.mkdir()
        chunks_data = {"chunks": [{"id": "c1", "text": "chunk text"}]}
        (chunks_dir / "chunks_mode_b_fr.json").write_text(
            json.dumps(chunks_data), encoding="utf-8"
        )

        with patch("scripts.evaluation.gs_validate.CORPUS_DIR", chunks_dir):
            items = _load_gs_items("fr", gs_path)

        assert len(items) == 1
        assert items[0]["gs_id"] == "gs:001"

    def test_filters_non_validated(self, tmp_path: Path) -> None:
        gs_data = {
            "questions": [
                {
                    "id": "gs:001",
                    "content": {"question": "Q?", "expected_answer": "A"},
                    "provenance": {"chunk_id": "c1"},
                    "classification": {"hard_type": "ANSWERABLE"},
                    "validation": {"status": "PENDING"},
                },
            ]
        }
        gs_path = tmp_path / "gs.json"
        gs_path.write_text(json.dumps(gs_data), encoding="utf-8")

        chunks_dir = tmp_path / "corpus"
        chunks_dir.mkdir()
        (chunks_dir / "chunks_mode_b_fr.json").write_text(
            json.dumps({"chunks": [{"id": "c1", "text": "txt"}]}),
            encoding="utf-8",
        )

        with patch("scripts.evaluation.gs_validate.CORPUS_DIR", chunks_dir):
            items = _load_gs_items("fr", gs_path)

        assert len(items) == 0

    def test_filters_empty_answer(self, tmp_path: Path) -> None:
        """HIGH-1 fix: items with empty expected_answer are excluded."""
        gs_data = {
            "questions": [
                {
                    "id": "gs:001",
                    "content": {"question": "Q?", "expected_answer": ""},
                    "provenance": {"chunk_id": "c1"},
                    "classification": {"hard_type": "ANSWERABLE"},
                    "validation": {"status": "VALIDATED"},
                },
            ]
        }
        gs_path = tmp_path / "gs.json"
        gs_path.write_text(json.dumps(gs_data), encoding="utf-8")

        chunks_dir = tmp_path / "corpus"
        chunks_dir.mkdir()
        (chunks_dir / "chunks_mode_b_fr.json").write_text(
            json.dumps({"chunks": [{"id": "c1", "text": "txt"}]}),
            encoding="utf-8",
        )

        with patch("scripts.evaluation.gs_validate.CORPUS_DIR", chunks_dir):
            items = _load_gs_items("fr", gs_path)

        assert len(items) == 0

    def test_missing_chunks_file_raises(self, tmp_path: Path) -> None:
        """MED-4 fix: missing chunks file raises FileNotFoundError."""
        gs_data = {"questions": []}
        gs_path = tmp_path / "gs.json"
        gs_path.write_text(json.dumps(gs_data), encoding="utf-8")

        with patch(
            "scripts.evaluation.gs_validate.CORPUS_DIR",
            tmp_path / "nonexistent",
        ):
            with pytest.raises(FileNotFoundError, match="Chunks not found"):
                _load_gs_items("fr", gs_path)

    def test_missing_gs_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            _load_gs_items("fr", Path("/nonexistent/gs.json"))

    def test_skips_item_without_id(self, tmp_path: Path) -> None:
        """HIGH-2 fix: item missing 'id' field is skipped, not KeyError."""
        gs_data = {
            "questions": [
                {
                    # No "id" field
                    "content": {"question": "Q?", "expected_answer": "A"},
                    "provenance": {"chunk_id": "c1"},
                    "classification": {"hard_type": "ANSWERABLE"},
                    "validation": {"status": "VALIDATED"},
                },
            ]
        }
        gs_path = tmp_path / "gs.json"
        gs_path.write_text(json.dumps(gs_data), encoding="utf-8")

        chunks_dir = tmp_path / "corpus"
        chunks_dir.mkdir()
        (chunks_dir / "chunks_mode_b_fr.json").write_text(
            json.dumps({"chunks": [{"id": "c1", "text": "txt"}]}),
            encoding="utf-8",
        )

        with patch("scripts.evaluation.gs_validate.CORPUS_DIR", chunks_dir):
            items = _load_gs_items("fr", gs_path)

        assert len(items) == 0  # Skipped, not crashed


# ── validate_gs with FAIL path ────────────────────────────────────────


class TestValidateGsFailPath:
    """Integration test exercising FAIL verdicts end-to-end."""

    def test_fail_path_produces_flagged_items(self, tmp_path: Path) -> None:
        """MED-3 fix: verify the full pipeline with FAILs, not just PASs."""
        gs_data = {
            "questions": [
                {
                    "id": "gs:test:001",
                    "content": {
                        "question": "Q about chess?",
                        "expected_answer": "Answer about chess.",
                    },
                    "provenance": {"chunk_id": "c1"},
                    "classification": {"hard_type": "ANSWERABLE"},
                    "validation": {"status": "VALIDATED"},
                },
            ]
        }
        gs_path = tmp_path / "gs.json"
        gs_path.write_text(json.dumps(gs_data), encoding="utf-8")

        chunks_dir = tmp_path / "corpus"
        chunks_dir.mkdir()
        (chunks_dir / "chunks_mode_b_fr.json").write_text(
            json.dumps({"chunks": [{"id": "c1", "text": "chunk"}]}),
            encoding="utf-8",
        )

        # Patch _call_llm to return FAIL for faithfulness
        call_count = 0

        def mock_fail_af(
            system_prompt: str,
            user_prompt: str,
            backend: str,
            model: str,
        ) -> str:
            nonlocal call_count
            call_count += 1
            if "faithfully grounded" in system_prompt.lower():
                return "[[FAIL]] Answer adds info not in chunk."
            return "[[PASS]] Ok."

        with (
            patch("scripts.evaluation.gs_validate.CORPUS_DIR", chunks_dir),
            patch(
                "scripts.evaluation.gs_validate._call_llm",
                side_effect=mock_fail_af,
            ),
        ):
            report = validate_gs(
                corpus="fr",
                llm_backend="ollama",
                model="test",
                output_dir=tmp_path / "output",
                gs_path=gs_path,
            )

        assert call_count == 3  # 3 criteria per item
        assert report["n_items"] == 1
        assert report["overall_pass_rate"] == 0.0  # AF failed
        assert report["metrics"]["answer_faithfulness"]["n_fail"] == 1
        assert report["metrics"]["context_relevance"]["n_fail"] == 0
        assert "gs:test:001" in report["flagged_items"]


# ── CLI entrypoint ────────────────────────────────────────────────────


class TestMainCli:
    """MED-5 fix: test the CLI entrypoint."""

    def test_main_mock_backend(self, tmp_path: Path) -> None:
        """main() with mock backend completes without error."""
        from scripts.evaluation.gs_validate import main

        gs_data = {
            "questions": [
                {
                    "id": "gs:cli:001",
                    "content": {"question": "Q?", "expected_answer": "A."},
                    "provenance": {"chunk_id": "c1"},
                    "classification": {"hard_type": "ANSWERABLE"},
                    "validation": {"status": "VALIDATED"},
                },
            ]
        }
        gs_path = tmp_path / "gs_scratch_v1.json"
        gs_path.write_text(json.dumps(gs_data), encoding="utf-8")

        chunks_dir = tmp_path / "corpus"
        chunks_dir.mkdir()
        (chunks_dir / "chunks_mode_b_fr.json").write_text(
            json.dumps({"chunks": [{"id": "c1", "text": "chunk text"}]}),
            encoding="utf-8",
        )

        with (
            patch("scripts.evaluation.gs_validate.CORPUS_DIR", chunks_dir),
            patch("scripts.evaluation.gs_validate.TESTS_DATA_DIR", tmp_path),
            patch(
                "sys.argv",
                ["gs_validate", "--backend", "mock", "--corpus", "fr"],
            ),
        ):
            # main() prints to stdout but should not crash
            main()
