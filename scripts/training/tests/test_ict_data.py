"""Tests for ICT data extraction."""

from scripts.training.config import SEED
from scripts.training.ict_data import (
    compute_data_stats,
    extract_random_sentence,
    generate_ict_pairs,
    generate_simcse_pairs,
    mask_sentence_from_chunk,
    save_pairs,
    validate_pairs,
)


def test_extract_random_sentence_returns_long():
    text = "Phrase courte. Deuxieme phrase assez longue pour passer le filtre minimum."
    s = extract_random_sentence(text, seed=SEED)
    assert s is not None
    assert len(s) > 20


def test_extract_random_sentence_none_if_all_short():
    assert extract_random_sentence("A. B. C.", seed=SEED) is None


def test_extract_random_sentence_reproducible():
    text = (
        "Premiere phrase longue. Deuxieme phrase aussi longue. Troisieme pour varier."
    )
    s1 = extract_random_sentence(text, seed=42)
    s2 = extract_random_sentence(text, seed=42)
    assert s1 == s2


def test_mask_sentence_removes_90_percent():
    chunk = "Phrase a masquer. Le reste du chunk continue ici."
    masked = sum(
        1
        for i in range(100)
        if "Phrase a masquer"
        not in mask_sentence_from_chunk(
            chunk, "Phrase a masquer.", mask_rate=0.9, seed=i
        )
    )
    assert 80 <= masked <= 100


def test_mask_sentence_preserves_at_zero():
    chunk = "Phrase gardee. Le reste."
    result = mask_sentence_from_chunk(chunk, "Phrase gardee.", mask_rate=0.0, seed=0)
    assert "Phrase gardee." in result


def test_generate_simcse_pairs():
    texts = ["Chunk texte numero un." for _ in range(10)]
    pairs = generate_simcse_pairs(texts)
    assert len(pairs) == 10
    assert pairs[0][0] == pairs[0][1]


def test_generate_ict_pairs_count():
    texts = [
        "Premiere phrase longue du chunk reglementaire. Deuxieme phrase du chunk."
        for _ in range(10)
    ]
    pairs = generate_ict_pairs(texts, seed=SEED)
    assert 0 < len(pairs) <= 10
    for query, chunk in pairs:
        assert len(query) > 20


def test_generate_ict_pairs_reproducible():
    texts = ["Phrase un longue. Phrase deux longue." for _ in range(10)]
    assert generate_ict_pairs(texts, seed=42) == generate_ict_pairs(texts, seed=42)


def test_validate_pairs_pass():
    pairs = [("Query longue suffisante ok", "Document texte complet")] * 10
    errors = validate_pairs(pairs, min_query_len=20)
    assert len(errors) == 0


def test_validate_pairs_fail_short():
    pairs = [("Short", "Document texte complet")]
    errors = validate_pairs(pairs, min_query_len=20)
    assert len(errors) == 1


def test_compute_data_stats():
    pairs = [("Query de test assez longue", "Document plus long que la query")] * 5
    stats = compute_data_stats(pairs)
    assert stats["count"] == 5
    assert "query_len_median" in stats
    assert "doc_len_median" in stats
    assert stats["query_len_min"] > 0


def test_save_pairs(tmp_path):
    import json

    pairs = [("query", "doc")]
    path = tmp_path / "test.jsonl"
    save_pairs(pairs, path)
    assert path.exists()
    with open(path) as f:
        row = json.loads(f.readline())
    assert row["query"] == "query"
    assert row["document"] == "doc"
