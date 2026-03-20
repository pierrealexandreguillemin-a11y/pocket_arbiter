"""Tests for training config sanity."""

from scripts.training.config import (
    ICT_CONFIG,
    LORA_CONFIG,
    SEED,
    SIMCSE_CONFIG,
)


def test_seed_is_42():
    assert SEED == 42


def test_simcse_config_keys():
    required = {
        "batch_size",
        "lr",
        "epochs",
        "temperature",
        "mini_batch_size",
        "weight_decay",
        "max_grad_norm",
        "warmup_ratio",
    }
    assert required.issubset(set(SIMCSE_CONFIG.keys()))


def test_ict_config_keys():
    required = {
        "batch_size",
        "lr",
        "epochs",
        "masking_rate",
        "mini_batch_size",
        "weight_decay",
        "max_grad_norm",
        "warmup_ratio",
    }
    assert required.issubset(set(ICT_CONFIG.keys()))


def test_lora_config_keys():
    required = {"rank", "alpha", "dropout", "target_modules"}
    assert required.issubset(set(LORA_CONFIG.keys()))


def test_ict_lr_less_than_simcse():
    assert ICT_CONFIG["lr"] < SIMCSE_CONFIG["lr"]


def test_lora_alpha_equals_rank():
    assert LORA_CONFIG["alpha"] == LORA_CONFIG["rank"]


def test_masking_rate_90_percent():
    assert ICT_CONFIG["masking_rate"] == 0.9
