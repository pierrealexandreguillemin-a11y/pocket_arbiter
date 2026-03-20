"""Kaggle automation: upload dataset, push kernel, wait, download outputs.

Usage: python -m scripts.training.kaggle_run

Workflow:
1. Copy training data to kaggle/dataset/
2. Create or update Kaggle dataset
3. Push kernel script with GPU T4
4. Poll status until complete
5. Download checkpoints from kernel output
"""

from __future__ import annotations

import logging
import shutil
import time
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATASET_DIR = PROJECT_ROOT / "kaggle" / "dataset"
KERNEL_DIR = PROJECT_ROOT / "kaggle" / "kernel"
TRAINING_DATA_DIR = PROJECT_ROOT / "data" / "training"
MODELS_DIR = PROJECT_ROOT / "models"

# Kaggle identifiers
DATASET_SLUG = "pguillemin/pocket-arbiter-training-data"
KERNEL_SLUG = "pguillemin/pocket-arbiter-simcse-ict"


def step1_prepare_dataset() -> None:
    """Copy training data files to kaggle/dataset/ for upload."""
    for fname in ["simcse_pairs.jsonl", "ict_pairs.jsonl"]:
        src = TRAINING_DATA_DIR / fname
        dst = DATASET_DIR / fname
        if not src.exists():
            msg = f"Missing training data: {src}. Run Task 3 first."
            raise FileNotFoundError(msg)
        shutil.copy2(src, dst)
        logger.info("Copied %s -> %s", src, dst)


def step2_upload_dataset(api: KaggleApi) -> None:
    """Create or update Kaggle dataset."""
    logger.info("Uploading dataset to %s", DATASET_SLUG)
    try:
        api.dataset_create_new(
            folder=str(DATASET_DIR),
            dir_mode="zip",
            quiet=False,
        )
        logger.info("Dataset created: %s", DATASET_SLUG)
    except Exception:
        logger.info("Dataset exists, creating new version...")
        api.dataset_create_version(
            folder=str(DATASET_DIR),
            version_notes="Update training data",
            dir_mode="zip",
            quiet=False,
        )
        logger.info("Dataset updated: %s", DATASET_SLUG)


def step3_push_kernel(api: KaggleApi) -> None:
    """Push kernel script with T4 GPU."""
    logger.info("Pushing kernel %s with T4 GPU", KERNEL_SLUG)
    api.kernels_push(str(KERNEL_DIR))
    logger.info("Kernel pushed successfully")


def step4_wait_for_completion(api: KaggleApi, poll_interval: int = 60) -> str:
    """Poll kernel status until complete. Returns final status."""
    logger.info("Waiting for kernel to complete (polling every %ds)...", poll_interval)
    while True:
        status = api.kernels_status(KERNEL_SLUG)
        state = (
            status.get("status", "unknown") if isinstance(status, dict) else str(status)
        )
        logger.info("Status: %s", state)
        if "complete" in state.lower() or "error" in state.lower():
            return state
        time.sleep(poll_interval)


def step5_download_outputs(api: KaggleApi) -> None:
    """Download kernel outputs (model checkpoints)."""
    output_dir = MODELS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading kernel outputs to %s", output_dir)
    api.kernels_output(KERNEL_SLUG, path=str(output_dir), force=True)
    logger.info("Outputs downloaded")

    # List what we got
    for p in sorted(output_dir.rglob("*")):
        if p.is_file():
            size_mb = p.stat().st_size / 1024 / 1024
            logger.info("  %s (%.1f MB)", p.relative_to(PROJECT_ROOT), size_mb)


def main() -> None:
    """Run full Kaggle workflow."""
    api = KaggleApi()
    api.authenticate()
    logger.info("Authenticated as: %s", api.config_values.get("username", "unknown"))

    step1_prepare_dataset()
    step2_upload_dataset(api)
    step3_push_kernel(api)
    status = step4_wait_for_completion(api)

    if "error" in status.lower():
        logger.error("Kernel FAILED: %s", status)
        logger.error("Check logs: https://www.kaggle.com/code/%s", KERNEL_SLUG)
        return

    step5_download_outputs(api)
    logger.info("=== Kaggle workflow complete ===")
    logger.info(
        "Next: run Task 5b (recall.py model_id fix) then Task 6 (recall measurement)"
    )


if __name__ == "__main__":
    main()
