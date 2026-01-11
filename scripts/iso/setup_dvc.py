#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DVC Setup Script - Pocket Arbiter
=================================
Initializes DVC and tracks corpus files.

Usage:
    python scripts/iso/setup_dvc.py [--remote URL]

ISO 12207 Compliance: Large file versioning for traceability.
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Optional, Tuple

# Fix Windows encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore


def run_command(cmd: list, cwd: Optional[Path] = None) -> Tuple[bool, str]:
    """Run a command and return success status and output."""
    try:
        result = subprocess.run(
            cmd, cwd=cwd, capture_output=True, text=True, timeout=300
        )
        return result.returncode == 0, result.stdout + result.stderr
    except Exception as e:
        return False, str(e)


def check_dvc_installed() -> bool:
    """Check if DVC is installed."""
    success, _ = run_command(["dvc", "--version"])
    if not success:
        success, _ = run_command([sys.executable, "-m", "dvc", "--version"])
    return success


def _step_check_dvc() -> bool:
    """Step 1: Check DVC installation."""
    print("\n[1/5] Checking DVC installation...", end=" ")
    if not check_dvc_installed():
        print("FAILED")
        print("\nDVC not found. Install with:")
        print("  pip install dvc")
        return False
    print("OK")
    return True


def _step_init_dvc(project_root: Path) -> bool:
    """Step 2: Initialize DVC if needed."""
    print("[2/5] Initializing DVC...", end=" ")
    dvc_config = project_root / ".dvc" / "config"
    if dvc_config.exists():
        print("OK")
        return True
    success, output = run_command(["dvc", "init"], cwd=project_root)
    if not success:
        print("FAILED")
        print(output)
        return False
    print("OK")
    return True


def _step_configure_remote(project_root: Path, remote_url: Optional[str]) -> None:
    """Step 3: Configure remote if provided."""
    if not remote_url:
        return
    print(f"[3/5] Configuring remote: {remote_url}...", end=" ")
    success, _ = run_command(
        ["dvc", "remote", "add", "-d", "origin", remote_url], cwd=project_root
    )
    print("OK" if success else "WARN (may already exist)")


def _step_track_corpus(project_root: Path) -> int:
    """Step 4: Track corpus PDFs. Returns count of tracked files."""
    print("[4/5] Tracking corpus files...", end=" ")
    corpus_dirs = [project_root / "corpus" / "fr", project_root / "corpus" / "intl"]
    tracked = 0
    for corpus_dir in corpus_dirs:
        if not corpus_dir.is_dir():
            continue
        pdfs = list(corpus_dir.glob("*.pdf"))
        if pdfs:
            success, _ = run_command(["dvc", "add", str(corpus_dir)], cwd=project_root)
            if success:
                tracked += len(pdfs)
    msg = f"OK ({tracked} PDFs tracked)" if tracked else "WARN (no PDFs found)"
    print(msg)
    return tracked


def _step_track_models(project_root: Path) -> None:
    """Step 5: Track models directory if it has large files."""
    print("[5/5] Tracking models directory...", end=" ")
    models_dir = project_root / "models"
    if not models_dir.is_dir():
        print("SKIP (directory not created)")
        return
    patterns = ["*.gguf", "*.onnx", "*.tflite"]
    large_files = [f for p in patterns for f in models_dir.glob(p)]
    if not large_files:
        print("SKIP (no model files yet)")
        return
    run_command(["dvc", "add", str(models_dir)], cwd=project_root)
    print(f"OK ({len(large_files)} model files)")


def _print_summary(has_remote: bool) -> None:
    """Print final summary and next steps."""
    print("\n" + "=" * 60)
    print("  DVC Setup Complete")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. git add .dvc corpus/*.dvc models/*.dvc")
    print("  2. git commit -m '[chore] Add DVC tracking'")
    if not has_remote:
        print("  3. Configure remote: dvc remote add -d origin <URL>")
    print("  4. Push data: dvc push")


def main() -> int:
    """Main entry point for DVC setup."""
    parser = argparse.ArgumentParser(
        description="Initialize DVC for Pocket Arbiter corpus files"
    )
    parser.add_argument(
        "--remote", help="Remote storage URL (e.g., gdrive://..., s3://...)"
    )
    parser.add_argument(
        "--init-only", action="store_true", help="Only initialize DVC, don't track"
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent.parent

    print("=" * 60)
    print("  DVC Setup - Pocket Arbiter")
    print("=" * 60)

    if not _step_check_dvc():
        return 1
    if not _step_init_dvc(project_root):
        return 1
    _step_configure_remote(project_root, args.remote)

    if args.init_only:
        print("\n[DONE] DVC initialized. Run without --init-only to track files.")
        return 0

    _step_track_corpus(project_root)
    _step_track_models(project_root)
    _print_summary(bool(args.remote))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
