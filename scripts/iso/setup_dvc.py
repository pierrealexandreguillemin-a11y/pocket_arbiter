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

# Fix Windows encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')


def run_command(cmd: list, cwd: Path = None) -> tuple:
    """Run a command and return success status and output."""
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=300
        )
        return result.returncode == 0, result.stdout + result.stderr
    except Exception as e:
        return False, str(e)


def check_dvc_installed() -> bool:
    """Check if DVC is installed."""
    success, output = run_command(["dvc", "--version"])
    if not success:
        # Try via Python module
        success, output = run_command([sys.executable, "-m", "dvc", "--version"])
    return success


def main():
    parser = argparse.ArgumentParser(
        description="Initialize DVC for Pocket Arbiter corpus files"
    )
    parser.add_argument(
        "--remote",
        help="Remote storage URL (e.g., gdrive://..., s3://...)"
    )
    parser.add_argument(
        "--init-only",
        action="store_true",
        help="Only initialize DVC, don't track files"
    )
    args = parser.parse_args()

    # Find project root
    script_path = Path(__file__).resolve()
    project_root = script_path.parent.parent.parent

    print("=" * 60)
    print("  DVC Setup - Pocket Arbiter")
    print("=" * 60)

    # Step 1: Check DVC is installed
    print("\n[1/5] Checking DVC installation...", end=" ")
    if not check_dvc_installed():
        print("FAILED")
        print("\nDVC not found. Install with:")
        print("  pip install dvc")
        print("  # or")
        print("  pip install dvc[gdrive]  # for Google Drive remote")
        return 1
    print("OK")

    # Step 2: Initialize DVC if needed
    dvc_dir = project_root / ".dvc"
    print("[2/5] Initializing DVC...", end=" ")
    if not (dvc_dir / "config").exists():
        success, output = run_command(["dvc", "init"], cwd=project_root)
        if not success:
            print("FAILED")
            print(output)
            return 1
    print("OK")

    # Step 3: Configure remote if provided
    if args.remote:
        print(f"[3/5] Configuring remote: {args.remote}...", end=" ")
        success, output = run_command(
            ["dvc", "remote", "add", "-d", "origin", args.remote],
            cwd=project_root
        )
        if success:
            print("OK")
        else:
            print("WARN (may already exist)")

    if args.init_only:
        print("\n[DONE] DVC initialized. Run without --init-only to track files.")
        return 0

    # Step 4: Track corpus PDFs
    print("[4/5] Tracking corpus files...", end=" ")
    corpus_dirs = [
        project_root / "corpus" / "fr",
        project_root / "corpus" / "intl",
    ]

    tracked_count = 0
    for corpus_dir in corpus_dirs:
        if corpus_dir.is_dir():
            pdf_files = list(corpus_dir.glob("*.pdf"))
            if pdf_files:
                # Track the whole directory
                success, output = run_command(
                    ["dvc", "add", str(corpus_dir)],
                    cwd=project_root
                )
                if success:
                    tracked_count += len(pdf_files)

    if tracked_count > 0:
        print(f"OK ({tracked_count} PDFs tracked)")
    else:
        print("WARN (no PDFs found)")

    # Step 5: Track models directory if it has large files
    print("[5/5] Tracking models directory...", end=" ")
    models_dir = project_root / "models"
    if models_dir.is_dir():
        large_files = list(models_dir.glob("*.gguf")) + \
                      list(models_dir.glob("*.onnx")) + \
                      list(models_dir.glob("*.tflite"))
        if large_files:
            success, output = run_command(
                ["dvc", "add", str(models_dir)],
                cwd=project_root
            )
            print(f"OK ({len(large_files)} model files)")
        else:
            print("SKIP (no model files yet)")
    else:
        print("SKIP (directory not created)")

    # Summary
    print("\n" + "=" * 60)
    print("  DVC Setup Complete")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. git add .dvc corpus/*.dvc models/*.dvc")
    print("  2. git commit -m '[chore] Add DVC tracking'")
    if not args.remote:
        print("  3. Configure remote: dvc remote add -d origin <URL>")
    print("  4. Push data: dvc push")

    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
