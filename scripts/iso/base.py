#!/usr/bin/env python3
"""Base class for ISO checks."""

from pathlib import Path


class BaseChecker:
    """Base class providing common check functionality."""

    def __init__(
        self,
        root: Path,
        errors: list[str],
        warnings: list[str],
        passed: list[str],
        verbose: bool = False,
    ):
        self.root = root
        self.errors = errors
        self.warnings = warnings
        self.passed = passed
        self.verbose = verbose

    def log(self, message: str) -> None:
        """Log verbose messages."""
        if self.verbose:
            print(f"  {message}")

    def check_file_exists(self, path: str, description: str) -> bool:
        """Check if a required file exists."""
        full_path = self.root / path
        if full_path.exists():
            self.passed.append(f"{description}: {path}")
            return True
        else:
            self.errors.append(f"{description} manquant: {path}")
            return False

    def check_dir_exists(self, path: str, description: str) -> bool:
        """Check if a required directory exists."""
        full_path = self.root / path
        if full_path.is_dir():
            self.passed.append(f"{description}: {path}/")
            return True
        else:
            self.errors.append(f"{description} manquant: {path}/")
            return False
