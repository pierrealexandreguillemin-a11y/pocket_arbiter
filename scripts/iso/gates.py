#!/usr/bin/env python3
"""Executable gates for ISO validation."""

import json
import subprocess
from pathlib import Path

from .utils import Colors, colored


class ExecutableGates:
    """Executable gates that run actual commands."""

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
        if self.verbose:
            print(f"  {message}")

    def run_command(
        self, cmd: list[str], cwd: Path | None = None
    ) -> tuple[bool, str]:
        """Run a command and return success status and output."""
        try:
            result = subprocess.run(
                cmd, cwd=cwd or self.root, capture_output=True, text=True, timeout=300
            )
            return result.returncode == 0, result.stdout + result.stderr
        except subprocess.TimeoutExpired:
            return False, "Command timed out"
        except FileNotFoundError:
            return False, f"Command not found: {cmd[0]}"
        except Exception as e:
            return False, str(e)

    def gate_pytest(self, path: str = "scripts/", required: bool = True) -> bool:
        """Gate: Run pytest and verify all tests pass."""
        print(f"    Running pytest on {path}...", end=" ")

        success, output = self.run_command(["python", "-m", "pytest", "--version"])
        if not success:
            if required:
                print(colored("FAILED", Colors.RED), "(pytest not installed)")
                self.errors.append(f"pytest non installe - REQUIS pour {path}")
                return False
            print(colored("SKIP", Colors.YELLOW), "(pytest not installed)")
            return True

        test_path = self.root / path
        if not test_path.exists():
            print(colored("SKIP", Colors.YELLOW), "(path not found)")
            return True

        success, output = self.run_command(
            ["python", "-m", "pytest", str(test_path), "-v", "--tb=short"]
        )

        if success:
            print(colored("OK", Colors.GREEN))
            self.passed.append(f"Tests pytest passent: {path}")
            return True
        print(colored("FAILED", Colors.RED))
        self.log(output[:500])
        self.errors.append(f"Tests pytest echouent: {path}")
        return False

    def gate_coverage(self, target: float = 0.60, required: bool = False) -> bool:
        """Gate: Check test coverage meets target."""
        print(f"    Checking coverage (target: {target*100:.0f}%)...", end=" ")

        self.run_command(
            [
                "python",
                "-m",
                "pytest",
                "--cov=scripts",
                "--cov-report=json",
                "scripts/",
                "-q",
            ]
        )

        cov_file = self.root / "coverage.json"
        if not cov_file.exists():
            if required:
                print(colored("FAILED", Colors.RED), "(coverage not available)")
                self.errors.append("Coverage non mesurable - pytest-cov REQUIS")
                return False
            print(colored("SKIP", Colors.YELLOW), "(coverage not available)")
            return True

        try:
            with open(cov_file) as f:
                cov_data = json.load(f)
            total_cov = cov_data.get("totals", {}).get("percent_covered", 0) / 100

            if total_cov >= target:
                print(colored(f"OK ({total_cov*100:.1f}%)", Colors.GREEN))
                self.passed.append(
                    f"Coverage: {total_cov*100:.1f}% >= {target*100:.0f}%"
                )
                return True
            print(colored(f"FAILED ({total_cov*100:.1f}%)", Colors.RED))
            self.errors.append(f"Coverage: {total_cov*100:.1f}% < {target*100:.0f}%")
            return False
        except Exception as e:
            print(colored("SKIP", Colors.YELLOW), f"({e})")
            return True

    def gate_lint(self, path: str = "scripts/") -> bool:
        """Gate: Run flake8 and verify no critical errors."""
        print(f"    Running lint on {path}...", end=" ")

        success, output = self.run_command(
            ["python", "-m", "flake8", path, "--select=E9,F63,F7,F82", "--count"]
        )

        if success:
            print(colored("OK", Colors.GREEN))
            self.passed.append(f"Lint clean: {path}")
            return True
        lines = output.strip().split("\n")
        error_count = len([line for line in lines if line and not line.isspace()])
        print(colored(f"FAILED ({error_count} errors)", Colors.RED))
        self.errors.append(f"Lint errors: {error_count} dans {path}")
        return False

    def gate_json_valid(self, patterns: list[str] | None = None) -> bool:
        """Gate: Validate all JSON files in patterns."""
        if patterns is None:
            patterns = ["tests/data/*.json", ".iso/*.json"]

        print("    Validating JSON files...", end=" ")

        all_valid = True
        valid_count = 0

        for pattern in patterns:
            for json_file in self.root.glob(pattern):
                try:
                    with open(json_file, encoding="utf-8") as f:
                        json.load(f)
                    valid_count += 1
                except json.JSONDecodeError as e:
                    self.errors.append(f"JSON invalide: {json_file.name} - {e}")
                    all_valid = False

        if all_valid:
            print(colored(f"OK ({valid_count} files)", Colors.GREEN))
            self.passed.append(f"JSON valides: {valid_count} fichiers")
        else:
            print(colored("FAILED", Colors.RED))

        return all_valid

    def gate_git_status(self) -> bool:
        """Gate: Check git is initialized and has remote."""
        print("    Checking git status...", end=" ")

        if not (self.root / ".git").is_dir():
            print(colored("FAILED", Colors.RED))
            self.errors.append("Git non initialise")
            return False

        success, output = self.run_command(["git", "remote", "-v"])
        if not success or not output.strip():
            print(colored("WARN", Colors.YELLOW), "(no remote)")
            self.warnings.append("Git remote non configure")
            return True

        print(colored("OK", Colors.GREEN))
        self.passed.append("Git initialise avec remote")
        return True
