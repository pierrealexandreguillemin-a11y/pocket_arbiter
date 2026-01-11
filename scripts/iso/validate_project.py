#!/usr/bin/env python3
"""
ISO Project Validator - Pocket Arbiter
======================================
Entry point for ISO compliance validation.

Usage:
    python scripts/iso/validate_project.py [--phase N] [--verbose] [--gates]
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Tuple

from .utils import Icons, Colors, colored
from .gates import ExecutableGates
from .phases import PhaseValidator
from .checks import ISO12207Checks, ISO25010Checks, ISO29119Checks, ISO42001Checks


class ISOValidator:
    """Main ISO validator orchestrating all checks."""

    def __init__(self, project_root: Path, verbose: bool = False):
        self.root = project_root
        self.verbose = verbose
        self.errors = []
        self.warnings = []
        self.passed = []

    def _make_checker(self, cls):
        """Create a checker instance with shared state."""
        return cls(self.root, self.errors, self.warnings, self.passed, self.verbose)

    def validate_all(self, phase: int = None, run_gates: bool = False) -> Tuple[bool, Dict]:
        """Run all validations."""
        print(colored("=" * 60, Colors.BLUE))
        print(colored("  ISO Validation - Pocket Arbiter", Colors.BLUE))
        print(colored("=" * 60, Colors.BLUE))

        # Run ISO checks
        iso12207 = self._make_checker(ISO12207Checks)
        iso12207.validate_structure()
        iso12207.validate_docs()

        iso42001 = self._make_checker(ISO42001Checks)
        iso42001.validate_policy()
        iso42001.validate_antihallu()

        iso25010 = self._make_checker(ISO25010Checks)
        iso25010.validate_quality()

        iso29119 = self._make_checker(ISO29119Checks)
        iso29119.validate_testing()

        # Run executable gates if requested
        if run_gates:
            print(colored(f"\n{Icons.SHIELD} Executable Gates", Colors.BLUE))
            gates = ExecutableGates(
                self.root, self.errors, self.warnings, self.passed, self.verbose
            )
            gates.gate_json_valid()
            gates.gate_lint("scripts/")
            gates.gate_git_status()

            if list((self.root / "scripts").rglob("test_*.py")):
                gates.gate_pytest("scripts/")

        # Run phase-specific check if requested
        if phase is not None:
            phase_validator = PhaseValidator(
                self.root, self.errors, self.warnings, self.passed, self.verbose
            )
            phase_validator.validate_phase(phase)

        # Print summary
        self._print_summary()

        success = len(self.errors) == 0
        return success, {
            "passed": len(self.passed),
            "warnings": len(self.warnings),
            "errors": len(self.errors),
            "details": {
                "passed": self.passed,
                "warnings": self.warnings,
                "errors": self.errors,
            }
        }

    def _print_summary(self):
        """Print validation summary."""
        print(colored("\n" + "=" * 60, Colors.BLUE))
        print(colored("  RÉSUMÉ", Colors.BLUE))
        print(colored("=" * 60, Colors.BLUE))

        print(colored(f"\n{Icons.CHECK} Passe: {len(self.passed)}", Colors.GREEN))
        for item in self.passed:
            print(f"   • {item}")

        if self.warnings:
            print(colored(f"\n{Icons.WARN} Warnings: {len(self.warnings)}", Colors.YELLOW))
            for item in self.warnings:
                print(f"   • {item}")

        if self.errors:
            print(colored(f"\n{Icons.CROSS} Erreurs: {len(self.errors)}", Colors.RED))
            for item in self.errors:
                print(f"   • {item}")

        print(colored("\n" + "=" * 60, Colors.BLUE))
        if len(self.errors) == 0:
            print(colored(f"  {Icons.CHECK} VALIDATION ISO REUSSIE", Colors.GREEN))
        else:
            print(colored(f"  {Icons.CROSS} VALIDATION ISO ECHOUEE", Colors.RED))
        print(colored("=" * 60, Colors.BLUE))


def main():
    parser = argparse.ArgumentParser(description="Valide la conformité ISO")
    parser.add_argument("--phase", "-p", type=int, choices=[0, 1, 2, 3, 4, 5])
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--gates", "-g", action="store_true")

    args = parser.parse_args()

    script_path = Path(__file__).resolve()
    project_root = script_path.parent.parent.parent

    validator = ISOValidator(project_root, verbose=args.verbose)
    success, results = validator.validate_all(phase=args.phase, run_gates=args.gates)

    if args.json:
        print(json.dumps(results, indent=2))

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
