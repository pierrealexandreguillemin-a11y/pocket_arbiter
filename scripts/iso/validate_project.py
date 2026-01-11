#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ISO Project Validator - Pocket Arbiter
======================================
Valide la conformité ISO du projet à chaque étape.

Normes vérifiées:
- ISO/IEC 25010: Qualité logicielle
- ISO/IEC 42001: Gouvernance IA
- ISO/IEC 12207: Cycle de vie logiciel
- ISO/IEC 29119: Tests

Usage:
    python scripts/iso/validate_project.py [--phase N] [--verbose]
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# Colors for terminal output (disable on Windows if not supported)
class Colors:
    if sys.platform == 'win32' and not os.environ.get('WT_SESSION'):
        # Windows without Windows Terminal - no colors
        RED = ''
        GREEN = ''
        YELLOW = ''
        BLUE = ''
        NC = ''
    else:
        RED = '\033[0;31m'
        GREEN = '\033[0;32m'
        YELLOW = '\033[1;33m'
        BLUE = '\033[0;34m'
        NC = '\033[0m'

# Icons (ASCII fallback for Windows)
class Icons:
    if sys.platform == 'win32' and not os.environ.get('WT_SESSION'):
        FOLDER = '[DIR]'
        DOC = '[DOC]'
        ROBOT = '[AI]'
        SHIELD = '[SEC]'
        SPARKLE = '[QA]'
        TEST = '[TEST]'
        PIN = '[PHASE]'
        CHECK = '[OK]'
        CROSS = '[FAIL]'
        WARN = '[WARN]'
    else:
        FOLDER = '📁'
        DOC = '📝'
        ROBOT = '🤖'
        SHIELD = '🛡️'
        SPARKLE = '✨'
        TEST = '🧪'
        PIN = '📌'
        CHECK = '✅'
        CROSS = '❌'
        WARN = '⚠️'


def colored(text: str, color: str) -> str:
    """Return colored text for terminal."""
    return f"{color}{text}{Colors.NC}"


class ISOValidator:
    """Validates ISO compliance for Pocket Arbiter project."""

    def __init__(self, project_root: Path, verbose: bool = False):
        self.root = project_root
        self.verbose = verbose
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.passed: List[str] = []

    def log(self, message: str):
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

    # ==========================================================================
    # ISO 12207 - Software Lifecycle Checks
    # ==========================================================================

    def validate_iso12207_structure(self) -> bool:
        """Validate project structure per ISO 12207."""
        print(colored(f"\n{Icons.FOLDER} ISO 12207 - Structure projet", Colors.BLUE))

        required_dirs = [
            ("android", "Projet Android"),
            ("scripts", "Scripts pipeline"),
            ("corpus", "Données sources"),
            ("docs", "Documentation"),
            ("prompts", "Prompts LLM"),
            ("tests", "Tests"),
        ]

        required_files = [
            ("README.md", "README projet"),
            ("CLAUDE_CODE_INSTRUCTIONS.md", "Instructions Claude Code"),
            (".gitignore", "Git ignore"),
        ]

        all_ok = True
        for path, desc in required_dirs:
            if not self.check_dir_exists(path, desc):
                all_ok = False

        for path, desc in required_files:
            if not self.check_file_exists(path, desc):
                all_ok = False

        return all_ok

    def validate_iso12207_docs(self) -> bool:
        """Validate documentation per ISO 12207."""
        print(colored(f"\n{Icons.DOC} ISO 12207 - Documentation", Colors.BLUE))

        required_docs = [
            ("docs/VISION.md", "Vision projet"),
            ("docs/QUALITY_REQUIREMENTS.md", "Exigences qualité"),
            ("docs/TEST_PLAN.md", "Plan de tests"),
            ("corpus/INVENTORY.md", "Inventaire corpus"),
        ]

        all_ok = True
        for path, desc in required_docs:
            if not self.check_file_exists(path, desc):
                all_ok = False

        return all_ok

    # ==========================================================================
    # ISO 42001 - AI Governance Checks
    # ==========================================================================

    def validate_iso42001_policy(self) -> bool:
        """Validate AI policy per ISO 42001."""
        print(colored(f"\n{Icons.ROBOT} ISO 42001 - Gouvernance IA", Colors.BLUE))

        all_ok = True

        # Check AI Policy exists
        if not self.check_file_exists("docs/AI_POLICY.md", "Politique IA"):
            all_ok = False

        # Check prompts directory has content
        prompts_dir = self.root / "prompts"
        if prompts_dir.is_dir():
            prompt_files = list(prompts_dir.glob("*.txt")) + list(prompts_dir.glob("*.md"))
            if len(prompt_files) < 2:  # At least README + one prompt
                self.warnings.append("Peu de prompts versionnés dans prompts/")
            else:
                self.passed.append(f"Prompts versionnés: {len(prompt_files)} fichiers")

        # Check for CHANGELOG in prompts
        if not self.check_file_exists("prompts/CHANGELOG.md", "Changelog prompts"):
            all_ok = False

        return all_ok

    def validate_iso42001_antihallu(self) -> bool:
        """Check for anti-hallucination patterns in AI code."""
        print(colored(f"\n{Icons.SHIELD} ISO 42001 - Anti-hallucination", Colors.BLUE))

        # This is a basic check - could be expanded
        dangerous_patterns = [
            "generate_without_context",
            "response_without_source",
            "skip_citation",
        ]

        ai_files = list(self.root.glob("**/*llm*.py")) + \
                   list(self.root.glob("**/*ai*.py")) + \
                   list(self.root.glob("**/*generat*.py"))

        issues_found = False
        for f in ai_files:
            if f.is_file():
                content = f.read_text(encoding='utf-8', errors='ignore')
                for pattern in dangerous_patterns:
                    if pattern in content.lower():
                        self.errors.append(f"Pattern dangereux '{pattern}' dans {f}")
                        issues_found = True

        if not issues_found:
            self.passed.append("Pas de patterns anti-hallucination dangereux détectés")

        return not issues_found

    # ==========================================================================
    # ISO 25010 - Software Quality Checks
    # ==========================================================================

    def validate_iso25010_quality(self) -> bool:
        """Validate quality requirements per ISO 25010."""
        print(colored(f"\n{Icons.SPARKLE} ISO 25010 - Qualite logicielle", Colors.BLUE))

        all_ok = True

        # Check quality doc exists
        if not self.check_file_exists("docs/QUALITY_REQUIREMENTS.md", "Exigences qualité"):
            all_ok = False

        # Check for test data
        test_data_dir = self.root / "tests" / "data"
        if test_data_dir.is_dir():
            test_files = list(test_data_dir.glob("*.json"))
            if len(test_files) >= 2:
                self.passed.append(f"Données de test: {len(test_files)} fichiers JSON")
            else:
                self.warnings.append("Peu de fichiers de test dans tests/data/")
        else:
            self.errors.append("Dossier tests/data/ manquant")
            all_ok = False

        return all_ok

    # ==========================================================================
    # ISO 29119 - Testing Checks
    # ==========================================================================

    def validate_iso29119_testing(self) -> bool:
        """Validate test plan per ISO 29119."""
        print(colored(f"\n{Icons.TEST} ISO 29119 - Tests", Colors.BLUE))

        all_ok = True

        # Check test plan exists
        if not self.check_file_exists("docs/TEST_PLAN.md", "Plan de tests"):
            all_ok = False

        # Check test directories
        if not self.check_dir_exists("tests/data", "Données de test"):
            all_ok = False

        if not self.check_dir_exists("tests/reports", "Rapports de test"):
            all_ok = False

        # Validate test JSON files
        test_data_dir = self.root / "tests" / "data"
        if test_data_dir.is_dir():
            for json_file in test_data_dir.glob("*.json"):
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    self.passed.append(f"JSON valide: {json_file.name}")
                except json.JSONDecodeError as e:
                    self.errors.append(f"JSON invalide {json_file.name}: {e}")
                    all_ok = False

        return all_ok

    # ==========================================================================
    # Phase-specific validation
    # ==========================================================================

    def validate_phase(self, phase: int) -> bool:
        """Validate requirements for a specific phase."""
        print(colored(f"\n{Icons.PIN} Validation Phase {phase}", Colors.BLUE))

        phase_checks = {
            0: self._validate_phase0,
            1: self._validate_phase1,
            2: self._validate_phase2,
            3: self._validate_phase3,
            4: self._validate_phase4,
            5: self._validate_phase5,
        }

        if phase in phase_checks:
            return phase_checks[phase]()
        else:
            self.errors.append(f"Phase {phase} non reconnue")
            return False

    def _validate_phase0(self) -> bool:
        """Phase 0: Fondations."""
        checks = [
            self.check_dir_exists(".git", "Git initialisé"),
            self.check_file_exists(".iso/config.json", "Config ISO"),
        ]
        return all(checks)

    def _validate_phase1(self) -> bool:
        """Phase 1: Pipeline de données."""
        checks = [
            self.check_file_exists("scripts/requirements.txt", "Requirements Python"),
        ]

        # Check corpus has PDFs
        corpus_fr = self.root / "corpus" / "fr"
        corpus_intl = self.root / "corpus" / "intl"

        pdf_count = 0
        for corpus in [corpus_fr, corpus_intl]:
            if corpus.is_dir():
                pdf_count += len(list(corpus.rglob("*.pdf")))

        if pdf_count > 0:
            self.passed.append(f"Corpus: {pdf_count} PDF trouvés")
        else:
            self.errors.append("Aucun PDF dans corpus/")
            checks.append(False)

        return all(checks)

    def _validate_phase2(self) -> bool:
        """Phase 2: Prototype Android - Retrieval."""
        checks = [
            self.check_dir_exists("android/app", "App Android"),
        ]
        return all(checks)

    def _validate_phase3(self) -> bool:
        """Phase 3: Synthèse LLM."""
        checks = [
            self.check_file_exists("prompts/interpretation_v1.txt", "Prompt interprétation"),
        ]
        return all(checks)

    def _validate_phase4(self) -> bool:
        """Phase 4: Qualité et optimisation."""
        # Check for test coverage report
        return True

    def _validate_phase5(self) -> bool:
        """Phase 5: Validation et beta."""
        checks = [
            self.check_file_exists("docs/USER_GUIDE.md", "Guide utilisateur"),
        ]
        return all(checks)

    # ==========================================================================
    # Main validation
    # ==========================================================================

    def validate_all(self, phase: int = None) -> Tuple[bool, Dict]:
        """Run all validations."""
        print(colored("=" * 60, Colors.BLUE))
        print(colored("  ISO Validation - Pocket Arbiter", Colors.BLUE))
        print(colored("=" * 60, Colors.BLUE))

        # Run all ISO checks
        self.validate_iso12207_structure()
        self.validate_iso12207_docs()
        self.validate_iso42001_policy()
        self.validate_iso42001_antihallu()
        self.validate_iso25010_quality()
        self.validate_iso29119_testing()

        # Run phase-specific check if requested
        if phase is not None:
            self.validate_phase(phase)

        # Summary
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

        # Final verdict
        success = len(self.errors) == 0

        print(colored("\n" + "=" * 60, Colors.BLUE))
        if success:
            print(colored(f"  {Icons.CHECK} VALIDATION ISO REUSSIE", Colors.GREEN))
        else:
            print(colored(f"  {Icons.CROSS} VALIDATION ISO ECHOUEE", Colors.RED))
        print(colored("=" * 60, Colors.BLUE))

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


def main():
    parser = argparse.ArgumentParser(
        description="Valide la conformité ISO du projet Pocket Arbiter"
    )
    parser.add_argument(
        "--phase", "-p",
        type=int,
        choices=[0, 1, 2, 3, 4, 5],
        help="Valider une phase spécifique"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Affichage détaillé"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Sortie JSON"
    )

    args = parser.parse_args()

    # Find project root
    script_path = Path(__file__).resolve()
    project_root = script_path.parent.parent.parent

    # Run validation
    validator = ISOValidator(project_root, verbose=args.verbose)
    success, results = validator.validate_all(phase=args.phase)

    if args.json:
        print(json.dumps(results, indent=2))

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
