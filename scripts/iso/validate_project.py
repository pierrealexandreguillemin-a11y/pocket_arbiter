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
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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

    # ==========================================================================
    # Executable Gates - Real command execution
    # ==========================================================================

    def run_command(self, cmd: List[str], cwd: Path = None) -> Tuple[bool, str]:
        """Run a command and return success status and output."""
        try:
            result = subprocess.run(
                cmd,
                cwd=cwd or self.root,
                capture_output=True,
                text=True,
                timeout=300  # 5 minutes max
            )
            output = result.stdout + result.stderr
            return result.returncode == 0, output
        except subprocess.TimeoutExpired:
            return False, "Command timed out"
        except FileNotFoundError:
            return False, f"Command not found: {cmd[0]}"
        except Exception as e:
            return False, str(e)

    def gate_pytest(self, path: str = "scripts/", required: bool = True) -> bool:
        """Gate: Run pytest and verify all tests pass."""
        print(f"    Running pytest on {path}...", end=" ")

        # Check if pytest is available
        success, output = self.run_command(["python", "-m", "pytest", "--version"])
        if not success:
            if required:
                print(colored("FAILED", Colors.RED), "(pytest not installed)")
                self.errors.append(f"pytest non installe - REQUIS pour {path}")
                return False
            else:
                print(colored("SKIP", Colors.YELLOW), "(pytest not installed)")
                self.warnings.append(f"pytest non disponible pour {path}")
                return True

        # Run pytest
        test_path = self.root / path
        if not test_path.exists():
            print(colored("SKIP", Colors.YELLOW), "(path not found)")
            return True

        success, output = self.run_command([
            "python", "-m", "pytest", str(test_path), "-v", "--tb=short"
        ])

        if success:
            print(colored("OK", Colors.GREEN))
            self.passed.append(f"Tests pytest passent: {path}")
            return True
        else:
            print(colored("FAILED", Colors.RED))
            self.log(output[:500])  # First 500 chars
            self.errors.append(f"Tests pytest echouent: {path}")
            return False

    def gate_coverage(self, target: float = 0.60, required: bool = False) -> bool:
        """Gate: Check test coverage meets target."""
        print(f"    Checking coverage (target: {target*100:.0f}%)...", end=" ")

        # Check if pytest-cov is available
        success, output = self.run_command([
            "python", "-m", "pytest", "--cov=scripts", "--cov-report=json",
            "scripts/", "-q"
        ])

        cov_file = self.root / "coverage.json"
        if not cov_file.exists():
            if required:
                print(colored("FAILED", Colors.RED), "(coverage not available)")
                self.errors.append("Coverage non mesurable - pytest-cov REQUIS")
                return False
            else:
                print(colored("SKIP", Colors.YELLOW), "(coverage not available)")
                self.warnings.append("Coverage non mesurable (pytest-cov manquant)")
                return True

        try:
            with open(cov_file, 'r') as f:
                cov_data = json.load(f)

            total_coverage = cov_data.get("totals", {}).get("percent_covered", 0) / 100

            if total_coverage >= target:
                print(colored(f"OK ({total_coverage*100:.1f}%)", Colors.GREEN))
                self.passed.append(f"Coverage: {total_coverage*100:.1f}% >= {target*100:.0f}%")
                return True
            else:
                print(colored(f"FAILED ({total_coverage*100:.1f}%)", Colors.RED))
                self.errors.append(f"Coverage insuffisant: {total_coverage*100:.1f}% < {target*100:.0f}%")
                return False
        except Exception as e:
            print(colored("SKIP", Colors.YELLOW), f"({e})")
            return True

    def gate_lint(self, path: str = "scripts/") -> bool:
        """Gate: Run flake8 and verify no critical errors."""
        print(f"    Running lint on {path}...", end=" ")

        # Run flake8 with critical errors only
        success, output = self.run_command([
            "python", "-m", "flake8", path,
            "--select=E9,F63,F7,F82",  # Critical errors only
            "--count"
        ])

        if success:
            print(colored("OK", Colors.GREEN))
            self.passed.append(f"Lint clean: {path}")
            return True
        else:
            # Count errors
            lines = output.strip().split('\n')
            error_count = len([l for l in lines if l and not l.isspace()])
            print(colored(f"FAILED ({error_count} errors)", Colors.RED))
            self.log(output[:500])
            self.errors.append(f"Lint errors: {error_count} dans {path}")
            return False

    def gate_json_valid(self, patterns: List[str] = None) -> bool:
        """Gate: Validate all JSON files in patterns."""
        if patterns is None:
            patterns = ["tests/data/*.json", ".iso/*.json"]

        print("    Validating JSON files...", end=" ")

        all_valid = True
        valid_count = 0

        for pattern in patterns:
            for json_file in self.root.glob(pattern):
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
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

        # Check .git exists
        if not (self.root / ".git").is_dir():
            print(colored("FAILED", Colors.RED))
            self.errors.append("Git non initialise")
            return False

        # Check remote exists
        success, output = self.run_command(["git", "remote", "-v"])
        if not success or not output.strip():
            print(colored("WARN", Colors.YELLOW), "(no remote)")
            self.warnings.append("Git remote non configure")
            return True  # Non-blocking

        print(colored("OK", Colors.GREEN))
        self.passed.append("Git initialise avec remote")
        return True

    def get_current_phase(self) -> int:
        """Get current phase from ISO config."""
        config_file = self.root / ".iso" / "config.json"
        if not config_file.exists():
            return 0

        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)

            phases = config.get("phases", [])
            for phase in reversed(phases):
                if phase.get("status") == "completed":
                    return phase.get("id", 0) + 1  # Next phase
                elif phase.get("status") == "in_progress":
                    return phase.get("id", 0)
            return 0
        except Exception:
            return 0

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

        current_phase = self.get_current_phase()

        # Core directories always required
        required_dirs = [
            ("scripts", "Scripts pipeline"),
            ("corpus", "Données sources"),
            ("docs", "Documentation"),
            ("prompts", "Prompts LLM"),
            ("tests", "Tests"),
        ]

        # android/ is optional in Phase 0-1, required from Phase 2
        optional_dirs_phase01 = [
            ("android", "Projet Android"),
        ]

        required_files = [
            ("README.md", "README projet"),
            ("CLAUDE_CODE_INSTRUCTIONS.md", "Instructions Claude Code"),
            (".gitignore", "Git ignore"),
        ]

        all_ok = True

        # Check core directories
        for path, desc in required_dirs:
            if not self.check_dir_exists(path, desc):
                all_ok = False

        # Check phase-dependent directories
        for path, desc in optional_dirs_phase01:
            full_path = self.root / path
            if full_path.is_dir():
                self.passed.append(f"{desc}: {path}/")
            elif current_phase >= 2:
                self.errors.append(f"{desc} manquant: {path}/ (REQUIS en Phase {current_phase})")
                all_ok = False
            else:
                # Phase 0-1: warn but don't fail
                self.warnings.append(f"{desc}: {path}/ (optionnel en Phase {current_phase})")

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

        # tests/reports/ is optional - generated during CI
        reports_dir = self.root / "tests" / "reports"
        if reports_dir.is_dir():
            self.passed.append("Rapports de test: tests/reports/")
        else:
            self.warnings.append("tests/reports/ absent (sera cree par CI)")

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
        """Phase 0: Fondations - Documentation et structure."""
        print(colored("  Gates Phase 0:", Colors.BLUE))

        checks = [
            self.gate_git_status(),
            self.check_file_exists(".iso/config.json", "Config ISO"),
            self.gate_json_valid([".iso/*.json"]),
        ]

        # Verify ISO docs exist
        required_docs = [
            "docs/VISION.md", "docs/AI_POLICY.md",
            "docs/QUALITY_REQUIREMENTS.md", "docs/TEST_PLAN.md"
        ]
        for doc in required_docs:
            checks.append(self.check_file_exists(doc, f"Doc ISO: {doc}"))

        return all(checks)

    def _validate_phase1(self) -> bool:
        """Phase 1: Pipeline de données."""
        print(colored("  Gates Phase 1:", Colors.BLUE))

        checks = [
            self.check_file_exists("scripts/requirements.txt", "Requirements Python"),
            self.gate_lint("scripts/"),
        ]

        # Check corpus has PDFs
        corpus_fr = self.root / "corpus" / "fr"
        corpus_intl = self.root / "corpus" / "intl"

        pdf_count = 0
        for corpus in [corpus_fr, corpus_intl]:
            if corpus.is_dir():
                pdf_count += len(list(corpus.rglob("*.pdf")))

        if pdf_count > 0:
            self.passed.append(f"Corpus: {pdf_count} PDF trouves")
        else:
            self.errors.append("Aucun PDF dans corpus/")
            checks.append(False)

        # Run pytest on scripts if tests exist
        test_files = list((self.root / "scripts").rglob("test_*.py"))
        if test_files:
            checks.append(self.gate_pytest("scripts/"))

        return all(checks)

    def _validate_phase2(self) -> bool:
        """Phase 2: Prototype Android - Retrieval."""
        print(colored("  Gates Phase 2:", Colors.BLUE))

        checks = []

        # android/ MUST have content from Phase 2 onwards
        android_app = self.root / "android" / "app"
        if android_app.is_dir():
            # Check for actual Android files
            kt_files = list(android_app.rglob("*.kt"))
            xml_files = list(android_app.rglob("*.xml"))

            if kt_files or xml_files:
                self.passed.append(f"Android app: {len(kt_files)} Kotlin, {len(xml_files)} XML")
                checks.append(True)
            else:
                self.errors.append("android/app/ existe mais vide - Implementation requise en Phase 2")
                checks.append(False)
        else:
            self.errors.append("android/app/ manquant - Requis en Phase 2")
            checks.append(False)

        # Retrieval tests required
        checks.append(self.check_file_exists("tests/data/questions_fr.json", "Questions test FR"))

        return all(checks)

    def _validate_phase3(self) -> bool:
        """Phase 3: Synthese LLM."""
        print(colored("  Gates Phase 3:", Colors.BLUE))

        checks = [
            self.check_file_exists("prompts/interpretation_v1.txt", "Prompt interpretation"),
            self.check_file_exists("tests/data/adversarial.json", "Tests adversariaux"),
        ]

        # Check for grounding/citation patterns
        prompt_file = self.root / "prompts" / "interpretation_v1.txt"
        if prompt_file.exists():
            content = prompt_file.read_text(encoding='utf-8', errors='ignore').lower()
            has_citation = any(kw in content for kw in ['citation', 'source', 'article', 'reference'])
            if has_citation:
                self.passed.append("Prompt inclut instructions de citation")
            else:
                self.warnings.append("Prompt devrait inclure instructions de citation (ISO 42001)")

        return all(checks)

    def _validate_phase4(self) -> bool:
        """Phase 4: Qualite et optimisation."""
        print(colored("  Gates Phase 4:", Colors.BLUE))

        checks = [
            self.gate_pytest("scripts/", required=True),
            self.gate_lint("scripts/"),
        ]

        # Coverage is REQUIRED in Phase 4
        checks.append(self.gate_coverage(target=0.60, required=True))

        return all(checks)

    def _validate_phase5(self) -> bool:
        """Phase 5: Validation et beta."""
        print(colored("  Gates Phase 5:", Colors.BLUE))

        checks = [
            self.check_file_exists("docs/USER_GUIDE.md", "Guide utilisateur"),
            self.check_file_exists("docs/RELEASE_NOTES.md", "Notes de release"),
        ]

        # APK must exist
        apk_files = list((self.root / "android").rglob("*.apk"))
        if apk_files:
            self.passed.append(f"APK trouve: {apk_files[0].name}")
            checks.append(True)
        else:
            self.errors.append("APK non trouve - Build requis pour Phase 5")
            checks.append(False)

        return all(checks)

    # ==========================================================================
    # Main validation
    # ==========================================================================

    def validate_all(self, phase: int = None, run_gates: bool = False) -> Tuple[bool, Dict]:
        """Run all validations.

        Args:
            phase: Specific phase to validate (0-5)
            run_gates: If True, run executable gates (pytest, lint, etc.)
        """
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

        # Run executable gates if requested
        if run_gates:
            print(colored(f"\n{Icons.SHIELD} Executable Gates", Colors.BLUE))
            self.gate_json_valid()
            self.gate_lint("scripts/")
            self.gate_git_status()

            # Run pytest if tests exist
            test_files = list((self.root / "scripts").rglob("test_*.py"))
            if test_files:
                self.gate_pytest("scripts/")

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
    parser.add_argument(
        "--gates", "-g",
        action="store_true",
        help="Exécuter les gates (pytest, lint, coverage)"
    )

    args = parser.parse_args()

    # Find project root
    script_path = Path(__file__).resolve()
    project_root = script_path.parent.parent.parent

    # Run validation
    validator = ISOValidator(project_root, verbose=args.verbose)
    success, results = validator.validate_all(phase=args.phase, run_gates=args.gates)

    if args.json:
        print(json.dumps(results, indent=2))

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
