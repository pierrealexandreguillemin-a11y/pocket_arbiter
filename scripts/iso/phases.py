#!/usr/bin/env python3
"""Phase-specific validation for ISO compliance."""

from pathlib import Path
from typing import List

from .base import BaseChecker
from .gates import ExecutableGates
from .utils import Icons, Colors, colored


class PhaseValidator(BaseChecker):
    """Validates requirements for specific project phases."""

    def __init__(self, root: Path, errors: List[str], warnings: List[str],
                 passed: List[str], verbose: bool = False):
        super().__init__(root, errors, warnings, passed, verbose)
        self.gates = ExecutableGates(root, errors, warnings, passed, verbose)

    def validate_phase(self, phase: int) -> bool:
        """Validate requirements for a specific phase."""
        print(colored(f"\n{Icons.PIN} Validation Phase {phase}", Colors.BLUE))

        phase_methods = {
            0: self._phase0, 1: self._phase1, 2: self._phase2,
            3: self._phase3, 4: self._phase4, 5: self._phase5,
        }

        if phase in phase_methods:
            return phase_methods[phase]()
        self.errors.append(f"Phase {phase} non reconnue")
        return False

    def _phase0(self) -> bool:
        """Phase 0: Foundations."""
        print(colored("  Gates Phase 0:", Colors.BLUE))
        checks = [
            self.gates.gate_git_status(),
            self.check_file_exists(".iso/config.json", "Config ISO"),
            self.gates.gate_json_valid([".iso/*.json"]),
        ]
        for doc in ["docs/VISION.md", "docs/AI_POLICY.md",
                    "docs/QUALITY_REQUIREMENTS.md", "docs/TEST_PLAN.md"]:
            checks.append(self.check_file_exists(doc, f"Doc ISO: {doc}"))
        return all(checks)

    def _phase1(self) -> bool:
        """Phase 1: Data Pipeline."""
        print(colored("  Gates Phase 1:", Colors.BLUE))
        checks = [
            self.check_file_exists("scripts/requirements.txt", "Requirements Python"),
            self.gates.gate_lint("scripts/"),
        ]

        pdf_count = 0
        for corpus in [self.root / "corpus" / "fr", self.root / "corpus" / "intl"]:
            if corpus.is_dir():
                pdf_count += len(list(corpus.rglob("*.pdf")))

        if pdf_count > 0:
            self.passed.append(f"Corpus: {pdf_count} PDF trouves")
        else:
            self.errors.append("Aucun PDF dans corpus/")
            checks.append(False)

        if list((self.root / "scripts").rglob("test_*.py")):
            checks.append(self.gates.gate_pytest("scripts/"))

        return all(checks)

    def _phase2(self) -> bool:
        """Phase 2: Android Retrieval."""
        print(colored("  Gates Phase 2:", Colors.BLUE))
        checks = []

        android_app = self.root / "android" / "app"
        if android_app.is_dir():
            kt = list(android_app.rglob("*.kt"))
            xml = list(android_app.rglob("*.xml"))
            if kt or xml:
                self.passed.append(f"Android: {len(kt)} Kotlin, {len(xml)} XML")
                checks.append(True)
            else:
                self.errors.append("android/app/ vide - Implementation requise")
                checks.append(False)
        else:
            self.errors.append("android/app/ manquant - Requis Phase 2")
            checks.append(False)

        checks.append(self.check_file_exists("tests/data/questions_fr.json", "Questions FR"))
        return all(checks)

    def _phase3(self) -> bool:
        """Phase 3: LLM Synthesis."""
        print(colored("  Gates Phase 3:", Colors.BLUE))
        checks = [
            self.check_file_exists("prompts/interpretation_v1.txt", "Prompt interpretation"),
            self.check_file_exists("tests/data/adversarial.json", "Tests adversariaux"),
        ]

        prompt_file = self.root / "prompts" / "interpretation_v1.txt"
        if prompt_file.exists():
            content = prompt_file.read_text(encoding='utf-8', errors='ignore').lower()
            if any(kw in content for kw in ['citation', 'source', 'article']):
                self.passed.append("Prompt inclut instructions de citation")
            else:
                self.warnings.append("Prompt devrait inclure instructions citation")

        return all(checks)

    def _phase4(self) -> bool:
        """Phase 4: Quality & Optimization."""
        print(colored("  Gates Phase 4:", Colors.BLUE))
        return all([
            self.gates.gate_pytest("scripts/", required=True),
            self.gates.gate_lint("scripts/"),
            self.gates.gate_coverage(target=0.60, required=True),
        ])

    def _phase5(self) -> bool:
        """Phase 5: Validation & Release."""
        print(colored("  Gates Phase 5:", Colors.BLUE))
        checks = [
            self.check_file_exists("docs/USER_GUIDE.md", "Guide utilisateur"),
            self.check_file_exists("docs/RELEASE_NOTES.md", "Notes de release"),
        ]

        apks = list((self.root / "android").rglob("*.apk"))
        if apks:
            self.passed.append(f"APK: {apks[0].name}")
            checks.append(True)
        else:
            self.errors.append("APK non trouve - Build requis Phase 5")
            checks.append(False)

        return all(checks)
