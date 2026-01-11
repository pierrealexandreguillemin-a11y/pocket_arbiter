#!/usr/bin/env python3
"""ISO/IEC 12207 - Software Lifecycle Checks."""

import json

from ..base import BaseChecker
from ..utils import Icons, Colors, colored


class ISO12207Checks(BaseChecker):
    """Validates project structure and docs per ISO 12207."""

    def get_current_phase(self) -> int:
        """Get current phase from ISO config."""
        config_file = self.root / ".iso" / "config.json"
        if not config_file.exists():
            return 0
        try:
            with open(config_file, "r", encoding="utf-8") as f:
                config = json.load(f)
            phases = config.get("phases", [])
            for phase in reversed(phases):
                if phase.get("status") == "completed":
                    return phase.get("id", 0) + 1
                elif phase.get("status") == "in_progress":
                    return phase.get("id", 0)
            return 0
        except Exception:
            return 0

    def validate_structure(self) -> bool:
        """Validate project structure per ISO 12207."""
        print(colored(f"\n{Icons.FOLDER} ISO 12207 - Structure projet", Colors.BLUE))

        current_phase = self.get_current_phase()

        required_dirs = [
            ("scripts", "Scripts pipeline"),
            ("corpus", "Données sources"),
            ("docs", "Documentation"),
            ("prompts", "Prompts LLM"),
            ("tests", "Tests"),
        ]

        optional_dirs_phase01 = [("android", "Projet Android")]

        required_files = [
            ("README.md", "README projet"),
            ("CLAUDE_CODE_INSTRUCTIONS.md", "Instructions Claude Code"),
            (".gitignore", "Git ignore"),
        ]

        all_ok = True

        for path, desc in required_dirs:
            if not self.check_dir_exists(path, desc):
                all_ok = False

        for path, desc in optional_dirs_phase01:
            full_path = self.root / path
            if full_path.is_dir():
                self.passed.append(f"{desc}: {path}/")
            elif current_phase >= 2:
                self.errors.append(
                    f"{desc} manquant: {path}/ (REQUIS Phase {current_phase})"
                )
                all_ok = False
            else:
                self.warnings.append(
                    f"{desc}: {path}/ (optionnel Phase {current_phase})"
                )

        for path, desc in required_files:
            if not self.check_file_exists(path, desc):
                all_ok = False

        return all_ok

    def validate_docs(self) -> bool:
        """Validate documentation per ISO 12207."""
        print(colored(f"\n{Icons.DOC} ISO 12207 - Documentation", Colors.BLUE))

        required_docs = [
            ("docs/VISION.md", "Vision projet"),
            ("docs/ARCHITECTURE.md", "Architecture technique"),
            ("docs/QUALITY_REQUIREMENTS.md", "Exigences qualité"),
            ("docs/TEST_PLAN.md", "Plan de tests"),
            ("corpus/INVENTORY.md", "Inventaire corpus"),
        ]

        all_ok = True
        for path, desc in required_docs:
            if not self.check_file_exists(path, desc):
                all_ok = False

        return all_ok
