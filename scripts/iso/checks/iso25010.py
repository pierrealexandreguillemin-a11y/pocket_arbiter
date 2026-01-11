#!/usr/bin/env python3
"""ISO/IEC 25010 - Software Quality Checks."""

from ..base import BaseChecker
from ..utils import Icons, Colors, colored


class ISO25010Checks(BaseChecker):
    """Validates quality requirements per ISO 25010."""

    def validate_quality(self) -> bool:
        """Validate quality requirements per ISO 25010."""
        print(colored(f"\n{Icons.SPARKLE} ISO 25010 - Qualite logicielle", Colors.BLUE))

        all_ok = True

        if not self.check_file_exists("docs/QUALITY_REQUIREMENTS.md", "Exigences qualité"):
            all_ok = False

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
