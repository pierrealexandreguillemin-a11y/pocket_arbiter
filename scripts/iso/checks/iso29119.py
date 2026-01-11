#!/usr/bin/env python3
"""ISO/IEC 29119 - Software Testing Checks."""

import json

from ..base import BaseChecker
from ..utils import Icons, Colors, colored


class ISO29119Checks(BaseChecker):
    """Validates test plan and data per ISO 29119."""

    def validate_testing(self) -> bool:
        """Validate test plan per ISO 29119."""
        print(colored(f"\n{Icons.TEST} ISO 29119 - Tests", Colors.BLUE))

        all_ok = True

        if not self.check_file_exists("docs/TEST_PLAN.md", "Plan de tests"):
            all_ok = False

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
                        json.load(f)
                    self.passed.append(f"JSON valide: {json_file.name}")
                except json.JSONDecodeError as e:
                    self.errors.append(f"JSON invalide {json_file.name}: {e}")
                    all_ok = False

        return all_ok
