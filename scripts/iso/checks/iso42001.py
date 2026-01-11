#!/usr/bin/env python3
"""ISO/IEC 42001 - AI Governance Checks."""

from ..base import BaseChecker
from ..utils import Icons, Colors, colored


class ISO42001Checks(BaseChecker):
    """Validates AI policy and anti-hallucination per ISO 42001."""

    def validate_policy(self) -> bool:
        """Validate AI policy per ISO 42001."""
        print(colored(f"\n{Icons.ROBOT} ISO 42001 - Gouvernance IA", Colors.BLUE))

        all_ok = True

        if not self.check_file_exists("docs/AI_POLICY.md", "Politique IA"):
            all_ok = False

        prompts_dir = self.root / "prompts"
        if prompts_dir.is_dir():
            prompt_files = list(prompts_dir.glob("*.txt")) + list(prompts_dir.glob("*.md"))
            if len(prompt_files) < 2:
                self.warnings.append("Peu de prompts versionnés dans prompts/")
            else:
                self.passed.append(f"Prompts versionnés: {len(prompt_files)} fichiers")

        if not self.check_file_exists("prompts/CHANGELOG.md", "Changelog prompts"):
            all_ok = False

        return all_ok

    def validate_antihallu(self) -> bool:
        """Check for anti-hallucination patterns in AI code."""
        print(colored(f"\n{Icons.SHIELD} ISO 42001 - Anti-hallucination", Colors.BLUE))

        dangerous_patterns = [
            "generate_without_context",
            "response_without_source",
            "skip_citation",
        ]

        ai_files = (
            list(self.root.glob("**/*llm*.py")) +
            list(self.root.glob("**/*ai*.py")) +
            list(self.root.glob("**/*generat*.py"))
        )

        issues_found = False
        for f in ai_files:
            if f.is_file() and 'test_' not in f.name:
                content = f.read_text(encoding='utf-8', errors='ignore')
                for pattern in dangerous_patterns:
                    if pattern in content.lower():
                        self.errors.append(f"Pattern dangereux '{pattern}' dans {f}")
                        issues_found = True

        if not issues_found:
            self.passed.append("Pas de patterns anti-hallucination dangereux détectés")

        return not issues_found
