#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pre-commit Hook - Pocket Arbiter
================================
ISO Enforcement: validates compliance before each commit.
Cross-platform (Windows, Linux, macOS).

Installation:
    git config core.hooksPath .githooks

ISO Standards enforced:
- ISO 25010: No critical TODOs, no secrets, valid JSON
- ISO 42001: No hallucination-prone patterns in AI code
- ISO 12207: Required documentation exists
"""

import os
import re
import sys
import json
import subprocess
from pathlib import Path
from typing import List, Tuple

# Fix Windows encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')


class Colors:
    """Terminal colors with Windows fallback."""
    if sys.platform == 'win32' and not os.environ.get('WT_SESSION'):
        RED = ''
        GREEN = ''
        YELLOW = ''
        NC = ''
    else:
        RED = '\033[0;31m'
        GREEN = '\033[0;32m'
        YELLOW = '\033[1;33m'
        NC = '\033[0m'


def colored(text: str, color: str) -> str:
    return f"{color}{text}{Colors.NC}"


def run_git(args: List[str]) -> Tuple[int, str]:
    """Run git command and return (returncode, output)."""
    try:
        result = subprocess.run(
            ['git'] + args,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace'
        )
        return result.returncode, result.stdout.strip()
    except Exception as e:
        return 1, str(e)


def get_staged_files(extensions: List[str] = None) -> List[Path]:
    """Get list of staged files, optionally filtered by extension."""
    _, output = run_git(['diff', '--cached', '--name-only', '--diff-filter=ACM'])
    if not output:
        return []

    files = [Path(f) for f in output.split('\n') if f]

    if extensions:
        files = [f for f in files if f.suffix.lower() in extensions]

    return files


def check_critical_todos(files: List[Path]) -> Tuple[bool, List[str]]:
    """Check for critical TODO/FIXME markers. ISO 25010 - Maintainability."""
    patterns = [
        r'TODO.*CRITICAL',
        r'FIXME.*CRITICAL',
        r'TODO.*URGENT',
        r'FIXME.*URGENT',
        r'XXX',
    ]
    combined = '|'.join(patterns)

    issues = []
    for f in files:
        if f.exists():
            try:
                content = f.read_text(encoding='utf-8', errors='ignore')
                for i, line in enumerate(content.split('\n'), 1):
                    if re.search(combined, line, re.IGNORECASE):
                        issues.append(f"{f}:{i}: {line.strip()[:60]}")
            except Exception:
                pass

    return len(issues) == 0, issues


def check_secrets(files: List[Path]) -> Tuple[bool, List[str]]:
    """Check for hardcoded secrets. ISO 25010 - Security."""
    patterns = [
        r'password\s*=\s*["\'][^"\']+["\']',
        r'api_key\s*=\s*["\'][^"\']+["\']',
        r'secret\s*=\s*["\'][^"\']+["\']',
        r'token\s*=\s*["\'][A-Za-z0-9]{20,}["\']',
        r'private_key\s*=\s*["\']',
    ]
    combined = '|'.join(patterns)

    issues = []
    for f in files:
        if f.exists():
            try:
                content = f.read_text(encoding='utf-8', errors='ignore')
                for i, line in enumerate(content.split('\n'), 1):
                    # Skip comments
                    if line.strip().startswith('#') or line.strip().startswith('//'):
                        continue
                    if re.search(combined, line, re.IGNORECASE):
                        issues.append(f"{f}:{i}: potential secret")
            except Exception:
                pass

    return len(issues) == 0, issues


def check_json_validity(files: List[Path]) -> Tuple[bool, List[str]]:
    """Check JSON files are valid. ISO 25010 - Reliability."""
    issues = []
    for f in files:
        if f.exists():
            try:
                with open(f, 'r', encoding='utf-8') as fp:
                    json.load(fp)
            except json.JSONDecodeError as e:
                issues.append(f"{f}: {e.msg} at line {e.lineno}")
            except Exception as e:
                issues.append(f"{f}: {str(e)}")

    return len(issues) == 0, issues


def check_iso_docs() -> Tuple[bool, List[str]]:
    """Check required ISO documentation exists. ISO 12207."""
    required = [
        'docs/VISION.md',
        'docs/ARCHITECTURE.md',
        'docs/AI_POLICY.md',
        'docs/QUALITY_REQUIREMENTS.md',
        'docs/TEST_PLAN.md',
    ]

    missing = []
    for doc in required:
        if not Path(doc).exists():
            missing.append(doc)

    return len(missing) == 0, missing


def check_ai_safety(files: List[Path]) -> Tuple[bool, List[str]]:
    """Check for hallucination-prone patterns. ISO 42001."""
    # Only check files that might contain AI code
    ai_files = [f for f in files if any(
        kw in f.name.lower() for kw in ['llm', 'ai', 'prompt', 'generat', 'model']
    )]

    dangerous_patterns = [
        r'generate.*without.*context',
        r'response.*without.*source',
        r'skip.*citation',
        r'no.*grounding',
    ]
    combined = '|'.join(dangerous_patterns)

    issues = []
    for f in ai_files:
        if f.exists() and 'validate_project' not in str(f):
            try:
                content = f.read_text(encoding='utf-8', errors='ignore')
                for i, line in enumerate(content.split('\n'), 1):
                    if re.search(combined, line, re.IGNORECASE):
                        issues.append(f"{f}:{i}: AI safety concern")
            except Exception:
                pass

    return len(issues) == 0, issues


def check_python_syntax(files: List[Path]) -> Tuple[bool, List[str]]:
    """Check Python files have valid syntax. ISO 25010."""
    issues = []
    for f in files:
        if f.exists():
            try:
                with open(f, 'r', encoding='utf-8') as fp:
                    source = fp.read()
                compile(source, str(f), 'exec')
            except SyntaxError as e:
                issues.append(f"{f}:{e.lineno}: {e.msg}")

    return len(issues) == 0, issues


def run_flake8(path: str = "scripts/") -> Tuple[bool, str]:
    """Run flake8 lint. ISO 25010 - Code Quality - MANDATORY."""
    try:
        result = subprocess.run(
            [sys.executable, '-m', 'flake8', path,
             '--select=E9,F63,F7,F82',  # Critical errors only
             '--show-source', '--statistics'],
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',
            timeout=60
        )
        output = result.stdout + result.stderr
        return result.returncode == 0, output.strip()
    except subprocess.TimeoutExpired:
        return False, "Timeout: lint took > 60s"
    except Exception as e:
        return False, str(e)


def run_pytest() -> Tuple[bool, str]:
    """Run pytest on ISO validator tests. ISO 29119 - MANDATORY."""
    try:
        result = subprocess.run(
            [sys.executable, '-m', 'pytest', 'scripts/iso/tests/', '-q', '--tb=line'],
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',
            timeout=120
        )
        output = result.stdout + result.stderr
        return result.returncode == 0, output.strip()
    except subprocess.TimeoutExpired:
        return False, "Timeout: tests took > 120s"
    except Exception as e:
        return False, str(e)


def run_coverage() -> Tuple[bool, float, str]:
    """Run pytest with coverage, return (pass, percentage, output). ISO 29119."""
    try:
        result = subprocess.run(
            [sys.executable, '-m', 'pytest', 'scripts/iso/tests/',
             '--cov=scripts/iso', '--cov-report=term-missing',
             '--cov-fail-under=75', '-q'],
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',
            timeout=180
        )
        output = result.stdout + result.stderr

        # Extract coverage percentage
        import re
        match = re.search(r'TOTAL\s+\d+\s+\d+\s+(\d+)%', output)
        coverage = float(match.group(1)) if match else 0.0

        return result.returncode == 0, coverage, output.strip()
    except subprocess.TimeoutExpired:
        return False, 0.0, "Timeout: coverage took > 180s"
    except Exception as e:
        return False, 0.0, str(e)


def main() -> int:
    print("ISO Pre-commit validation...")
    print("=" * 50)

    errors = 0
    warnings = 0

    # Get staged files (exclude hook scripts themselves)
    exclude_paths = ['.githooks/', 'scripts/iso/validate_project.py', 'scripts/iso/pre-commit']
    code_files = [f for f in get_staged_files(['.py', '.kt', '.java', '.js', '.ts'])
                  if not any(excl in str(f).replace('\\', '/') for excl in exclude_paths)]
    json_files = get_staged_files(['.json'])
    python_files = [f for f in get_staged_files(['.py'])
                    if not any(excl in str(f).replace('\\', '/') for excl in exclude_paths)]

    # Check 1: Critical TODOs (BLOCKING)
    print("  [1/9] Critical TODO/FIXME...", end=" ")
    ok, issues = check_critical_todos(code_files)
    if ok:
        print(colored("OK", Colors.GREEN))
    else:
        print(colored("FAILED", Colors.RED))
        for issue in issues:
            print(f"        {issue}")
        errors += 1

    # Check 2: Secrets (BLOCKING)
    print("  [2/9] Hardcoded secrets...", end=" ")
    ok, issues = check_secrets(code_files)
    if ok:
        print(colored("OK", Colors.GREEN))
    else:
        print(colored("FAILED", Colors.RED))
        for issue in issues:
            print(f"        {issue}")
        errors += 1

    # Check 3: JSON validity (BLOCKING)
    print("  [3/9] JSON validity...", end=" ")
    if json_files:
        ok, issues = check_json_validity(json_files)
        if ok:
            print(colored("OK", Colors.GREEN))
        else:
            print(colored("FAILED", Colors.RED))
            for issue in issues:
                print(f"        {issue}")
            errors += 1
    else:
        print(colored("SKIP", Colors.YELLOW), "(no JSON files)")

    # Check 4: ISO documentation (BLOCKING)
    print("  [4/9] ISO documentation...", end=" ")
    ok, issues = check_iso_docs()
    if ok:
        print(colored("OK", Colors.GREEN))
    else:
        print(colored("FAILED", Colors.RED))
        for issue in issues:
            print(f"        Missing: {issue}")
        errors += 1

    # Check 5: AI safety (BLOCKING)
    print("  [5/9] AI safety patterns...", end=" ")
    ok, issues = check_ai_safety(code_files)
    if ok:
        print(colored("OK", Colors.GREEN))
    else:
        print(colored("FAILED", Colors.RED))
        for issue in issues:
            print(f"        {issue}")
        errors += 1

    # Check 6: Python syntax (BLOCKING)
    print("  [6/9] Python syntax...", end=" ")
    if python_files:
        ok, issues = check_python_syntax(python_files)
        if ok:
            print(colored("OK", Colors.GREEN))
        else:
            print(colored("FAILED", Colors.RED))
            for issue in issues:
                print(f"        {issue}")
            errors += 1
    else:
        print(colored("SKIP", Colors.YELLOW), "(no Python files)")

    # Check 7: Lint/flake8 (BLOCKING) - ISO 25010
    print("  [7/9] Lint (flake8)...", end=" ")
    if Path('scripts/').exists():
        ok, output = run_flake8('scripts/')
        if ok:
            print(colored("OK", Colors.GREEN))
        else:
            print(colored("FAILED", Colors.RED))
            lines = output.split('\n')[:5]
            for line in lines:
                if line.strip():
                    print(f"        {line}")
            errors += 1
    else:
        print(colored("SKIP", Colors.YELLOW), "(no scripts/)")

    # Check 8: Unit tests (BLOCKING) - ISO 29119
    print("  [8/9] Unit tests (pytest)...", end=" ")
    if Path('scripts/iso/tests').exists():
        ok, output = run_pytest()
        if ok:
            # Extract passed count from output
            import re
            match = re.search(r'(\d+) passed', output)
            count = match.group(1) if match else '?'
            print(colored(f"OK ({count} passed)", Colors.GREEN))
        else:
            print(colored("FAILED", Colors.RED))
            # Show last few lines of output
            lines = output.split('\n')[-5:]
            for line in lines:
                if line.strip():
                    print(f"        {line}")
            errors += 1
    else:
        print(colored("SKIP", Colors.YELLOW), "(no tests directory)")

    # Check 9: Coverage (BLOCKING >= 75%) - ISO 29119
    print("  [9/9] Coverage (>= 75%)...", end=" ")
    if Path('scripts/iso/tests').exists():
        ok, coverage, output = run_coverage()
        if ok:
            print(colored(f"OK ({coverage:.0f}%)", Colors.GREEN))
        else:
            print(colored(f"FAILED ({coverage:.0f}%)", Colors.RED))
            print("        Minimum coverage: 75%")
            errors += 1
    else:
        print(colored("SKIP", Colors.YELLOW), "(no tests directory)")

    # Summary
    print("=" * 50)
    if errors > 0:
        print(colored(f"BLOCKED: {errors} error(s) found", Colors.RED))
        print("Fix issues above before committing.")
        return 1
    else:
        print(colored("All ISO checks passed", Colors.GREEN))
        return 0


if __name__ == '__main__':
    sys.exit(main())
