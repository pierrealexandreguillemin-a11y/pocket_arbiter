#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Commit-msg Hook - Pocket Arbiter
================================
ISO Enforcement: validates commit message format.
Cross-platform (Windows, Linux, macOS).

Required format: [type] Description
Types: feat, fix, test, docs, refactor, perf, chore

ISO 12207 - Traceability requirement.
"""

import os
import re
import sys
from pathlib import Path

# Fix Windows encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')


class Colors:
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


def main() -> int:
    if len(sys.argv) < 2:
        print(colored("ERROR: No commit message file provided", Colors.RED))
        return 1

    msg_file = Path(sys.argv[1])
    if not msg_file.exists():
        print(colored("ERROR: Commit message file not found", Colors.RED))
        return 1

    commit_msg = msg_file.read_text(encoding='utf-8', errors='replace').strip()
    first_line = commit_msg.split('\n')[0]

    print("ISO Commit message validation...")
    print("=" * 50)

    errors = 0

    # Check 1: Format [type] description
    pattern = r'^\[(feat|fix|test|docs|refactor|perf|chore)\]\s+.+'
    print("  [1/4] Message format...", end=" ")
    if re.match(pattern, first_line):
        print(colored("OK", Colors.GREEN))
    else:
        print(colored("FAILED", Colors.RED))
        print()
        print("        Expected: [type] Description")
        print("        Valid types: feat, fix, test, docs, refactor, perf, chore")
        print(f"        Got: {first_line[:50]}...")
        errors += 1

    # Check 2: Minimum length
    print("  [2/4] Minimum length...", end=" ")
    if len(first_line) >= 15:
        print(colored("OK", Colors.GREEN))
    else:
        print(colored("FAILED", Colors.RED))
        print("        Minimum 15 characters required")
        errors += 1

    # Check 3: No WIP on main branch
    print("  [3/4] No WIP on main...", end=" ")
    # Get current branch
    import subprocess
    try:
        result = subprocess.run(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
            capture_output=True, text=True
        )
        branch = result.stdout.strip()
    except Exception:
        branch = 'unknown'

    if branch in ['main', 'master']:
        wip_patterns = ['WIP', 'work in progress', 'TODO', 'FIXME']
        has_wip = any(p.lower() in commit_msg.lower() for p in wip_patterns)
        if has_wip:
            print(colored("FAILED", Colors.RED))
            print("        WIP commits not allowed on main branch")
            errors += 1
        else:
            print(colored("OK", Colors.GREEN))
    else:
        print(colored("SKIP", Colors.YELLOW), f"(branch: {branch})")

    # Check 4: Co-Authored-By (BLOCKING - ISO requirement)
    print("  [4/4] Co-Author present...", end=" ")
    if 'Co-Authored-By:' in commit_msg:
        print(colored("OK", Colors.GREEN))
    else:
        print(colored("FAILED", Colors.RED))
        print("        Co-Authored-By required for traceability")
        print("        Add: Co-Authored-By: Name <email>")
        errors += 1

    # Summary
    print("=" * 50)
    if errors > 0:
        print(colored(f"BLOCKED: {errors} error(s)", Colors.RED))
        return 1
    else:
        print(colored("Commit message valid", Colors.GREEN))
        return 0


if __name__ == '__main__':
    sys.exit(main())
