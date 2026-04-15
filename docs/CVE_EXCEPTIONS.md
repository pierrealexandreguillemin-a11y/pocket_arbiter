# CVE Exception Register

> **ISO 27001 A.12.6** - Technical Vulnerability Management
> **Document ID**: SEC-CVE-001
> **Last reviewed**: 2026-02-21
> **Next review**: When any pinned dependency releases a compatible fix

## Active Exceptions

| CVE | Package | Version | Fix Version | Reason for Exception | Risk Assessment |
|-----|---------|---------|-------------|---------------------|-----------------|
| CVE-2025-69872 | diskcache | 5.6.3 | None (no upstream fix) | Pickle deserialization vulnerability. diskcache is a transitive dependency (via dvc-data/instructor/ragas), not used directly. Local dev tooling only, no user-facing exposure. | **Low** - no untrusted pickle input |

## Review Process

1. Run `pip-audit --strict --local` monthly
2. Check if upstream dependencies have relaxed their pins
3. Update this register and `.pre-commit-config.yaml` accordingly
4. Remove exceptions as soon as compatible fix versions become available

## Resolved Exceptions (2026-04-15)

| CVE | Package | Old Version | Fix Version | Resolution |
|-----|---------|-------------|-------------|------------|
| CVE-2026-26007 | cryptography | 44.0.3 | 46.0.7 | Upgraded cryptography to 46.0.7; docling 2.88.0 accepts it, pyopenssl 26.0.0 accepts it. pydrive2 still pins `<44` but is metadata-only conflict (no runtime issue). |
| CVE-2026-25990 | pillow | 11.3.0 | 12.2.0 | Upgraded pillow to 12.2.0; docling 2.88.0 accepts `pillow<13.0.0`. |
| CVE-2026-27448 | pyopenssl | 24.2.1 | 26.0.0 | Upgraded pyopenssl to 26.0.0 (accepts `cryptography<47,>=46.0.0`). |
| CVE-2026-27459 | pyopenssl | 24.2.1 | 26.0.0 | Same as above. |
| CVE-2026-30922 | pyasn1 | 0.6.2 | 0.6.3 | Upgraded pyasn1 to 0.6.3. |
| CVE-2024-12797 | cryptography | 43.0.3 | 46.0.7 | Resolved by cryptography 46.0.7 upgrade. |
| CVE-2026-4539 | pygments | 2.19.2 | 2.20.0 | Upgraded pygments to 2.20.0 (upstream fix now available). |
