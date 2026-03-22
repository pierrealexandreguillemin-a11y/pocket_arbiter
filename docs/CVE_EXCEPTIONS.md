# CVE Exception Register

> **ISO 27001 A.12.6** - Technical Vulnerability Management
> **Document ID**: SEC-CVE-001
> **Last reviewed**: 2026-02-21
> **Next review**: When any pinned dependency releases a compatible fix

## Active Exceptions

| CVE | Package | Version | Fix Version | Reason for Exception | Risk Assessment |
|-----|---------|---------|-------------|---------------------|-----------------|
| CVE-2025-69872 | diskcache | 5.6.3 | None (no upstream fix) | Pickle deserialization vulnerability. diskcache is a transitive dependency (via dvc-data/instructor/ragas), not used directly. Local dev tooling only, no user-facing exposure. | **Low** - no untrusted pickle input |
| CVE-2026-26007 | cryptography | 44.0.3 | 46.0.5 | pydrive2 pins `cryptography<44`, pyopenssl pins `cryptography<44`. Cannot upgrade without breaking these transitive deps. | **Medium** - monitor for pydrive2/pyopenssl release with relaxed pin |
| CVE-2026-25990 | pillow | 11.3.0 | 12.1.1 | docling pins `pillow<12.0.0`. Cannot upgrade without breaking docling. | **Medium** - monitor for docling release with relaxed pin |
| CVE-2026-27448 | pyopenssl | 24.2.1 | 26.0.0 | Transitive dependency (via urllib3/requests). Local dev tooling only. | **Low** - no server-side TLS usage |
| CVE-2026-27459 | pyopenssl | 24.2.1 | 26.0.0 | Same as above. | **Low** - no server-side TLS usage |
| CVE-2026-30922 | pyasn1 | 0.6.2 | 0.6.3 | Transitive dependency. Local dev tooling only. | **Low** - no ASN.1 parsing of untrusted input |
| CVE-2024-12797 | cryptography | 43.0.3 | 44.0.1 | Transitive dependency. Local build tooling only, no TLS server. | **Low** - no server-side TLS |

## Review Process

1. Run `pip-audit --strict --local` monthly
2. Check if upstream dependencies have relaxed their pins
3. Update this register and `.pre-commit-config.yaml` accordingly
4. Remove exceptions as soon as compatible fix versions become available

## Resolved Exceptions

None yet.
