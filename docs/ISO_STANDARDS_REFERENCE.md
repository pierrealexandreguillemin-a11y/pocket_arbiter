# ISO Standards Reference - Pocket Arbiter

## Document Control
| Version | Date | Author | Description |
|---------|------|--------|-------------|
| 1.0 | 2026-01-11 | Claude Opus 4.5 | Initial version |

---

## 1. Applicable Standards

### 1.1 ISO/IEC 25010:2011 - Software Quality
**Scope**: Systems and software Quality Requirements and Evaluation (SQuaRE)

**Application to Pocket Arbiter**:
- **Functional Suitability**: RAG system must accurately retrieve and cite chess regulations
- **Performance Efficiency**: Response time < 5 seconds on mobile
- **Compatibility**: Android 8+ support
- **Usability**: French interface, accessible to non-technical arbiters
- **Reliability**: Offline capability, graceful degradation
- **Security**: No sensitive data collection, local-first processing
- **Maintainability**: Modular architecture, documented code
- **Portability**: Standard Android APIs

**Enforcement**:
- Quality requirements documented in `docs/QUALITY_REQUIREMENTS.md`
- Automated validation via `scripts/iso/validate_project.py`
- CI gate: `python-quality` job

---

### 1.2 ISO/IEC 42001:2023 - AI Management Systems
**Scope**: Requirements for establishing, implementing, and improving AI management systems

> **Document de référence détaillé**: [`docs/AI_POLICY.md`](AI_POLICY.md)

**Résumé des contrôles** (voir AI_POLICY.md pour détails):

| Contrôle | Référence AI_POLICY.md |
|----------|------------------------|
| Analyse des risques | Section 3 (AI-R01 à AI-R06) |
| Mesures anti-hallucination | Section 3.2 |
| Spécifications modèles | Section 5 |
| Contrôles ISO 42001 Annex A | Section 4 |
| Disclaimer utilisateur | Section 7.1 |
| Checklist release | Section 9 |

**Principes clés**:
- **Tolérance zéro hallucination**: Chaque réponse doit citer sa source
- **Grounding obligatoire**: Le LLM ne génère jamais librement
- **Humain décisionnaire**: L'app conseille, l'arbitre décide
- **Transparence**: Disclaimer visible, sources citées

**Enforcement**:
- Pre-commit: Patterns dangereux bloqués
- CI: Documentation AI_POLICY.md requise
- Tests: Suite adversariale anti-hallucination

---

### 1.3 ISO/IEC 12207:2017 - Software Life Cycle Processes
**Scope**: Framework for software life cycle processes

**Application to Pocket Arbiter**:

| Process | Implementation |
|---------|----------------|
| Agreement | CLAUDE_CODE_INSTRUCTIONS.md |
| Project Planning | docs/VISION.md, .iso/config.json |
| Quality Assurance | CI/CD pipeline, pre-commit hooks |
| Configuration Management | Git, DVC for large files |
| Documentation | docs/, prompts/, corpus/INVENTORY.md |
| Verification | pytest, manual testing |
| Validation | User acceptance tests (Phase 5) |

**Required Structure**:
```
pocket_arbiter/
├── android/          # Android application
├── scripts/          # Python pipeline scripts
├── corpus/           # Source documents (DVC tracked)
├── docs/             # Documentation
├── prompts/          # LLM prompts (versioned)
├── tests/            # Test data and reports
├── .iso/             # ISO configuration
└── .githooks/        # Git hooks
```

**Phase Gates**:
| Phase | Gate | Validation |
|-------|------|------------|
| 0 | docs_exist, structure_valid, git_initialized | ISO docs, directories, git remote |
| 1 | specs_exist, scripts_tested, corpus_processed | Pipeline functional |
| 2 | ui_implemented, retrieval_tested | Android app retrieves correctly |
| 3 | grounding_verified, hallucination_tested | LLM responses are grounded |
| 4 | coverage_met, lint_clean, performance_validated | Quality metrics met |
| 5 | user_tested, docs_complete, release_ready | Production ready |

---

### 1.4 ISO/IEC 29119:2013 - Software Testing
**Scope**: Software testing processes, documentation, and techniques

**Application to Pocket Arbiter**:

**Test Plan**: `docs/TEST_PLAN.md`

**Test Levels**:
1. **Unit Tests**: pytest for Python scripts
2. **Integration Tests**: Pipeline end-to-end
3. **System Tests**: Full RAG pipeline
4. **Acceptance Tests**: User scenarios

**Test Data**:
| File | Purpose |
|------|---------|
| `tests/data/questions_fr.json` | French regulation questions |
| `tests/data/questions_intl.json` | FIDE questions |
| `tests/data/adversarial.json` | Hallucination test cases |

**Coverage Requirements**:
- Minimum: 60%
- Target: 80%
- Critical paths: 100%

**Test Automation**:
```bash
# Run all tests
pytest scripts/iso/ -v

# With coverage
pytest scripts/iso/ --cov=scripts/iso --cov-fail-under=60

# Specific phase validation
python scripts/iso/validate_project.py --phase 1 --gates
```

---

## 2. Enforcement Mechanisms

### 2.1 Pre-commit Hooks
Location: `.githooks/pre-commit.py`

| Check | Blocking | Description |
|-------|----------|-------------|
| Critical TODOs | Yes | No CRITICAL/URGENT markers |
| Secrets | Yes | No hardcoded passwords/keys |
| JSON validity | Yes | All JSON files parse correctly |
| ISO documentation | Yes | Required docs exist |
| AI safety | Yes | No dangerous AI patterns |
| Python syntax | Yes | All .py files compile |

### 2.2 Commit Message Hook
Location: `.githooks/commit-msg.py`

Format: `[type] Description`

Valid types: `feat`, `fix`, `test`, `docs`, `refactor`, `perf`, `chore`

Requirements:
- Minimum 15 characters
- No WIP on main branch
- Co-Authored-By required (traceability)

### 2.3 CI/CD Pipeline
Location: `.github/workflows/ci.yml`

| Job | Dependencies | Blocking |
|-----|--------------|----------|
| iso-validation | - | Yes |
| validate-test-data | iso-validation | Yes |
| python-quality | iso-validation | Yes |
| docs-quality | iso-validation | Yes |
| build-android | all above | Manual only |

**Key Principle**: No `|| true`, no `continue-on-error`

---

## 3. Compliance Checklist

### Phase 0: Foundations
- [ ] `docs/VISION.md` exists and is complete
- [ ] `docs/AI_POLICY.md` defines anti-hallucination policy
- [ ] `docs/QUALITY_REQUIREMENTS.md` lists ISO 25010 requirements
- [ ] `docs/TEST_PLAN.md` defines test strategy
- [ ] Git repository initialized with remote
- [ ] DVC initialized for corpus tracking
- [ ] Pre-commit hooks configured
- [ ] CI/CD pipeline active

### Phase 1: Data Pipeline
- [ ] PDF corpus tracked by DVC
- [ ] `corpus/INVENTORY.md` documents all sources
- [ ] Extraction scripts tested
- [ ] Chunking strategy defined
- [ ] Embedding model selected

### Phase 2: Android Retrieval
- [ ] UI mockups approved
- [ ] Retrieval API implemented
- [ ] Recall >= 80% on test questions
- [ ] Offline mode functional

### Phase 3: LLM Synthesis
- [ ] Interpretation prompt versioned
- [ ] Citation format defined
- [ ] Hallucination rate = 0% on adversarial tests
- [ ] Response time < 5s

### Phase 4: Quality & Optimization
- [ ] Test coverage >= 60%
- [ ] No critical lint warnings
- [ ] Performance benchmarks met
- [ ] Accessibility verified

### Phase 5: Validation & Release
- [ ] User guide complete
- [ ] Beta testers validated (n >= 5)
- [ ] Release notes written
- [ ] Signed APK generated

---

## 4. Audit Trail

### Validation Command
```bash
python scripts/iso/validate_project.py --phase N --gates --verbose
```

### Reports Location
- Test reports: `tests/reports/`
- Coverage: `coverage.json`, `htmlcov/`
- Validation logs: CI artifacts

### Document References
| Document | ISO Reference | Rôle |
|----------|---------------|------|
| `docs/VISION.md` | ISO 12207 - Project Planning | Objectifs projet |
| `docs/AI_POLICY.md` | ISO 42001 - AI Governance | **Document maître IA** |
| `docs/QUALITY_REQUIREMENTS.md` | ISO 25010 - Quality Model | Exigences qualité |
| `docs/TEST_PLAN.md` | ISO 29119 - Test Documentation | Stratégie de test |
| `docs/ISO_STANDARDS_REFERENCE.md` | Toutes normes | **Index et vue d'ensemble** |
| `corpus/INVENTORY.md` | ISO 12207 - Configuration Management | Traçabilité corpus |
| `prompts/CHANGELOG.md` | ISO 42001 - AI Transparency | Historique prompts |

**Hiérarchie documentaire**:
```
ISO_STANDARDS_REFERENCE.md (index)
├── AI_POLICY.md (détail ISO 42001)
├── QUALITY_REQUIREMENTS.md (détail ISO 25010)
├── TEST_PLAN.md (détail ISO 29119)
└── VISION.md (détail ISO 12207)
```

---

## 5. Non-Conformance Handling

### Severity Levels
| Level | Description | Action |
|-------|-------------|--------|
| BLOCKER | Violates ISO requirement | Immediate fix, blocks release |
| CRITICAL | Affects core functionality | Fix before phase gate |
| MAJOR | Quality concern | Fix in current phase |
| MINOR | Enhancement opportunity | Backlog |

### Escalation Path
1. Pre-commit hook failure → Developer fixes locally
2. CI failure → PR blocked until resolved
3. Phase gate failure → Phase cannot advance
4. Audit finding → Documented in issue tracker

---

## 6. Continuous Improvement

### Metrics Tracked
| Metric | Target | Current | ISO Reference |
|--------|--------|---------|---------------|
| Test Pass Rate | 100% | TBD | ISO 29119 |
| Code Coverage | 60% | 74% | ISO 25010 |
| Lint Warnings | 0 | TBD | ISO 25010 |
| Retrieval Recall | 80% | TBD | ISO 25010 |
| Hallucination Rate | 0% | TBD | ISO 42001 |
| Response Latency | < 5s | TBD | ISO 25010 |

### Review Cadence
- Pre-commit: Every commit
- CI: Every push
- Phase gate: Phase completion
- Full audit: Major release

---

*This document is maintained as part of the Pocket Arbiter ISO compliance system.*
