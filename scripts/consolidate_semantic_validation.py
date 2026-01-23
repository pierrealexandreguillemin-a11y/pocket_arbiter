#!/usr/bin/env python3
"""Consolidate semantic validation results from 6 agents and apply corrections."""

import json
from datetime import datetime
from pathlib import Path


def consolidate_results() -> dict:
    """Load and consolidate all agent results."""
    all_results = []
    summaries = []

    for i in range(1, 7):
        filepath = Path(f'data/semantic_validation/agent_{i}_results.json')
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        all_results.extend(data['results'])
        summaries.append({
            'agent': i,
            'batch': data.get('batch', i),
            'summary': data.get('summary', {})
        })

    # Calculate totals
    total_summary = {
        'total': len(all_results),
        'keep': sum(1 for r in all_results if r['verdict'] == 'KEEP'),
        'wrong_corrected': sum(1 for r in all_results if r['verdict'] == 'WRONG_CORRECTED'),
        'wrong_no_chunk': sum(1 for r in all_results if r['verdict'] == 'WRONG_NO_CHUNK'),
        'partial_acceptable': sum(1 for r in all_results if r['verdict'] == 'PARTIAL_ACCEPTABLE'),
        'partial_improved': sum(1 for r in all_results if r['verdict'] == 'PARTIAL_IMPROVED'),
    }

    # Extract corrections
    corrections = []
    for r in all_results:
        if r.get('new_chunk_id') and r['new_chunk_id'] not in (None, 'null'):
            corrections.append({
                'id': r['id'],
                'old_chunk_id': r['current_chunk_id'],
                'new_chunk_id': r['new_chunk_id'],
                'verdict': r['verdict'],
                'reason': r.get('reason', ''),
                'confidence': r.get('confidence', 'MEDIUM')
            })

    consolidated = {
        'validated_at': datetime.now().isoformat(),
        'agent_summaries': summaries,
        'total_summary': total_summary,
        'corrections_count': len(corrections),
        'corrections': corrections,
        'all_results': all_results
    }

    return consolidated


def apply_corrections(corrections: list[dict]) -> tuple[int, int]:
    """Apply corrections to gold standard files."""
    # Load gold standards
    with open('tests/data/gold_standard_fr.json', 'r', encoding='utf-8') as f:
        gs_fr = json.load(f)
    with open('tests/data/gold_standard_intl.json', 'r', encoding='utf-8') as f:
        gs_intl = json.load(f)

    # Create lookup by question id
    corrections_map = {c['id']: c['new_chunk_id'] for c in corrections}

    # Apply corrections to FR
    fr_updated = 0
    for q in gs_fr['questions']:
        if q['id'] in corrections_map:
            q['expected_chunk_id'] = corrections_map[q['id']]
            fr_updated += 1

    # Apply corrections to INTL
    intl_updated = 0
    for q in gs_intl['questions']:
        if q['id'] in corrections_map:
            q['expected_chunk_id'] = corrections_map[q['id']]
            intl_updated += 1

    # Save updated gold standards
    with open('tests/data/gold_standard_fr.json', 'w', encoding='utf-8') as f:
        json.dump(gs_fr, f, ensure_ascii=False, indent=2)
    with open('tests/data/gold_standard_intl.json', 'w', encoding='utf-8') as f:
        json.dump(gs_intl, f, ensure_ascii=False, indent=2)

    return fr_updated, intl_updated


def main():
    print("=== CONSOLIDATION VALIDATION SEMANTIQUE ===\n")

    # Consolidate results
    consolidated = consolidate_results()

    # Save consolidated results
    output_path = Path('data/semantic_validation/consolidated_results.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(consolidated, f, ensure_ascii=False, indent=2)
    print(f"Resultats consolides: {output_path}")

    # Print summary
    ts = consolidated['total_summary']
    print(f"\n=== RESUME FINAL ===")
    print(f"Total questions: {ts['total']}")
    print(f"  KEEP: {ts['keep']} ({ts['keep']/ts['total']*100:.1f}%)")
    print(f"  WRONG_CORRECTED: {ts['wrong_corrected']}")
    print(f"  WRONG_NO_CHUNK: {ts['wrong_no_chunk']}")
    print(f"  PARTIAL_ACCEPTABLE: {ts['partial_acceptable']}")
    print(f"  PARTIAL_IMPROVED: {ts['partial_improved']}")
    print(f"\nCorrections a appliquer: {consolidated['corrections_count']}")

    # Apply corrections
    print("\n=== APPLICATION DES CORRECTIONS ===")
    fr_updated, intl_updated = apply_corrections(consolidated['corrections'])
    print(f"FR questions mises a jour: {fr_updated}")
    print(f"INTL questions mises a jour: {intl_updated}")
    print(f"Total corrections appliquees: {fr_updated + intl_updated}")

    # Final status
    print("\n=== STATUS FINAL ===")
    print("Gold standards mis a jour:")
    print("  - tests/data/gold_standard_fr.json")
    print("  - tests/data/gold_standard_intl.json")
    print("\nValidation semantique ISO 42001 completee!")


if __name__ == '__main__':
    main()
