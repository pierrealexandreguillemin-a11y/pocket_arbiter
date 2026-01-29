"""
Unified Training Data Generation Pipeline.

Implements UNIFIED_TRAINING_DATA_SPEC.md workflow:
1. map_pages_to_chunks - Map GS questions to corpus chunks
2. reformulate_questions - BY DESIGN reformulation with chunk visible
3. generate_hard_negatives - Hard negative mining (same_doc, cross_doc, random)
4. export_formats - Multi-format export (TRIPLETS, ARES, BEIR, RAGAS)
5. validate_dataset - Schema and distribution validation

ISO Reference: ISO 42001, ISO 25010, ISO 29119, ISO 12207
"""
