"""
Pipeline de traitement des PDF - Pocket Arbiter

Ce module contient les scripts d'extraction et chunking des PDF
pour le pipeline RAG.

Modules:
    extract_pdf: Extraction texte depuis PDF via PyMuPDF
    chunker: Segmentation en chunks de 256 tokens
    utils: Utilitaires communs

Usage:
    python -m scripts.pipeline.extract_pdf --input corpus/fr --output corpus/processed
    python -m scripts.pipeline.chunker --input corpus/processed --output chunks_fr.json

ISO Reference: ISO/IEC 12207 - Development Process
"""

__version__ = "0.1.0"
__author__ = "Pocket Arbiter Team"
