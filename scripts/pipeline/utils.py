"""
Utilitaires communs - Pipeline Pocket Arbiter

Ce module contient les fonctions utilitaires partagees
entre les modules du pipeline.

ISO Reference: ISO/IEC 12207 - Reusability
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def load_json(file_path: Path) -> dict:
    """
    Charge un fichier JSON.

    Args:
        file_path: Chemin vers le fichier JSON.

    Returns:
        Contenu du fichier JSON.

    Raises:
        FileNotFoundError: Si le fichier n'existe pas.
        json.JSONDecodeError: Si le JSON est invalide.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, file_path: Path, indent: int = 2) -> None:
    """
    Sauvegarde des donnees en JSON.

    Args:
        data: Donnees a sauvegarder.
        file_path: Chemin du fichier de sortie.
        indent: Indentation JSON (default 2).
    """
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)
    logger.info(f"Saved: {file_path}")


def get_timestamp() -> str:
    """
    Retourne le timestamp actuel au format ISO 8601.

    Returns:
        Timestamp ISO 8601 (ex: "2026-01-14T10:30:00").
    """
    return datetime.now().isoformat(timespec="seconds")


def get_date() -> str:
    """
    Retourne la date actuelle au format ISO.

    Returns:
        Date ISO (ex: "2026-01-14").
    """
    return datetime.now().strftime("%Y-%m-%d")


def normalize_text(text: str) -> str:
    """
    Normalise le texte pour le traitement.

    - Supprime les espaces multiples
    - Normalise les sauts de ligne
    - Supprime les caracteres de controle

    Args:
        text: Texte brut a normaliser.

    Returns:
        Texte normalise.
    """
    import re

    # Normaliser les sauts de ligne
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Supprimer les caracteres de controle (sauf newline et tab)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

    # Normaliser les espaces multiples
    text = re.sub(r"[ \t]+", " ", text)

    # Normaliser les lignes vides multiples
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def list_pdf_files(directory: Path) -> list[Path]:
    """
    Liste tous les fichiers PDF dans un dossier (recursif).

    Args:
        directory: Dossier a scanner.

    Returns:
        Liste des chemins vers les fichiers PDF.
    """
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    return sorted(directory.rglob("*.pdf"))


def _validate_chunk_id(chunk_id: str) -> list[str]:
    """Validate chunk ID format."""
    import re

    if not re.match(r"^(FR|INTL)-\d{3}-\d{3}-\d{2}$", chunk_id):
        return [f"Invalid chunk ID format: {chunk_id}"]
    return []


def _validate_chunk_metadata(metadata: dict) -> list[str]:
    """Validate chunk metadata fields."""
    errors = []
    meta_required = ["corpus", "extraction_date", "version"]
    for field in meta_required:
        if field not in metadata:
            errors.append(f"Missing metadata field: {field}")
    return errors


def validate_chunk_schema(chunk: dict) -> list[str]:
    """
    Valide un chunk contre le schema attendu.

    Args:
        chunk: Chunk a valider.

    Returns:
        Liste des erreurs de validation (vide si valide).
    """
    errors = []

    required_fields = ["id", "text", "source", "page", "tokens", "metadata"]
    for field in required_fields:
        if field not in chunk:
            errors.append(f"Missing required field: {field}")

    if "id" in chunk:
        errors.extend(_validate_chunk_id(chunk["id"]))

    if "text" in chunk and len(chunk["text"]) < 50:
        errors.append(f"Text too short: {len(chunk['text'])} chars")

    if "tokens" in chunk and chunk["tokens"] > 512:
        errors.append(f"Too many tokens: {chunk['tokens']}")

    if "metadata" in chunk:
        errors.extend(_validate_chunk_metadata(chunk["metadata"]))

    return errors
