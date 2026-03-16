"""
Export Serialization - Pocket Arbiter

Fonctions de serialisation pour embeddings SQLite.

ISO Reference:
    - ISO/IEC 12207 S7.3.3 - Implementation
"""

import numpy as np

# --- Constants ---

EMBEDDING_DTYPE = np.float32


def embedding_to_blob(embedding: np.ndarray) -> bytes:
    """
    Convertit un embedding numpy en BLOB SQLite.

    Args:
        embedding: Vecteur numpy float32 de dimension D.

    Returns:
        Representation binaire (bytes) du vecteur.

    Raises:
        ValueError: Si l'embedding n'est pas 1D.

    Example:
        >>> emb = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        >>> blob = embedding_to_blob(emb)
        >>> len(blob)
        12
    """
    if embedding.ndim != 1:
        raise ValueError(f"Embedding must be 1D, got shape {embedding.shape}")

    return embedding.astype(EMBEDDING_DTYPE).tobytes()


def blob_to_embedding(blob: bytes, dim: int) -> np.ndarray:
    """
    Convertit un BLOB SQLite en embedding numpy.

    Args:
        blob: Bytes du BLOB SQLite.
        dim: Dimension attendue de l'embedding.

    Returns:
        Vecteur numpy float32.

    Raises:
        ValueError: Si la taille du blob ne correspond pas a dim.

    Example:
        >>> blob = struct.pack('3f', 0.1, 0.2, 0.3)
        >>> emb = blob_to_embedding(blob, 3)
        >>> emb.shape
        (3,)
    """
    expected_size = dim * 4  # float32 = 4 bytes
    if len(blob) != expected_size:
        raise ValueError(f"Blob size {len(blob)} != expected {expected_size}")

    return np.frombuffer(blob, dtype=EMBEDDING_DTYPE)
