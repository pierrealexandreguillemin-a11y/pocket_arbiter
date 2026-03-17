"""Chess FR synonyms and Snowball FR stemming for BM25 query expansion.

Build-time: stem_text() stems corpus text for FTS5 indexing.
Query-time: expand_query() stems + expands query with chess synonyms.
"""

from __future__ import annotations

import re

import snowballstemmer

_stemmer = snowballstemmer.stemmer("french")

# Bidirectional chess/arbitrage FR synonyms.
# Keys and values are raw (unstemmed) — stemming applied at expansion time.
CHESS_SYNONYMS: dict[str, list[str]] = {
    "cadence": ["temps", "rythme", "controle"],
    "elo": ["classement", "rating"],
    "forfait": ["absence", "defaut"],
    "mat": ["echec et mat"],
    "pendule": ["horloge", "montre"],
    "nul": ["nulle", "partie nulle", "remise"],
    "appariement": ["pairage", "tirage"],
    "homologation": ["validation", "officialisation"],
    "departage": ["tie-break", "barrage"],
    "roque": ["grand roque", "petit roque"],
    "mutation": ["transfert", "changement de club"],
    "licence": ["inscription", "affiliation"],
    "arbitre": ["juge", "directeur de tournoi"],
    "blitz": ["parties eclair"],
    "rapide": ["parties rapides"],
    "classement": ["elo", "rating", "niveau"],
    "equipe": ["club", "formation"],
    "joueur": ["participant", "competiteur"],
    "partie": ["rencontre", "match"],
    "victoire": ["gain", "point"],
    "defaite": ["perte"],
    "abandon": ["resignation"],
    "promotion": ["transformation", "sous-promotion"],
    "zeitnot": ["pression au temps", "drapeau"],
    "drapeau": ["chute du drapeau", "temps depasse"],
}


def stem_text(text: str) -> str:
    """Stem French text using Snowball FR.

    Args:
        text: Raw French text.

    Returns:
        Space-joined stemmed words. Numbers and punctuation preserved.
    """
    if not text:
        return ""
    words = text.split()
    stemmed = []
    for word in words:
        # Preserve numbers and punctuation-only tokens
        if re.match(r"^[\d.,/\-:]+$", word):
            stemmed.append(word)
        else:
            stemmed.append(_stemmer.stemWord(word.lower()))
    return " ".join(stemmed)


def build_reverse_synonyms(
    synonyms: dict[str, list[str]],
) -> dict[str, list[str]]:
    """Build reverse lookup: for each synonym value, list its keys.

    Args:
        synonyms: Forward synonym dict (term -> [synonyms]).

    Returns:
        Reverse dict (synonym -> [terms that have it as synonym]).
    """
    reverse: dict[str, list[str]] = {}
    for key, values in synonyms.items():
        for val in values:
            reverse.setdefault(val, []).append(key)
    return reverse


# Pre-built reverse lookup
_REVERSE_SYNONYMS = build_reverse_synonyms(CHESS_SYNONYMS)


def expand_query(query: str) -> str:
    """Stem query and expand with chess synonyms.

    Args:
        query: Raw user query in French.

    Returns:
        Stemmed query with synonym terms appended.
    """
    if not query:
        return ""
    query_lower = query.lower()
    extra_terms: list[str] = []

    # Forward: if query contains a key, add its synonyms
    for term, synonyms in CHESS_SYNONYMS.items():
        if term in query_lower:
            extra_terms.extend(synonyms)

    # Reverse: if query contains a synonym value, add its key
    for synonym, keys in _REVERSE_SYNONYMS.items():
        if synonym in query_lower:
            extra_terms.extend(keys)

    # Stem everything
    stemmed_query = stem_text(query)
    if extra_terms:
        stemmed_extras = stem_text(" ".join(extra_terms))
        return f"{stemmed_query} {stemmed_extras}"
    return stemmed_query
