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
#
# Two categories:
# A) Intra-corpus: both sides verified present in corpus_v2_fr.db
# B) Colloquial->corpus: key = terrain/oral phrasing, values = corpus terms
#    Key may have 0 corpus hits (that's the point: user says it, corpus doesn't)
#
# Audit: 2026-03-17, SELECT text FROM children fulltext scan on 1253 chunks.
CHESS_SYNONYMS: dict[str, list[str]] = {
    # --- Temps et cadences ---
    "cadence": ["temps", "rythme", "pendule"],  # A: all in corpus
    "pendule": ["horloge", "montre"],  # A: 261, 6, 34
    "rapide": ["parties rapides", "cadence rapide"],  # A: 113, 10, 17
    "blitz": ["cadence rapide"],  # A: 58, 17
    "drapeau": ["chute du drapeau", "temps"],  # A: 57, 9, 241
    "fischer": ["cadence fischer"],  # A: 19, 10
    # B: colloquial time expressions -> corpus terms
    "temps ecoule": ["chute du drapeau", "drapeau"],  # B: 0->9,57
    "temps depasse": ["chute du drapeau", "drapeau"],  # B: 0->9,57
    # --- Scores et classement ---
    "elo": ["classement", "rating"],  # A: 477, 571, 10
    "classement": ["elo", "rating", "niveau"],  # A: 571, 477, 10, 310
    "departage": ["tie-break", "barrage"],  # A: 94, 2, 15
    "victoire": ["gain", "score"],  # A: 42, 37, 249
    "defaite": ["perte"],  # A: 49, 49
    "nul": ["nulle", "partie nulle", "remise"],  # A: 244, 161, 30, 48
    "score": ["resultat", "classement"],  # A: 249, 2, 571
    # B: colloquial result terms
    "match nul": ["partie nulle", "nulle"],  # B: 28->30,161
    "gagner": ["victoire", "gain"],  # B: 21->42,37
    "perdre": ["defaite", "perte"],  # B: 9->49,49
    # --- Coups et regles de jeu ---
    "roque": ["grand roque", "petit roque", "roquer"],  # A: 61, 5, 5, 12
    "prise en passant": ["en passant"],  # A: 4, 10
    "mat": ["mater"],  # A: 1067, 14
    "pat": ["position morte", "nulle"],  # A: 104, 4, 161
    "promotion": ["promouvoir"],  # A: 34, 9
    # B: colloquial checkmate terms
    "echec et mat": ["mat", "mater"],  # B: 0->1067,14
    # B: colloquial piece/move terms
    "piece touchee": ["touche", "article 4"],  # B: 0->11,36
    "touche bouge": ["touche", "article 4"],  # B: 0->11,36
    "materiel insuffisant": ["insuffisance", "position morte"],  # B: 0->2,4
    "triple repetition": ["trois fois", "position"],  # B: 0->6,408
    "cinquante coups": ["50 coups", "75 coups"],  # B: 0->3,4
    # --- Resultat et fin de partie ---
    "forfait": ["absence", "retard"],  # A: 289, 47, 52
    "abandon": ["abandonner", "forfait"],  # A: 26, 7, 289
    "ajournement": ["ajourne", "report"],  # A: 11, 14, 13
    # --- Organisation et tournoi ---
    "appariement": ["tirage", "appariements"],  # A: 374, 43, 175
    "homologation": ["validation", "homologuer"],  # A: 76, 19, 5
    "ronde": ["tour"],  # A: 755, 1054
    "poule": ["groupe"],  # A: 15, 344
    "tableau": ["grille"],  # A: 215, 12
    "inscription": ["engagement"],  # A: 70, 46
    "suisse": ["toutes rondes"],  # A: 96, 58 (tournament systems)
    "open": ["tournoi"],  # A: 60, 575
    "coupe": ["coupe de france", "championnat"],  # A: 115, 37, 276
    "interclubs": ["championnat", "equipe"],  # A: 29, 276, 4
    "feuille de partie": ["feuille", "notation"],  # A: 44, 162, 30
    # B: colloquial orga terms
    "feuille de match": ["feuille de partie", "feuille"],  # B: 39->44,162
    # --- Personnes et roles ---
    "arbitre": ["juge", "directeur"],  # A: 1442, 15, 242
    "joueur": ["participant", "adversaire"],  # A: 1755, 130, 446
    "equipe": ["club", "formation"],  # A: 4, 492, 209
    "capitaine": ["chef"],  # A: 121, 96
    "organisateur": ["directeur"],  # A: 136, 242
    # --- Administration ---
    "mutation": ["transfert", "changement de club"],  # A: 3, 8, 2
    "licence": ["inscription", "affiliation"],  # A: 196, 70, 18
    "qualification": ["norme", "titre"],  # A: 83, 162, 320
    # --- Categories jeunes (FFE noms <-> codes age) ---
    "pupille": ["u10"],  # A: 2, 27
    "poussin": ["u8"],  # A: 5, 21
    "benjamin": ["u12"],  # A: 2, 13
    "minime": ["u14"],  # A: 4, 15
    "cadet": ["u16"],  # A: 10, 16
    "junior": ["u18"],  # A: 11, 11
    # --- Sanctions et discipline ---
    "sanction": ["avertissement", "exclusion", "suspension"],  # A: 192,16,34,7
    "exclusion": ["disqualification"],  # A: 34, 5
    "triche": ["fraude", "tricher"],  # A: 106, 11, 56
    "plainte": ["contestation", "recours"],  # A: 49, 2, 12
    "appel": ["faire appel", "jury"],  # A: 240, 18, 22
    # B: colloquial discipline terms
    "reclamation": ["plainte", "contestation", "recours"],  # B: 0->49,2,12
    "protestation": ["plainte", "contestation"],  # B: 0->49,2
    "tricheur": ["triche", "fraude"],  # B: 0->106,11
    # B: electronic devices (frequent terrain question)
    "telephone": ["portable", "electronique"],  # B: 0->4,66
    "smartphone": ["portable", "electronique"],  # B: 0->4,66
    # --- Partie et rencontre ---
    "partie": ["rencontre", "match"],  # A: 1199, 193, 474
    # B: colloquial offer draw
    "proposer nulle": ["offre de nulle", "nulle par accord"],  # B: 5->2,3
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
