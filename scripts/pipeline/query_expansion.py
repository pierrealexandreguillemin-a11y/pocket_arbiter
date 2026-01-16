"""
Query Expansion - Pocket Arbiter

Expansion des requetes avec synonymes et termes chess FR.

ISO Reference:
    - ISO/IEC 25010 - Performance efficiency (Recall >= 80%)
    - ISO/IEC 42001 - AI traceability (domain-specific vocabulary)
"""

import re
import unicodedata
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


def normalize_text(text: str) -> str:
    """Normalise le texte (accents, casse)."""
    # Decompose accents then remove combining characters
    normalized = unicodedata.normalize("NFD", text.lower())
    without_accents = "".join(c for c in normalized if unicodedata.category(c) != "Mn")
    return without_accents


# =============================================================================
# Chess FR Synonyms Dictionary
# =============================================================================

# Synonymes et termes associes pour le vocabulaire echecs FR/FFE
# Les cles sont normalisees (sans accents) pour le matching
CHESS_SYNONYMS: dict[str, list[str]] = {
    # Temps et cadences
    "temps": ["pendule", "horloge", "chrono", "cadence", "delai"],
    "drapeau": ["chute", "temps depasse", "pendule tombee"],
    "blitz": ["rapide", "eclair", "cadence rapide", "5 minutes"],
    "rapide": ["semi-rapide", "cadence rapide", "15 minutes"],
    "cadence": ["temps de reflexion", "controle du temps"],
    # Regles de jeu
    "toucher": ["toucher-jouer", "piece touchee", "j'adoube"],
    "roque": ["petit roque", "grand roque", "O-O", "O-O-O"],
    "en passant": ["prise en passant", "capture en passant"],
    "promotion": ["promotion du pion", "transformer", "dame"],
    "mat": ["echec et mat", "mater", "matage"],
    "pat": ["position de pat", "partie nulle", "blocage roi"],
    "nulle": ["partie nulle", "match nul", "egalite"],
    "repetition": ["triple repetition", "position repetee"],
    "50 coups": ["regle des 50 coups", "cinquante coups"],
    # Arbitrage
    "arbitre": ["directeur de tournoi", "juge", "officiel"],
    "reclamation": ["contestation", "plainte", "appel", "litige"],
    "appel": ["recours", "contestation", "commission d'appel"],
    "sanction": ["penalite", "avertissement", "exclusion"],
    "forfait": ["defaut", "absence", "perte par forfait"],
    "coup illegal": ["coup irregulier", "mouvement illegal"],
    "temps depasse": ["drapeau", "chute", "pendule tombee"],
    "depasse": ["drapeau", "chute du drapeau"],
    # Competition
    "departage": ["tie-break", "egalite", "Buchholz", "Sonneborn-Berger"],
    "buchholz": ["departage", "score buchholz", "tie-break"],
    "egalite": ["tie-break", "departage", "ex aequo"],
    "suisse": ["systeme suisse", "appariement suisse"],
    "appariement": ["pairage", "couplage", "match"],
    # Materiel et organisation
    "materiel": ["equipement", "fournitures", "pieces", "echiquier"],
    "salle": ["lieu", "local", "espace de jeu"],
    "conditions": ["exigences", "prerequis", "specifications"],
    # Notation
    "notation": ["feuille de partie", "enregistrement", "transcription"],
    "algebrique": ["notation algebrique", "notation standard"],
    # Comportement
    "triche": ["anti-triche", "fraude", "tricherie", "detection"],
    "telephone": ["portable", "appareil electronique", "smartphone"],
    "electronique": ["appareil electronique", "dispositif"],
    # Documents FFE
    "championnat": ["competition", "tournoi officiel"],
    "federal": ["FFE", "federation", "national"],
    "jeunes": ["juniors", "minimes", "cadets", "benjamins"],
}

# Termes techniques qui ne doivent pas etre expandes
TECHNICAL_TERMS = {
    "article",
    "section",
    "annexe",
    "chapitre",
    "page",
    "alinea",
    "paragraphe",
}


def expand_query(
    query: str,
    max_expansions: int = 3,
    include_original: bool = True,
) -> str:
    """
    Expande une requete avec des synonymes chess FR.

    Args:
        query: Requete originale.
        max_expansions: Nombre max de synonymes par terme.
        include_original: Inclure le terme original dans l'expansion.

    Returns:
        Requete expandue avec synonymes.

    Example:
        >>> expand_query("Comment se deroule une reclamation pour temps depasse ?")
        "Comment se deroule une reclamation pour temps depasse ? contestation plainte drapeau"
    """
    query_normalized = normalize_text(query)
    expansions = []

    for term, synonyms in CHESS_SYNONYMS.items():
        # Verifier si le terme est dans la query (avec word boundaries)
        # Utilise regex pour eviter faux positifs (ex: "mat" dans "materiel")
        pattern = r"\b" + re.escape(term) + r"\b"
        if re.search(pattern, query_normalized):
            # Ajouter les synonymes (limites a max_expansions)
            for syn in synonyms[:max_expansions]:
                syn_normalized = normalize_text(syn)
                if syn_normalized not in query_normalized:
                    expansions.append(syn)

    if not expansions:
        return query

    # Construire la requete expandue
    if include_original:
        expanded = f"{query} {' '.join(expansions)}"
    else:
        expanded = " ".join(expansions)

    return expanded


def expand_query_bm25(
    query: str,
    max_expansions: int = 2,
) -> str:
    """
    Expande une requete pour BM25 (keywords only).

    Pour BM25, on veut les synonymes sans la question complete.

    Args:
        query: Requete originale.
        max_expansions: Nombre max de synonymes par terme.

    Returns:
        Keywords expandes pour BM25.

    Example:
        >>> expand_query_bm25("Regles du blitz ?")
        "blitz rapide eclair"
    """
    query_normalized = normalize_text(query)
    keywords = []

    for term, synonyms in CHESS_SYNONYMS.items():
        # Verifier avec word boundaries
        pattern = r"\b" + re.escape(term) + r"\b"
        if re.search(pattern, query_normalized):
            keywords.append(term)
            keywords.extend(synonyms[:max_expansions])

    # Deduplicate while preserving order
    seen = set()
    unique_keywords = []
    for kw in keywords:
        kw_normalized = normalize_text(kw)
        if kw_normalized not in seen:
            seen.add(kw_normalized)
            unique_keywords.append(kw)

    return " ".join(unique_keywords)


def get_query_keywords(query: str) -> list[str]:
    """
    Extrait les mots-cles d'une query.

    Args:
        query: Requete utilisateur.

    Returns:
        Liste de mots-cles extraits.
    """
    # Retirer ponctuation et mots vides
    stopwords_fr = {
        "le",
        "la",
        "les",
        "un",
        "une",
        "des",
        "de",
        "du",
        "et",
        "ou",
        "que",
        "qui",
        "quoi",
        "quel",
        "quelle",
        "quelles",
        "quels",
        "comment",
        "pourquoi",
        "quand",
        "est",
        "sont",
        "a",
        "au",
        "aux",
        "ce",
        "cette",
        "ces",
        "se",
        "en",
        "pour",
        "sur",
        "par",
        "avec",
        "dans",
        "il",
        "elle",
        "ils",
        "elles",
        "on",
        "nous",
        "vous",
        "je",
        "tu",
        "mon",
        "ma",
        "mes",
        "ton",
        "ta",
        "tes",
        "son",
        "sa",
        "ses",
        "notre",
        "votre",
        "leur",
        "leurs",
        "ne",
        "pas",
        "plus",
        "faire",
    }

    # Nettoyer la query
    clean = re.sub(r"[^\w\s]", " ", query.lower())
    words = clean.split()

    # Filtrer stopwords et mots courts
    keywords = [w for w in words if w not in stopwords_fr and len(w) > 2]

    return keywords


def expand_for_embedding(
    query: str,
    model: "SentenceTransformer | None" = None,
) -> str:
    """
    Expande une query pour l'embedding semantique.

    Ajoute contexte et synonymes pour ameliorer le matching semantique.

    Args:
        query: Requete originale.
        model: Modele d'embedding (non utilise actuellement).

    Returns:
        Query expandue pour embedding.
    """
    # Expansion de base avec synonymes
    expanded = expand_query(query, max_expansions=2)

    # Ajouter prefixe contextuel pour EmbeddingGemma
    # (le modele utilise des prompts specifiques)
    context_prefix = "Reglement echecs FFE FIDE: "

    return f"{context_prefix}{expanded}"


# =============================================================================
# CLI for testing
# =============================================================================


def main():
    """CLI pour tester l'expansion de queries."""
    import argparse

    parser = argparse.ArgumentParser(description="Test query expansion")
    parser.add_argument("query", nargs="?", help="Query to expand")
    parser.add_argument("--all", action="store_true", help="Test all failing queries")

    args = parser.parse_args()

    if args.all:
        # Test sur les questions faibles
        failing_queries = [
            "Comment se déroule une réclamation pour temps dépassé ?",
            "Comment gérer une réclamation d'un joueur ?",
            "Quelles sont les règles du blitz ?",
            "Comment se déroule un départage ?",
            "Quelles sont les conditions matérielles minimales pour un tournoi ?",
        ]

        print("=== Query Expansion Test ===\n")
        for q in failing_queries:
            print(f"Original: {q}")
            print(f"Expanded: {expand_query(q)}")
            print(f"BM25:     {expand_query_bm25(q)}")
            print()
    elif args.query:
        print(f"Original: {args.query}")
        print(f"Expanded: {expand_query(args.query)}")
        print(f"BM25:     {expand_query_bm25(args.query)}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
