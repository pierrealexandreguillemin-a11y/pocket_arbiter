"""RAG system prompt for Pocket Arbiter generation model.

Used at inference time AND for eval. Single source of truth.
"""

SYSTEM_PROMPT = (
    "Tu es un assistant pour arbitres d'echecs. Tu reponds aux "
    "questions en te basant UNIQUEMENT sur le contexte fourni "
    "(extraits des reglements FFE/FIDE).\n\n"
    "REGLES:\n"
    "- Cite TOUJOURS le document source et l'article/section.\n"
    "- Si le contexte ne contient pas la reponse, dis "
    "'Information non trouvee dans les extraits fournis.'\n"
    "- Ne reponds JAMAIS avec des informations hors contexte.\n"
    "- Reponds en francais.\n"
    "- Sois concis et actionnable."
)


def build_rag_prompt(question: str, context: str) -> list[dict]:
    """Build chat messages for RAG inference.

    Args:
        question: The user question in French.
        context: Retrieved chunk text(s) to ground the response.

    Returns:
        List of chat message dicts (role/content) for the model.
    """
    return [
        {
            "role": "user",
            "content": (
                f"{SYSTEM_PROMPT}\n\n"
                f"Contexte:\n{context}\n\n"
                f"Question: {question}"
            ),
        },
    ]
