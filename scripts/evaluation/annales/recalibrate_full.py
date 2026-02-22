"""Comprehensive recalibration of GS classifications.

Applies ALL corrections from GO/NO-GO A->B validation in one pass:
1. answer_type: keyword overlap (answer vs chunk)
2. Q2 answer reformulation (ca25aa80)
3. 11 hard replacement questions (low kw overlap, genuine difficulty)
4. cognitive_level: pattern-based (Understand->Remember for simple recall)
5. question_type: pattern-based (procedural->factual for non-process Qs)
6. difficulty: keyword overlap (question vs chunk) + page-ref detection

Run from committed baseline (post-P2, pre-recalibration).
"""

import json
import re
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).parents[3]

DATE = "2026-02-22"


# =============================================================================
# Shared utilities
# =============================================================================


def kw_overlap(text: str, chunk: str) -> float:
    """Keyword overlap ratio between text and chunk."""
    words = set(w.lower() for w in re.findall(r"[a-zA-Z\u00e0-\u00ff]{4,}", text))
    c_words = set(w.lower() for w in re.findall(r"[a-zA-Z\u00e0-\u00ff]{4,}", chunk))
    if not words:
        return 0.0
    return len(words & c_words) / len(words)


def load_chunk_index() -> dict[str, str]:
    """Load chunk text index from corpus + P2 candidates."""
    chunks_path = PROJECT_ROOT / "corpus" / "processed" / "chunks_mode_b_fr.json"
    chunks_data = json.loads(chunks_path.read_text(encoding="utf-8"))
    idx = {c["id"]: c["text"] for c in chunks_data["chunks"]}

    p2_path = PROJECT_ROOT / "data" / "gs_generation" / "p2_candidates.json"
    if p2_path.exists():
        p2_cand = json.loads(p2_path.read_text(encoding="utf-8"))
        for tasks in p2_cand.values():
            for task in tasks:
                idx[task["chunk_id"]] = task["chunk_text"]

    return idx


# =============================================================================
# Step 1: answer_type recalibration
# =============================================================================


def recalibrate_answer_type(q: dict, chunk: str) -> str | None:
    """Return new answer_type if it should change, else None.

    Threshold 0.45: borderline cases (0.45-0.55) tend to be extractive
    per LLM judge validation (round 2 found 3 cases at 0.46-0.52 that
    the judge classified as extractive).
    """
    a_text = q["content"]["expected_answer"]
    a_ov = kw_overlap(a_text, chunk)
    current = q["classification"]["answer_type"]

    if a_ov >= 0.45 and current != "extractive":
        return "extractive"
    if a_ov < 0.45 and current != "inferential":
        return "inferential"
    return None


# =============================================================================
# Step 2: Q2 answer reformulation
# =============================================================================

Q2_ID = "gs:scratch:answerable:0189:ca25aa80"
Q2_NEW_ANSWER = (
    "Ce questionnaire recueille les antecedents familiaux de mort subite "
    "avant 50 ans, ce qui constitue un facteur de risque cardiovasculaire "
    "majeur pour la pratique sportive."
)


# =============================================================================
# Step 3: Hard replacement questions
# =============================================================================

HARD_REPLACEMENTS = [
    {
        "id": "gs:scratch:answerable:1050:e95b24f5",
        "question": (
            "Face a un conflit entre deux joueurs qui s'accusent mutuellement de "
            "tricherie en plein milieu de ronde, quelle posture et quelles actions "
            "le responsable de la salle doit-il adopter selon les principes "
            "directeurs de la fonction?"
        ),
        "answer": (
            "L'arbitre doit officier avec calme, dans un esprit de moderation, "
            "d'ouverture et de conciliation. Il joue un role pedagogique, s'assure "
            "du fair-play, et peut composer un Jury d'Appel si un appel est recevable."
        ),
        "difficulty": 0.8,
    },
    {
        "id": "gs:scratch:answerable:0544:8dc78fca",
        "question": (
            "Dans un tournoi toutes rondes a forte participation multinationale, "
            "comment le reglement prevoit-il d'eviter que les representants d'un "
            "meme pays se rencontrent trop tot dans le tableau?"
        ),
        "answer": (
            "Le tirage au sort restrictif impose un ordre : la federation avec le "
            "plus de representants tire en premier (a egalite, ordre alphabetique "
            "des codes FIDE). L'arbitre prepare des enveloppes contenant les numeros "
            "des series, garantissant la separation."
        ),
        "difficulty": 0.75,
    },
    {
        "id": "gs:scratch:answerable:1241:b32de191",
        "question": (
            "Quel impact a le delai de cloture des resultats sur la precision du "
            "rating publie mensuellement, et quelles informations individuelles "
            "figurent dans ces listes?"
        ),
        "answer": (
            "La cloture est fixee 3 jours avant publication. Chaque entree contient : "
            "titre, federation, classement, code FIDE, nombre de parties, date de "
            "naissance, sexe et coefficient K. La periode de classement definit la "
            "validite de chaque liste."
        ),
        "difficulty": 0.75,
    },
    {
        "id": "gs:scratch:answerable:1110:d2cf7cd4",
        "question": (
            "Un adversaire conteste l'eligibilite d'un membre d'equipe adverse "
            "pour des raisons de passeport. Quels mecanismes de verification le "
            "reglement impose-t-il a la structure d'accueil?"
        ),
        "answer": (
            "Le club doit justifier la nationalite dans les 15 jours suivant la "
            "notification de la reserve aupres de la direction de la competition "
            "ou la DTF. A defaut, le joueur n'est pas comptabilise dans le quota "
            "de nationalite francaise."
        ),
        "difficulty": 0.8,
    },
    {
        "id": "gs:scratch:answerable:0793:d9c23e23",
        "question": (
            "Quels mecanismes d'assistance au deplacement des pieces existent sur "
            "les interfaces numeriques, et en quoi modifient-ils l'experience par "
            "rapport a une partie sur echiquier physique?"
        ),
        "answer": (
            "Trois mecanismes : le smart move (selection de la case d'arrivee seule "
            "quand un seul coup est legal), le pre-move (programmation anticipee du "
            "coup suivant), et le conditional pre-move (sequence conditionnelle). "
            "Ces options n'existent pas sur echiquier physique."
        ),
        "difficulty": 0.75,
    },
    {
        "id": "gs:scratch:answerable:2135:2f8d44de",
        "question": (
            "En matiere de parcours de certification des officiels, quels jalons "
            "d'experience et de formation distinguent le premier echelon du second "
            "dans la hierarchie internationale?"
        ),
        "answer": (
            "Premier echelon (AF) : minimum 19 ans, un seminaire plus 3 attestations "
            "de normes. Second echelon (AI) : minimum 21 ans, 4 attestations de "
            "normes. Les deux exigent un formulaire signe par un representant de "
            "la federation."
        ),
        "difficulty": 0.8,
    },
    {
        "id": "gs:scratch:answerable:1376:f56ad7f4",
        "question": (
            "Lors du processus d'association des paires dans un groupe heterogene "
            "au systeme suisse, quels parametres numeriques limitent les combinaisons "
            "possibles et contraignent les transferts entre groupes?"
        ),
        "answer": (
            "MaxPaires est le plafond de paires realisables dans le niveau (defini "
            "en C.5). M1 est le nombre maximum de flotteurs descendants du niveau "
            "precedent pouvant etre apparies dans le niveau etudie (C.6). M0 "
            "represente les flotteurs entrants."
        ),
        "difficulty": 0.8,
    },
    {
        "id": "gs:scratch:answerable:2406:7e5874c8",
        "question": (
            "Pour un officiel souhaitant acceder au grade international, quels types "
            "d'evenements comptent pour la validation de son dossier et quels "
            "plafonds s'appliquent par categorie?"
        ),
        "answer": (
            "Les voies sont : finale nationale (2 normes max), tournoi/match officiel "
            "FIDE, tournoi international a normes, championnat rapide/blitz "
            "continental/mondial (1 norme max). Un seminaire AI avec evaluation "
            "positive est aussi requis."
        ),
        "difficulty": 0.75,
    },
    {
        "id": "gs:scratch:answerable:0155:de95f352",
        "question": (
            "Quelle distinction fondamentale le reglement etablit-il entre une "
            "situation ou plus aucune sequence de coups ne peut aboutir a un mat "
            "et l'action de retirer une piece adverse du plateau?"
        ),
        "answer": (
            "La position morte (Art. 5.2.2) designe une configuration ou aucun "
            "joueur ne peut mater par quelque suite de coups legaux, entrainant la "
            "nulle automatique. La prise (Art. 3.1) designe le deplacement d'une "
            "piece vers une case occupee par une piece adverse, qui est alors retiree."
        ),
        "difficulty": 0.7,
    },
    {
        "id": "gs:scratch:answerable:2437:bdaee54b",
        "question": (
            "Pour une rencontre par equipes impliquant un grand nombre de formations, "
            "comment le reglement definit-il la logistique materielle initiale et "
            "la constitution des alignements?"
        ),
        "answer": (
            "Le nombre d'echiquiers est calcule en multipliant le nombre d'equipes "
            "par la moitie de l'effectif (ex: 17x4=68). Les capitaines remettent "
            "la liste ordonnee. Un tirage au sort attribue une lettre par equipe. "
            "L'ordre ne peut etre permute."
        ),
        "difficulty": 0.75,
    },
    {
        "id": "gs:scratch:answerable:1892:36c49e26",
        "question": (
            "Comment le systeme de promotion/relegation du circuit jeunes par "
            "equipes articule-t-il les responsabilites entre l'echelon regional "
            "et l'echelon federal pour determiner les montants?"
        ),
        "answer": (
            "Les Ligues organisent la N3 librement et ne peuvent qualifier en N2 "
            "qu'une equipe par Zone Interdepartementale. Si elles disposent de "
            "plusieurs ZID, elles qualifient au maximum autant d'equipes. La N2 "
            "et au-dessus sont sous responsabilite de la Commission Technique "
            "Federale."
        ),
        "difficulty": 0.75,
    },
]


# =============================================================================
# Step 4: cognitive_level recalibration (TIGHTER rules)
# =============================================================================


def should_be_remember(
    q_text: str, a_text: str, answer_type: str, chunk: str, current_cl: str
) -> bool:
    """Check if cognitive level should be Remember (simple recall)."""
    if current_cl not in ("Understand",):
        return False
    if answer_type != "extractive":
        return False

    q_lower = q_text.lower()
    a_ov = kw_overlap(a_text, chunk)

    # "Quelle regle/obligation est enoncee a la page..."
    if re.search(r"quelle (r[eè]gle|obligation) est [eé]nonc[eé]e", q_lower):
        return True

    # "Qu'est-ce que [short noun]?" - simple definition lookup
    if re.search(r"qu['\u2019]est-ce qu", q_lower) and len(q_text) < 55:
        return True

    # "Que precise [document]?" - document content lookup
    if re.search(r"que pr[eé]cise", q_lower):
        return True

    # "Quels/Quelles sont les..." with very high answer overlap = just listing
    if re.search(r"quel(le)?s? sont les", q_lower) and a_ov >= 0.75:
        return True

    return False


# =============================================================================
# Step 5: question_type recalibration (TIGHTER rules)
# =============================================================================


def should_be_factual(
    q_text: str, a_text: str, answer_type: str, current_qt: str
) -> bool:
    """Check if question_type should be factual (not procedural/scenario)."""
    if current_qt not in ("procedural", "scenario"):
        return False

    q_lower = q_text.lower()

    # "Quelle regle/obligation est enoncee..."
    if re.search(r"quelle (r[eè]gle|obligation) est [eé]nonc[eé]e", q_lower):
        return True

    # "Que precise..."
    if re.search(r"que pr[eé]cise", q_lower):
        return True

    # "Que se passe-t-il..." with extractive answer = stating facts, not scenario
    if re.search(r"que se passe-t-il", q_lower) and answer_type == "extractive":
        return True

    # Simple "Qu'est-ce que" misclassified as procedural (no process words)
    if current_qt == "procedural" and re.search(r"^qu['\u2019]est-ce qu", q_lower):
        combined = (q_text + " " + a_text).lower()
        process_words = [
            "etapes",
            "procedure",
            "processus",
            "demarche",
            "comment faire",
            "comment proceder",
            "d'abord",
            "ensuite",
        ]
        if not any(w in combined for w in process_words):
            return True

    return False


# =============================================================================
# Step 6: difficulty recalibration (MODERATE rules)
# =============================================================================


def recalibrate_difficulty(q_text: str, chunk: str, current_diff: float) -> float:
    """Recalibrate difficulty based on retrieval analysis.

    Very conservative: only cap obviously easy questions (high q-chunk overlap).
    Page-number questions (95 total) are a design issue for Phase B, not
    reclassified here to avoid inflating hard count.
    """
    q_ov = kw_overlap(q_text, chunk)

    # Very high question-chunk overlap = easy retrieval (cap only)
    if q_ov >= 0.55:
        return min(current_diff, 0.35)

    return current_diff


# =============================================================================
# Main pipeline
# =============================================================================


def main() -> None:
    """Apply all recalibrations from committed baseline."""
    gs_path = PROJECT_ROOT / "tests" / "data" / "gs_scratch_v1_step1.json"
    gs = json.loads(gs_path.read_text(encoding="utf-8"))
    chunk_idx = load_chunk_index()

    stats = {"at": 0, "q2": 0, "hard": 0, "cl": 0, "qt": 0, "diff": 0}

    # --- Step 1: answer_type ---
    for q in gs["questions"]:
        if q["content"]["is_impossible"]:
            continue
        chunk = chunk_idx.get(q["provenance"]["chunk_id"], "")
        if not chunk:
            continue
        new_at = recalibrate_answer_type(q, chunk)
        if new_at:
            old = q["classification"]["answer_type"]
            q["classification"]["answer_type"] = new_at
            q["audit"]["history"] += f" | [RECALIB-AT] {old}->{new_at} on {DATE}"
            stats["at"] += 1

    # --- Step 2: Q2 answer fix ---
    for q in gs["questions"]:
        if q["id"] == Q2_ID:
            q["content"]["expected_answer"] = Q2_NEW_ANSWER
            q["mcq"]["correct_answer"] = Q2_NEW_ANSWER
            q["mcq"]["original_answer"] = Q2_NEW_ANSWER
            q["audit"]["history"] += f" | [RECALIB-Q2] answer reformulated on {DATE}"
            stats["q2"] = 1

    # --- Step 3: Hard replacements ---
    for repl in HARD_REPLACEMENTS:
        q = next(q for q in gs["questions"] if q["id"] == repl["id"])
        chunk = chunk_idx.get(q["provenance"]["chunk_id"], "")

        q["content"]["question"] = repl["question"]
        q["content"]["expected_answer"] = repl["answer"]
        q["classification"]["difficulty"] = repl["difficulty"]
        q["mcq"]["original_question"] = repl["question"]
        q["mcq"]["correct_answer"] = repl["answer"]
        q["mcq"]["original_answer"] = repl["answer"]

        # Recalculate answer_type for replacement
        ans_ov = kw_overlap(repl["answer"], chunk)
        q["classification"]["answer_type"] = (
            "extractive" if ans_ov >= 0.55 else "inferential"
        )

        q_ov = kw_overlap(repl["question"], chunk)
        q["audit"]["history"] += (
            f" | [RECALIB-HARD] question+answer rewritten "
            f"(q_ov={q_ov:.2f}, ans_ov={ans_ov:.2f}) on {DATE}"
        )
        stats["hard"] += 1

    # --- Steps 4-6: cognitive_level, question_type, difficulty ---
    for q in gs["questions"]:
        if q["content"]["is_impossible"]:
            continue
        chunk = chunk_idx.get(q["provenance"]["chunk_id"], "")
        if not chunk:
            continue

        q_text = q["content"]["question"]
        a_text = q["content"]["expected_answer"]
        at = q["classification"]["answer_type"]
        cl = q["classification"]["cognitive_level"]
        qt = q["classification"]["question_type"]
        diff = q["classification"]["difficulty"]

        changes = []

        if should_be_remember(q_text, a_text, at, chunk, cl):
            q["classification"]["cognitive_level"] = "Remember"
            changes.append(f"CL:{cl}->Remember")
            stats["cl"] += 1

        if should_be_factual(q_text, a_text, at, qt):
            q["classification"]["question_type"] = "factual"
            changes.append(f"QT:{qt}->factual")
            stats["qt"] += 1

        new_diff = recalibrate_difficulty(q_text, chunk, diff)
        if abs(new_diff - diff) > 0.01:
            q["classification"]["difficulty"] = round(new_diff, 2)
            changes.append(f"DIFF:{diff:.2f}->{new_diff:.2f}")
            stats["diff"] += 1

        if changes:
            q["audit"]["history"] += f" | [RECALIB-FULL] {', '.join(changes)} on {DATE}"

    # --- Write ---
    gs_path.write_text(json.dumps(gs, ensure_ascii=False, indent=2), encoding="utf-8")

    # --- Report ---
    answerable = [q for q in gs["questions"] if not q["content"]["is_impossible"]]
    cl_dist = Counter(q["classification"]["cognitive_level"] for q in answerable)
    qt_dist = Counter(q["classification"]["question_type"] for q in answerable)
    at_dist = Counter(q["classification"]["answer_type"] for q in answerable)
    hard = sum(1 for q in answerable if q["classification"]["difficulty"] >= 0.7)

    print("=== Recalibration Report ===")
    print(f"  answer_type changes:     {stats['at']}")
    print(f"  Q2 answer fix:           {stats['q2']}")
    print(f"  hard replacements:       {stats['hard']}")
    print(f"  cognitive_level changes: {stats['cl']}")
    print(f"  question_type changes:   {stats['qt']}")
    print(f"  difficulty changes:      {stats['diff']}")
    print("\nDistributions:")
    print(f"  cognitive_level: {dict(sorted(cl_dist.items()))}")
    print(f"  question_type:   {dict(sorted(qt_dist.items()))}")
    print(f"  answer_type:     {dict(sorted(at_dist.items()))}")
    print(f"\nHard answerable: {hard}/{len(answerable)} = {hard/len(answerable):.3f}")
    print(f"A-G2 (>=10%): {'PASS' if hard/len(answerable) >= 0.10 else 'FAIL'}")
    print(f"A-G3 (4 CL): {'PASS' if len(cl_dist) >= 4 else 'FAIL'}")
    print(f"Total: {len(gs['questions'])}")


if __name__ == "__main__":
    main()
