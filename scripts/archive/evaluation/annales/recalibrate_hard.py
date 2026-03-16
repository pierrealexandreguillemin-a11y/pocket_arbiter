"""Recalibrate 11 P2 questions to genuinely hard difficulty.

Replaces question+answer with formulations that have low keyword overlap
with the source chunk, making them genuinely hard for retrieval.
"""

import json
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).parents[3]


def kw_overlap(text: str, chunk: str) -> float:
    """Keyword overlap ratio between text and chunk."""
    words = set(w.lower() for w in re.findall(r"[a-zA-Z\u00e0-\u00ff]{4,}", text))
    c_words = set(w.lower() for w in re.findall(r"[a-zA-Z\u00e0-\u00ff]{4,}", chunk))
    if not words:
        return 0.0
    return len(words & c_words) / len(words)


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


def main() -> None:
    """Apply hard replacements to GS."""
    gs_path = PROJECT_ROOT / "tests" / "data" / "gs_scratch_v1_step1.json"
    gs = json.loads(gs_path.read_text(encoding="utf-8"))

    # Build chunk index
    chunks_path = PROJECT_ROOT / "corpus" / "processed" / "chunks_mode_b_fr.json"
    chunks_data = json.loads(chunks_path.read_text(encoding="utf-8"))
    chunk_idx = {c["id"]: c["text"] for c in chunks_data["chunks"]}

    p2_path = PROJECT_ROOT / "data" / "gs_generation" / "p2_candidates.json"
    p2_cand = json.loads(p2_path.read_text(encoding="utf-8"))
    for profile_tasks in p2_cand.values():
        for task in profile_tasks:
            chunk_idx[task["chunk_id"]] = task["chunk_text"]

    replaced = 0
    for repl in HARD_REPLACEMENTS:
        q = next(q for q in gs["questions"] if q["id"] == repl["id"])
        chunk = chunk_idx.get(q["provenance"]["chunk_id"], "")

        q["content"]["question"] = repl["question"]
        q["content"]["expected_answer"] = repl["answer"]
        q["classification"]["difficulty"] = repl["difficulty"]
        q["mcq"]["original_question"] = repl["question"]
        q["mcq"]["correct_answer"] = repl["answer"]
        q["mcq"]["original_answer"] = repl["answer"]

        # Recalculate answer_type
        ans_ov = kw_overlap(repl["answer"], chunk)
        q["classification"]["answer_type"] = (
            "extractive" if ans_ov >= 0.55 else "inferential"
        )

        q_ov = kw_overlap(repl["question"], chunk)
        q["audit"]["history"] += (
            f" | [RECALIB-HARD] question+answer rewritten for genuine difficulty "
            f"(q_overlap={q_ov:.2f}, ans_overlap={ans_ov:.2f}) on 2026-02-21"
        )

        replaced += 1
        print(
            f"{repl['id'][:45]}: diff={repl['difficulty']} "
            f"q_ov={q_ov:.2f} ans_ov={ans_ov:.2f} "
            f"at={q['classification']['answer_type']}"
        )

    gs_path.write_text(json.dumps(gs, ensure_ascii=False, indent=2), encoding="utf-8")

    answerable = [q for q in gs["questions"] if not q["content"]["is_impossible"]]
    hard = sum(1 for q in answerable if q["classification"]["difficulty"] >= 0.7)
    print(f"\nReplaced: {replaced}")
    print(f"Hard answerable: {hard}/{len(answerable)} = {hard/len(answerable):.3f}")
    print(f"A-G2 (>=10%): {'PASS' if hard/len(answerable) >= 0.10 else 'FAIL'}")
    print(f"Total questions: {len(gs['questions'])}")


if __name__ == "__main__":
    main()
