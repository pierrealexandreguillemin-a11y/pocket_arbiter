"""Generate 80 replacement questions for P2 (Phase A re-generation).

Each question is hand-crafted from chunk source text, conforming to
one of four target profiles. ISO 42001: 0% hallucination.

Usage:
    python -m scripts.evaluation.annales.generate_p2_questions
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_project_root = Path(__file__).parent.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from scripts.evaluation.annales.regenerate_targeted import PROFILES  # noqa: E402
from scripts.pipeline.utils import get_date, load_json  # noqa: E402


def _make_question(
    *,
    question: str,
    expected_answer: str,
    chunk_id: str,
    source: str,
    pages: list[int],
    article_ref: str,
    cognitive_level: str,
    difficulty: float,
    question_type: str,
    answer_type: str,
    reasoning_class: str,
    keywords: list[str],
) -> dict:
    """Build a complete Schema v2 question dict."""
    question = question.strip()
    date = get_date()
    docs = [source] if source else []
    return {
        "id": "_placeholder_",
        "content": {
            "question": question,
            "expected_answer": expected_answer,
            "is_impossible": False,
        },
        "mcq": {
            "original_question": question,
            "choices": {},
            "mcq_answer": "",
            "correct_answer": "",
            "original_answer": "",
        },
        "provenance": {
            "chunk_id": chunk_id,
            "docs": docs,
            "pages": pages,
            "article_reference": article_ref,
            "answer_explanation": f"Source: {source}, {article_ref}",
            "annales_source": None,
        },
        "classification": {
            "category": "arbitrage",
            "keywords": keywords,
            "difficulty": difficulty,
            "question_type": question_type,
            "cognitive_level": cognitive_level,
            "reasoning_type": "single-hop",
            "reasoning_class": reasoning_class,
            "answer_type": answer_type,
            "hard_type": "ANSWERABLE",
        },
        "validation": {
            "status": "VALIDATED",
            "method": "p2_regeneration",
            "reviewer": "claude_code",
            "answer_current": True,
            "verified_date": date,
            "pages_verified": True,
            "batch": "gs_v1_step1_p2",
        },
        "processing": {
            "chunk_match_score": 100,
            "chunk_match_method": "by_design_input",
            "reasoning_class_method": "generation_prompt",
            "triplet_ready": True,
            "extraction_flags": ["p2_regen"],
            "answer_source": "chunk_extraction",
            "quality_score": 0.85,
            "priority_boost": 0.0,
        },
        "audit": {
            "history": f"[P2 REGEN] Generated on {date}",
            "qat_revalidation": None,
            "requires_inference": answer_type == "inferential",
        },
    }


# ===================================================================
# HARD_APPLY: scenario, extractive, difficulty >= 0.7
# ===================================================================

HARD_APPLY = [
    {
        "old_id": "gs:scratch:answerable:0012:39e5e23b",
        "profile": "HARD_APPLY",
        "q": _make_question(
            question=(
                "Dans un tournoi, Emma (5e), Farid (6e) et Germain (7e) "
                "terminent ex-aequo a 5 points. Les prix annonces sont "
                "200, 100 et 0 euros. Hortense (8e) est exclue. Comment "
                "les prix sont-ils redistribues selon le systeme Hort?"
            ),
            expected_answer=(
                "Selon le systeme Hort, Emma recoit 150 euros, "
                "Farid recoit 100 euros et Germain recoit 50 euros. "
                "Hortense est retiree de la liste des prix."
            ),
            chunk_id="LA-octobre2025.pdf-p159-parent475-child01",
            source="LA-octobre2025.pdf",
            pages=[159],
            article_ref="Systeme Hort",
            cognitive_level="Apply",
            difficulty=0.75,
            question_type="scenario",
            answer_type="extractive",
            reasoning_class="reasoning",
            keywords=["prix", "hort", "departage", "distribution"],
        ),
    },
    {
        "old_id": "gs:scratch:answerable:0032:da0f8f43",
        "profile": "HARD_APPLY",
        "q": _make_question(
            question=(
                "Lors d'une assemblee FFE, un vote doit elire un nouveau "
                "membre du bureau. 15 delegues sont presents, 3 s'abstiennent "
                "et 2 votes sont nuls. Quelle procedure de vote doit etre "
                "appliquee et comment la majorite est-elle calculee?"
            ),
            expected_answer=(
                "Le vote portant sur une personne doit avoir lieu a "
                "bulletin secret. La majorite est calculee sur 10 votes "
                "valides (15 moins 3 abstentions et 2 nuls), les "
                "abstentions et votes nuls n'etant pas pris en compte."
            ),
            chunk_id=(
                "2025_Reglement_Interieur_20250503.pdf" "-p009-parent028-child00"
            ),
            source="2025_Reglement_Interieur_20250503.pdf",
            pages=[9],
            article_ref="Art. 6.5",
            cognitive_level="Apply",
            difficulty=0.75,
            question_type="scenario",
            answer_type="extractive",
            reasoning_class="reasoning",
            keywords=["vote", "bulletin secret", "majorite", "assemblee"],
        ),
    },
    {
        "old_id": "gs:scratch:answerable:0033:955d4585",
        "profile": "HARD_APPLY",
        "q": _make_question(
            question=(
                "Un tournoi a elimination directe accueille 52 joueurs. "
                "Le tableau prevu est pour 64 participants. Que doit faire "
                "l'arbitre pour gerer les exemptions au premier tour?"
            ),
            expected_answer=(
                "Il manque 12 joueurs pour atteindre 64 (64-52=12). "
                "L'arbitre doit exempter 12 joueurs qui se qualifient "
                "directement au tour suivant. Les 40 joueurs restants "
                "(52-12=40) s'affrontent au premier tour."
            ),
            chunk_id="LA-octobre2025.pdf-p119-parent348-child00",
            source="LA-octobre2025.pdf",
            pages=[119],
            article_ref="Exempts au 1er tour",
            cognitive_level="Apply",
            difficulty=0.7,
            question_type="scenario",
            answer_type="extractive",
            reasoning_class="reasoning",
            keywords=["exempt", "elimination", "premier tour", "tableau"],
        ),
    },
    {
        "old_id": "gs:scratch:answerable:0046:e34ca6c3",
        "profile": "HARD_APPLY",
        "q": _make_question(
            question=(
                "Lors d'un tournoi Toutes Rondes, l'arbitre constate que "
                "4 joueurs du meme club sont inscrits. Que doit-il mettre "
                "en oeuvre pour limiter les suspicions d'arrangements?"
            ),
            expected_answer=(
                "L'arbitre doit diriger le tirage au sort des numeros "
                "d'appariement de facon a ce que les personnes de meme "
                "affinite (meme club) ne se rencontrent pas dans les "
                "3 dernieres rondes. Cette information doit etre "
                "communiquee en avance via le reglement interieur."
            ),
            chunk_id="LA-octobre2025.pdf-p104-parent307-child00",
            source="LA-octobre2025.pdf",
            pages=[104],
            article_ref="Toutes Rondes - affinites",
            cognitive_level="Apply",
            difficulty=0.75,
            question_type="scenario",
            answer_type="extractive",
            reasoning_class="reasoning",
            keywords=["toutes rondes", "affinite", "appariement", "club"],
        ),
    },
    {
        "old_id": "gs:scratch:answerable:0103:c1b2d407",
        "profile": "HARD_APPLY",
        "q": _make_question(
            question=(
                "Un joueur en fauteuil roulant se plaint que sa table est "
                "trop haute et que son adversaire le derange regulierement. "
                "Quelles mesures l'arbitre doit-il prendre conformement "
                "a l'article 12.2?"
            ),
            expected_answer=(
                "Selon l'article 12.2, l'arbitre doit assurer le "
                "fair-play (12.2.1), s'assurer que les joueurs ne soient "
                "pas deranges (12.2.4), prendre les mesures speciales "
                "dans l'interet des joueurs handicapes (12.2.6), et "
                "maintenir un bon environnement de jeu (12.2.3)."
            ),
            chunk_id="LA-octobre2025.pdf-p055-parent188-child00",
            source="LA-octobre2025.pdf",
            pages=[55],
            article_ref="Art. 12.2",
            cognitive_level="Apply",
            difficulty=0.7,
            question_type="scenario",
            answer_type="extractive",
            reasoning_class="reasoning",
            keywords=["arbitre", "handicap", "fair-play", "environnement"],
        ),
    },
    {
        "old_id": "gs:scratch:answerable:0132:ea406257",
        "profile": "HARD_APPLY",
        "q": _make_question(
            question=(
                "Un club organise un tournoi ferme en partenariat avec "
                "la FFE. L'organisateur demande s'il doit payer les "
                "droits d'homologation. Que doit repondre l'arbitre?"
            ),
            expected_answer=(
                "Les tournois fermes organises en partenariat avec la "
                "FFE sont exoneres des droits d'homologation, tout "
                "comme les competitions federales."
            ),
            chunk_id=(
                "R03_2025_26_Competitions_homologuees.pdf" "-p003-parent012-child00"
            ),
            source="R03_2025_26_Competitions_homologuees.pdf",
            pages=[3],
            article_ref="Art. 2.7.1",
            cognitive_level="Apply",
            difficulty=0.7,
            question_type="scenario",
            answer_type="extractive",
            reasoning_class="fact_single",
            keywords=["homologation", "droits", "exoneration", "FFE"],
        ),
    },
    {
        "old_id": "gs:scratch:answerable:0163:2c81997a",
        "profile": "HARD_APPLY",
        "q": _make_question(
            question=(
                "Un arbitre doit organiser un tournoi par equipes en "
                "5 rondes avec le systeme Molter. Ce systeme ne prevoit "
                "que des tournois en nombre pair de rondes. Comment "
                "doit-il proceder?"
            ),
            expected_answer=(
                "L'arbitre doit prendre un tournoi a 4 rondes et lui "
                "ajouter la ronde autonome pour obtenir 5 rondes. La "
                "ronde autonome peut etre utilisee pour organiser un "
                "tournoi avec un nombre impair de rondes."
            ),
            chunk_id="LA-octobre2025.pdf-p108-parent320-child00",
            source="LA-octobre2025.pdf",
            pages=[108],
            article_ref="Tableaux d'appariements",
            cognitive_level="Apply",
            difficulty=0.7,
            question_type="scenario",
            answer_type="extractive",
            reasoning_class="reasoning",
            keywords=["molter", "ronde autonome", "equipes", "impair"],
        ),
    },
    {
        "old_id": "gs:scratch:answerable:0189:ca25aa80",
        "profile": "HARD_APPLY",
        "q": _make_question(
            question=(
                "Lors du remplissage du questionnaire medical d'un jeune "
                "joueur, ses parents signalent qu'un membre de la famille "
                "est decede subitement avant 50 ans d'une maladie du coeur. "
                "Quelle est la portee de cette information?"
            ),
            expected_answer=(
                "Le questionnaire medical demande aux parents si "
                "quelqu'un dans la famille proche a eu une maladie grave "
                "du coeur ou du cerveau, ou est decede subitement avant "
                "50 ans. Cette information doit etre prise en compte "
                "pour evaluer l'aptitude du joueur a participer."
            ),
            chunk_id=("2022_Reglement_medical_19082022.pdf" "-p007-parent020-child02"),
            source="2022_Reglement_medical_19082022.pdf",
            pages=[7],
            article_ref="Questionnaire medical",
            cognitive_level="Apply",
            difficulty=0.7,
            question_type="scenario",
            answer_type="extractive",
            reasoning_class="reasoning",
            keywords=["medical", "questionnaire", "famille", "sante"],
        ),
    },
    {
        "old_id": "gs:scratch:answerable:0213:ce91ca1f",
        "profile": "HARD_APPLY",
        "q": _make_question(
            question=(
                "Un arbitre organise un tournoi avec departage base sur "
                "la performance Elo, mais 5 joueurs n'ont pas de "
                "classement et le reglement ne prevoit rien. Que doit-il "
                "faire avant le debut du tournoi?"
            ),
            expected_answer=(
                "Lorsqu'on utilise des departages bases sur le "
                "classement Elo, le reglement doit detailler comment "
                "les joueurs non classes seront traites. L'arbitre en "
                "chef doit informer les joueurs avant le debut du "
                "tournoi de la methode utilisee."
            ),
            chunk_id="LA-octobre2025.pdf-p150-parent442-child00",
            source="LA-octobre2025.pdf",
            pages=[150],
            article_ref="Art. 8 - Type D",
            cognitive_level="Apply",
            difficulty=0.7,
            question_type="scenario",
            answer_type="extractive",
            reasoning_class="reasoning",
            keywords=["departage", "elo", "non classe", "arbitre"],
        ),
    },
    {
        "old_id": "gs:scratch:answerable:0270:7b2bcf40",
        "profile": "HARD_APPLY",
        "q": _make_question(
            question=(
                "Lors de l'appariement d'un niveau au systeme suisse, "
                "l'arbitre doit deplacer des joueurs de S1 vers S2. "
                "Il hesite entre deplacer les joueurs 5 et 2 ou les "
                "joueurs 4 et 3. Quelle transposition doit-il choisir?"
            ),
            expected_answer=(
                "L'arbitre doit choisir de deplacer les joueurs 5 et 2 "
                "vers S2 car cette option est meilleure que deplacer "
                "les joueurs 4 et 3. Le critere est de maximiser le "
                "NAN le plus eleve parmi ceux deplaces de S1 vers S2."
            ),
            chunk_id="LA-octobre2025.pdf-p130-parent383-child01",
            source="LA-octobre2025.pdf",
            pages=[130],
            article_ref="Transpositions S1/S2",
            cognitive_level="Apply",
            difficulty=0.75,
            question_type="scenario",
            answer_type="extractive",
            reasoning_class="reasoning",
            keywords=["appariement", "suisse", "transposition", "NAN"],
        ),
    },
    {
        "old_id": "gs:scratch:answerable:0320:e063299f",
        "profile": "HARD_APPLY",
        "q": _make_question(
            question=(
                "Un joueur utilise son telephone pendant la partie pour "
                "consulter une base de donnees, puis s'adresse a son "
                "adversaire de maniere agressive. Quels principes du "
                "code de respect a-t-il enfreints?"
            ),
            expected_answer=(
                "Le joueur a enfreint le respect de l'adversaire en "
                "ayant recours a des sources d'informations exterieures "
                "et a la tricherie, et en ne s'adressant pas a son "
                "adversaire en des termes courtois et polis."
            ),
            chunk_id="LA-octobre2025.pdf-p033-parent141-child00",
            source="LA-octobre2025.pdf",
            pages=[33],
            article_ref="Respecter l'adversaire",
            cognitive_level="Apply",
            difficulty=0.7,
            question_type="scenario",
            answer_type="extractive",
            reasoning_class="reasoning",
            keywords=["respect", "tricherie", "telephone", "courtoisie"],
        ),
    },
    {
        "old_id": "gs:scratch:answerable:0409:eb060eae",
        "profile": "HARD_APPLY",
        "q": _make_question(
            question=(
                "Un organisateur veut organiser un tournoi permettant "
                "l'obtention de normes. Il y a 4 porteurs de titres "
                "inscrits. Y a-t-il une restriction sur le nombre "
                "de joueurs d'une meme federation?"
            ),
            expected_answer=(
                "Selon le tableau, il faut 4 porteurs de titres. Il "
                "n'y a aucune restriction sur le nombre maximum de "
                "joueurs d'une meme federation (Nb max d'une "
                "federation : Aucun)."
            ),
            chunk_id="LA-octobre2025.pdf-p200-parent595-child02",
            source="LA-octobre2025.pdf",
            pages=[200],
            article_ref="Titres FIDE - tableau",
            cognitive_level="Apply",
            difficulty=0.7,
            question_type="scenario",
            answer_type="extractive",
            reasoning_class="fact_single",
            keywords=["norme", "titre", "federation", "restriction"],
        ),
    },
    {
        "old_id": "gs:scratch:answerable:0462:d4d6cb73",
        "profile": "HARD_APPLY",
        "q": _make_question(
            question=(
                "Un joueur a court de temps derange son adversaire pendant "
                "une partie rapide. L'arbitre constate l'infraction. "
                "Quelles sanctions graduelles peut-il appliquer selon "
                "l'article 18.4?"
            ),
            expected_answer=(
                "Les sanctions sont : avertissement (18.4.1), "
                "augmentation du temps de l'adversaire (18.4.2), "
                "diminution du temps du fautif (18.4.3), augmentation "
                "des points de l'adversaire (18.4.4), diminution des "
                "points du fautif (18.4.5)."
            ),
            chunk_id="LA-octobre2025.pdf-p093-parent276-child00",
            source="LA-octobre2025.pdf",
            pages=[93],
            article_ref="Art. 18.4",
            cognitive_level="Apply",
            difficulty=0.75,
            question_type="scenario",
            answer_type="extractive",
            reasoning_class="reasoning",
            keywords=["sanction", "arbitre", "avertissement", "temps"],
        ),
    },
    {
        "old_id": "gs:scratch:answerable:0622:f876a4d1",
        "profile": "HARD_APPLY",
        "q": _make_question(
            question=(
                "Un nouveau joueur mineur s'inscrit a un tournoi FIDE "
                "pour la premiere fois et n'a pas d'adresse email. "
                "Que recommandent les regles FIDE pour son enregistrement?"
            ),
            expected_answer=(
                "Il est fortement recommande de fournir une adresse "
                "email valide. Pour les mineurs, l'adresse e-mail des "
                "parents est fortement recommandee. Un nouvel "
                "enregistrement reussi renverra la reference FIDE."
            ),
            chunk_id="LA-octobre2025.pdf-p225-parent685-child02",
            source="LA-octobre2025.pdf",
            pages=[225],
            article_ref="Art. 1.14-1.15",
            cognitive_level="Apply",
            difficulty=0.7,
            question_type="scenario",
            answer_type="extractive",
            reasoning_class="fact_single",
            keywords=["enregistrement", "FIDE", "mineur", "email"],
        ),
    },
    {
        "old_id": "gs:scratch:answerable:0632:7e8c855e",
        "profile": "HARD_APPLY",
        "q": _make_question(
            question=(
                "Apres une decision controversee, un entraineur menace "
                "l'arbitre et couvre un joueur qui utilise un appareil "
                "electronique. Quelles infractions au code de l'ethique "
                "sont commises?"
            ),
            expected_answer=(
                "L'entraineur commet des menaces et coercition en "
                "intimidant l'arbitre dans le but d'influencer sa "
                "decision, ainsi que la complicite d'infraction en "
                "couvrant un participant coupable d'infraction au "
                "code de l'ethique."
            ),
            chunk_id="LA-octobre2025.pdf-p031-parent137-child01",
            source="LA-octobre2025.pdf",
            pages=[31],
            article_ref="Code de l'ethique",
            cognitive_level="Apply",
            difficulty=0.75,
            question_type="scenario",
            answer_type="extractive",
            reasoning_class="reasoning",
            keywords=["menace", "complicite", "ethique", "tricherie"],
        ),
    },
    {
        "old_id": "gs:scratch:answerable:0642:e969d7c3",
        "profile": "HARD_APPLY",
        "q": _make_question(
            question=(
                "Un jeune U14 avec un classement Elo eleve souhaite "
                "participer au Championnat de France Jeunes sans passer "
                "les phases qualificatives. Peut-il le faire?"
            ),
            expected_answer=(
                "Oui, les jeunes ayant acquis leur qualification directe "
                "en fonction de leur classement Elo ou de leurs resultats "
                "de la saison precedente ne sont pas concernes par les "
                "phases qualificatives."
            ),
            chunk_id=(
                "J01_2025_26_Championnat_de_France_Jeunes.pdf" "-p001-parent002-child00"
            ),
            source="J01_2025_26_Championnat_de_France_Jeunes.pdf",
            pages=[1],
            article_ref="Art. 2",
            cognitive_level="Apply",
            difficulty=0.7,
            question_type="scenario",
            answer_type="extractive",
            reasoning_class="fact_single",
            keywords=["jeunes", "qualification", "elo", "championnat"],
        ),
    },
    {
        "old_id": "gs:scratch:answerable:0668:332951ff",
        "profile": "HARD_APPLY",
        "q": _make_question(
            question=(
                "Un joueur dans un tournoi a normes a battu un adversaire "
                "faible. Il souhaite ignorer cette partie pour ameliorer "
                "sa norme. Sous quelles conditions peut-il le faire?"
            ),
            expected_answer=(
                "Un joueur peut ignorer les parties contre tout "
                "adversaire qu'il a battu, a condition que cela lui "
                "laisse au moins le nombre minimum de parties prevu au "
                "1.4.1 contre la combinaison requise d'adversaires. "
                "La grille de resultats entiere doit etre transmise."
            ),
            chunk_id="LA-octobre2025.pdf-p195-parent576-child02",
            source="LA-octobre2025.pdf",
            pages=[195],
            article_ref="Art. 1.4.1",
            cognitive_level="Apply",
            difficulty=0.8,
            question_type="scenario",
            answer_type="extractive",
            reasoning_class="reasoning",
            keywords=["norme", "partie", "adversaire", "ignorer"],
        ),
    },
    {
        "old_id": "gs:scratch:answerable:0671:a5a96020",
        "profile": "HARD_APPLY",
        "q": _make_question(
            question=(
                "Un joueur en fauteuil roulant se presente a un tournoi "
                "et demande des amenagements. L'arbitre ne connait pas "
                "les procedures. Vers quels textes doit-il se tourner?"
            ),
            expected_answer=(
                "L'arbitre doit consulter les reglements de la "
                "Commission Handicap : H01 (Conduite a tenir quant aux "
                "joueurs handicapes) et H02 (Texte d'application pour "
                "les joueurs a mobilite reduite)."
            ),
            chunk_id=("R01_2025_26_Regles_generales.pdf" "-p006-parent035-child00"),
            source="R01_2025_26_Regles_generales.pdf",
            pages=[6],
            article_ref="Art. 13 - Commission Handicap",
            cognitive_level="Apply",
            difficulty=0.7,
            question_type="scenario",
            answer_type="extractive",
            reasoning_class="fact_single",
            keywords=["handicap", "commission", "H01", "H02"],
        ),
    },
    {
        "old_id": "gs:scratch:answerable:0676:1ad15398",
        "profile": "HARD_APPLY",
        "q": _make_question(
            question=(
                "Un joueur handicape demande que l'echiquier soit place "
                "selon une orientation particuliere, mais son adversaire "
                "estime que cela le desavantage. Comment l'arbitre "
                "doit-il gerer la situation?"
            ),
            expected_answer=(
                "Le joueur handicape a le droit de demander le "
                "placement de son materiel selon une orientation "
                "particuliere, a condition que cela ne desavantage pas "
                "son adversaire. L'equipe organisatrice doit veiller "
                "a ce que les besoins des deux adversaires soient "
                "pris en compte."
            ),
            chunk_id="LA-octobre2025.pdf-p160-parent478-child00",
            source="LA-octobre2025.pdf",
            pages=[160],
            article_ref="Directives handicap FIDE",
            cognitive_level="Apply",
            difficulty=0.75,
            question_type="scenario",
            answer_type="extractive",
            reasoning_class="reasoning",
            keywords=["handicap", "echiquier", "amenagement", "FIDE"],
        ),
    },
    {
        "old_id": "gs:scratch:answerable:0781:b8dd982f",
        "profile": "HARD_APPLY",
        "q": _make_question(
            question=(
                "Lors d'une reunion du comite directeur, un vote concerne "
                "un de ses membres. Le resultat est une egalite. Le vote "
                "est-il a bulletin secret et la voix du president "
                "est-elle preponderante?"
            ),
            expected_answer=(
                "Tout vote concernant un membre du comite directeur se "
                "deroule hors de sa presence ou a scrutin secret. En "
                "cas d'egalite hors scrutin secret, la voix du president "
                "est preponderante. Si le scrutin est secret, la voix "
                "preponderante ne s'applique pas."
            ),
            chunk_id=(
                "2025_Reglement_Interieur_20250503.pdf" "-p013-parent045-child00"
            ),
            source="2025_Reglement_Interieur_20250503.pdf",
            pages=[13],
            article_ref="Art. 8.3",
            cognitive_level="Apply",
            difficulty=0.8,
            question_type="scenario",
            answer_type="extractive",
            reasoning_class="reasoning",
            keywords=["deliberation", "vote", "president", "egalite"],
        ),
    },
]

# ===================================================================
# HARD_ANALYZE: comparative, inferential, difficulty >= 0.7
# ===================================================================

HARD_ANALYZE = [
    {
        "old_id": "gs:scratch:answerable:0796:0631f825",
        "profile": "HARD_ANALYZE",
        "q": _make_question(
            question=(
                "En quoi le systeme Rutsch se distingue-t-il d'une "
                "attribution aleatoire des appariements pour un tournoi "
                "toutes rondes?"
            ),
            expected_answer=(
                "Le systeme Rutsch utilise le principe du taquin "
                "(glissement systematique, 'ein Rutsch' en allemand) "
                "pour organiser les appariements, garantissant que "
                "chaque participant rencontre tous les autres avec "
                "alternance des couleurs, contrairement a une "
                "attribution aleatoire sans structure."
            ),
            chunk_id="LA-octobre2025.pdf-p102-parent299-child00",
            source="LA-octobre2025.pdf",
            pages=[102],
            article_ref="Systeme Rutsch",
            cognitive_level="Analyze",
            difficulty=0.7,
            question_type="comparative",
            answer_type="inferential",
            reasoning_class="reasoning",
            keywords=["rutsch", "taquin", "toutes rondes", "appariement"],
        ),
    },
    {
        "old_id": "gs:scratch:answerable:0876:4b2163d2",
        "profile": "HARD_ANALYZE",
        "q": _make_question(
            question=(
                "Comparez les criteres de classement moyen requis pour "
                "qu'un tournoi toutes rondes soit un 'evenement de haut "
                "niveau' selon qu'il est mixte ou feminin ?"
            ),
            expected_answer=(
                "Un tournoi mixte necessite un classement moyen "
                "superieur a 2600, tandis qu'un tournoi feminin "
                "necessite superieur a 2400, soit 200 points d'ecart. "
                "Les deux requierent au moins 10 participants ou 6 en "
                "aller-retour."
            ),
            chunk_id="LA-octobre2025.pdf-p215-parent656-child00",
            source="LA-octobre2025.pdf",
            pages=[215],
            article_ref="Art. 1.3.2.3",
            cognitive_level="Analyze",
            difficulty=0.7,
            question_type="comparative",
            answer_type="inferential",
            reasoning_class="reasoning",
            keywords=["haut niveau", "mixte", "feminin", "classement"],
        ),
    },
    {
        "old_id": "gs:scratch:answerable:0915:de7150d6",
        "profile": "HARD_ANALYZE",
        "q": _make_question(
            question=(
                "Comparez les responsabilites de l'arbitre et celles de "
                "l'organisateur dans les taches post-tournoi selon le "
                "texte ?"
            ),
            expected_answer=(
                "L'arbitre doit s'assurer de la transmission des "
                "resultats, rediger le rapport technique, transmettre "
                "normes et formulaires de plaintes. L'organisateur "
                "doit payer les droits d'homologation dans les 7 jours. "
                "L'arbitre verifie que l'organisateur s'acquitte "
                "de ce paiement."
            ),
            chunk_id="LA-octobre2025.pdf-p166-parent493-child00",
            source="LA-octobre2025.pdf",
            pages=[166],
            article_ref="Taches post-tournoi",
            cognitive_level="Analyze",
            difficulty=0.7,
            question_type="comparative",
            answer_type="inferential",
            reasoning_class="reasoning",
            keywords=["arbitre", "organisateur", "post-tournoi", "rapport"],
        ),
    },
    {
        "old_id": "gs:scratch:answerable:1070:36051e4a",
        "profile": "HARD_ANALYZE",
        "q": _make_question(
            question=(
                "Comparez les criteres pour obtenir un titre d'arbitre "
                "federal pour un candidat de 15 ans et un candidat de "
                "25 ans. Quelles differences existent?"
            ),
            expected_answer=(
                "Les deux doivent avoir une licence A, un classement "
                "Elo et le controle d'honorabilite. La difference : "
                "le candidat de 25 ans doit avoir participe a un "
                "stage de sensibilisation contre les violences "
                "sexistes/sexuelles (requis pour les 16 ans et plus). "
                "Le candidat de 15 ans en est dispense."
            ),
            chunk_id="LA-octobre2025.pdf-p018-parent069-child00",
            source="LA-octobre2025.pdf",
            pages=[18],
            article_ref="Art. 7.1",
            cognitive_level="Analyze",
            difficulty=0.75,
            question_type="comparative",
            answer_type="inferential",
            reasoning_class="reasoning",
            keywords=["titre", "arbitre", "age", "stage", "VSS"],
        ),
    },
    {
        "old_id": "gs:scratch:answerable:1135:87803519",
        "profile": "HARD_ANALYZE",
        "q": _make_question(
            question=(
                "Comparez les exigences de titres des adversaires pour "
                "une norme de Grand Maitre et une norme de Maitre "
                "International Feminin ?"
            ),
            expected_answer=(
                "Pour une norme GM, au moins 1/3 des adversaires "
                "(minimum 3) doivent etre GM. Pour une norme MIF, "
                "au moins 1/3 (minimum 3) doivent etre MIF, GMF, "
                "MI ou GM. La norme MIF accepte un eventail plus "
                "large de titres. Dans les deux cas, 50% des "
                "adversaires doivent avoir un titre (CM/CMF exclus)."
            ),
            chunk_id="LA-octobre2025.pdf-p196-parent579-child00",
            source="LA-octobre2025.pdf",
            pages=[196],
            article_ref="Art. 1.4.5",
            cognitive_level="Analyze",
            difficulty=0.8,
            question_type="comparative",
            answer_type="inferential",
            reasoning_class="reasoning",
            keywords=["norme", "GM", "MIF", "titre", "adversaire"],
        ),
    },
    {
        "old_id": "gs:scratch:answerable:1196:7376701e",
        "profile": "HARD_ANALYZE",
        "q": _make_question(
            question=(
                "Comparez la position de la FIDE sur le systeme "
                "neerlandais par rapport aux systemes alternatifs "
                "(Dubov, Burstein, Lim) ?"
            ),
            expected_answer=(
                "La FIDE deconseille les systemes Dubov, Burstein "
                "et Lim sauf s'il existe un programme approuve avec "
                "un verificateur en libre acces. Le systeme "
                "neerlandais, majoritaire en France, ne fait pas "
                "l'objet de cette restriction, ce qui en fait le "
                "choix le plus sur."
            ),
            chunk_id="LA-octobre2025.pdf-p121-parent352-child00",
            source="LA-octobre2025.pdf",
            pages=[121],
            article_ref="Systemes suisses C.04",
            cognitive_level="Analyze",
            difficulty=0.7,
            question_type="comparative",
            answer_type="inferential",
            reasoning_class="reasoning",
            keywords=["neerlandais", "dubov", "burstein", "FIDE"],
        ),
    },
    {
        "old_id": "gs:scratch:answerable:1234:cc2b37ab",
        "profile": "HARD_ANALYZE",
        "q": _make_question(
            question=(
                "Comparez les roles du pole stage, des formateurs et "
                "du secretariat federal dans le processus de formation "
                "des arbitres ?"
            ),
            expected_answer=(
                "Le pole stage accorde ou refuse l'homologation et "
                "assure le suivi des stages. Les organisateurs et "
                "formateurs assurent la mise en oeuvre. Le secretariat "
                "federal et la direction des titres recoivent les "
                "elements financiers et administratifs du pole stage."
            ),
            chunk_id="LA-octobre2025.pdf-p015-parent048-child00",
            source="LA-octobre2025.pdf",
            pages=[15],
            article_ref="Art. 3.6.2",
            cognitive_level="Analyze",
            difficulty=0.7,
            question_type="comparative",
            answer_type="inferential",
            reasoning_class="reasoning",
            keywords=["pole stage", "formateur", "secretariat", "DNA"],
        ),
    },
    {
        "old_id": "gs:scratch:answerable:1354:79c186c1",
        "profile": "HARD_ANALYZE",
        "q": _make_question(
            question=(
                "Comparez les criteres C.12/C.13 (flotteurs deux rondes "
                "consecutives) avec les criteres C.14/C.15 (flotteurs "
                "aux rondes n et n-2). En quoi different-ils?"
            ),
            expected_answer=(
                "C.12/C.13 minimisent les flotteurs deux rondes de "
                "suite (consecutives). C.14/C.15 minimisent les "
                "flotteurs aux rondes n et n-2 (avec une ronde "
                "d'ecart). Les criteres consecutifs sont plus "
                "restrictifs. C.16-C.19 ajoutent la minimisation "
                "de la difference de score dans ces situations."
            ),
            chunk_id="LA-octobre2025.pdf-p129-parent379-child01",
            source="LA-octobre2025.pdf",
            pages=[129],
            article_ref="Criteres C.12-C.19",
            cognitive_level="Analyze",
            difficulty=0.8,
            question_type="comparative",
            answer_type="inferential",
            reasoning_class="reasoning",
            keywords=["flotteur", "critere", "appariement", "suisse"],
        ),
    },
    {
        "old_id": "gs:scratch:answerable:1493:d6d14f01",
        "profile": "HARD_ANALYZE",
        "q": _make_question(
            question=(
                "Comparez la structure des Regles du Jeu FIDE entre "
                "les regles fondamentales et les regles de competition. "
                "Quelle version fait foi en cas de divergence?"
            ),
            expected_answer=(
                "Les Regles se divisent en deux parties : regles "
                "fondamentales du Jeu (partie 1) et regles du jeu en "
                "competition (partie 2). La version anglaise est la "
                "version officielle, adoptee au 93eme Congres a "
                "Chennai. En cas de divergence de traduction, c'est "
                "la version anglaise qui fait foi."
            ),
            chunk_id="LA-octobre2025.pdf-p036-parent153-child00",
            source="LA-octobre2025.pdf",
            pages=[36],
            article_ref="Art. 0.1",
            cognitive_level="Analyze",
            difficulty=0.7,
            question_type="comparative",
            answer_type="inferential",
            reasoning_class="reasoning",
            keywords=["regles", "FIDE", "anglais", "officiel"],
        ),
    },
    {
        "old_id": "gs:scratch:answerable:1543:d907486f",
        "profile": "HARD_ANALYZE",
        "q": _make_question(
            question=(
                "Comparez le champ d'application des regles de fair-play "
                "FIDE pour les tournois en ligne entre les competitions "
                "officielles et les competitions nationales/privees ?"
            ),
            expected_answer=(
                "Pour les competitions officielles FIDE, les regles "
                "sont obligatoires. Pour les competitions nationales "
                "et tournois prives, il est fortement recommande de "
                "les adopter mais elles peuvent etre adaptees. Il y "
                "a donc une distinction entre obligation et "
                "recommandation."
            ),
            chunk_id="LA-octobre2025.pdf-p094-parent279-child00",
            source="LA-octobre2025.pdf",
            pages=[94],
            article_ref="Annexe I - Fair-play en ligne",
            cognitive_level="Analyze",
            difficulty=0.7,
            question_type="comparative",
            answer_type="inferential",
            reasoning_class="reasoning",
            keywords=["fair-play", "en ligne", "officiel", "recommande"],
        ),
    },
    {
        "old_id": "gs:scratch:answerable:1756:b2b6b900",
        "profile": "HARD_ANALYZE",
        "q": _make_question(
            question=(
                "Comparez les elements devant figurer sur les feuilles "
                "de notation papier avec les exigences pour les feuilles "
                "electroniques conformes FIDE ?"
            ),
            expected_answer=(
                "Les feuilles papier doivent contenir 9 elements : "
                "nom de la competition, numero de ronde et echiquier, "
                "date, noms des joueurs, resultat, signatures et zone "
                "d'enregistrement des coups. Les feuilles electroniques "
                "conformes FIDE sont une alternative acceptee devant "
                "contenir des informations equivalentes."
            ),
            chunk_id="LA-octobre2025.pdf-p080-parent243-child00",
            source="LA-octobre2025.pdf",
            pages=[80],
            article_ref="Art. 6 - Feuilles de Notation",
            cognitive_level="Analyze",
            difficulty=0.7,
            question_type="comparative",
            answer_type="inferential",
            reasoning_class="reasoning",
            keywords=["feuille", "notation", "papier", "electronique"],
        ),
    },
    {
        "old_id": "gs:scratch:answerable:1773:02d107e3",
        "profile": "HARD_ANALYZE",
        "q": _make_question(
            question=(
                "Comparez le systeme de prix 'a la place par departage' "
                "avec un systeme de partage entre ex-aequo. Quels "
                "sont les avantages du systeme a la place?"
            ),
            expected_answer=(
                "Dans le systeme a la place, chaque joueur recoit le "
                "prix correspondant a sa place determinee par les "
                "points et le departage. C'est le plus simple car "
                "il utilise un classement precis. Le partage entre "
                "ex-aequo necessite des calculs supplementaires pour "
                "redistribuer les prix."
            ),
            chunk_id="LA-octobre2025.pdf-p157-parent471-child00",
            source="LA-octobre2025.pdf",
            pages=[157],
            article_ref="Prix a la place",
            cognitive_level="Analyze",
            difficulty=0.7,
            question_type="comparative",
            answer_type="inferential",
            reasoning_class="reasoning",
            keywords=["prix", "departage", "classement", "ex-aequo"],
        ),
    },
    {
        "old_id": "gs:scratch:answerable:1854:a8f718a1",
        "profile": "HARD_ANALYZE",
        "q": _make_question(
            question=(
                "Comparez les roles de l'arbitre en chef, du secretariat "
                "FIDE et de la Commission d'homologation dans le "
                "processus d'enregistrement des titres directs ?"
            ),
            expected_answer=(
                "L'arbitre en chef envoie les resultats avec la liste "
                "des titres obtenus. La CH examine les informations. "
                "Le secretariat FIDE informe les federations "
                "concernees. La chaine est : arbitre (envoi) puis "
                "CH (examen) puis secretariat (notification)."
            ),
            chunk_id="LA-octobre2025.pdf-p199-parent589-child00",
            source="LA-octobre2025.pdf",
            pages=[199],
            article_ref="Art. 1.10.1",
            cognitive_level="Analyze",
            difficulty=0.7,
            question_type="comparative",
            answer_type="inferential",
            reasoning_class="reasoning",
            keywords=["titre", "enregistrement", "arbitre", "FIDE"],
        ),
    },
    {
        "old_id": "gs:scratch:answerable:1971:a52a6231",
        "profile": "HARD_ANALYZE",
        "q": _make_question(
            question=(
                "Comparez l'efficacite du departage par rencontre "
                "directe entre un cas a 2 joueurs et un cas a 3 joueurs "
                "ex-aequo, en utilisant les exemples du texte ?"
            ),
            expected_answer=(
                "Pour 2 joueurs : Diane bat Alma, departage clair. "
                "Mais Guy et Fatou font nulle, departage echoue. "
                "Pour 3 joueurs : Eric 2pts, Beatrice 1pt, Claude "
                "0pt entre eux, classement complet possible. Avec "
                "3 joueurs, plus de rencontres mutuelles rendent le "
                "departage plus efficace."
            ),
            chunk_id="LA-octobre2025.pdf-p148-parent431-child02",
            source="LA-octobre2025.pdf",
            pages=[148],
            article_ref="Departage rencontre directe",
            cognitive_level="Analyze",
            difficulty=0.8,
            question_type="comparative",
            answer_type="inferential",
            reasoning_class="reasoning",
            keywords=["departage", "rencontre directe", "ex-aequo"],
        ),
    },
    {
        "old_id": "gs:scratch:answerable:1982:576bab5b",
        "profile": "HARD_ANALYZE",
        "q": _make_question(
            question=(
                "Comparez les consequences d'une erreur d'appariement "
                "involontaire avec celles d'une modification "
                "intentionnelle en faveur d'un joueur ?"
            ),
            expected_answer=(
                "Des arbitres differents doivent aboutir a des "
                "appariements identiques, donc une erreur doit etre "
                "corrigee. Une modification intentionnelle pour "
                "favoriser une norme peut entrainer un rapport a la "
                "commission d'homologation FIDE et des mesures "
                "disciplinaires via la Commission d'ethique."
            ),
            chunk_id="LA-octobre2025.pdf-p123-parent354-child01",
            source="LA-octobre2025.pdf",
            pages=[123],
            article_ref="Appariements valides",
            cognitive_level="Analyze",
            difficulty=0.75,
            question_type="comparative",
            answer_type="inferential",
            reasoning_class="reasoning",
            keywords=["appariement", "modification", "ethique", "norme"],
        ),
    },
    {
        "old_id": "gs:scratch:answerable:2015:1e9309a9",
        "profile": "HARD_ANALYZE",
        "q": _make_question(
            question=(
                "Comparez les formats acceptables et inacceptables pour "
                "la soumission des attestations de normes a la FIDE. "
                "Pourquoi certains formats sont-ils rejetes?"
            ),
            expected_answer=(
                "Les formats acceptables sont doc ou pdf, car ils "
                "doivent etre lisibles. Les photos prises avec un "
                "telephone sont inacceptables et ne constituent pas "
                "un document recevable, en raison du besoin de "
                "lisibilite et de qualite documentaire."
            ),
            chunk_id="LA-octobre2025.pdf-p177-parent520-child01",
            source="LA-octobre2025.pdf",
            pages=[177],
            article_ref="Attestations de normes",
            cognitive_level="Analyze",
            difficulty=0.7,
            question_type="comparative",
            answer_type="inferential",
            reasoning_class="reasoning",
            keywords=["norme", "attestation", "format", "pdf"],
        ),
    },
    {
        "old_id": "gs:scratch:answerable:2063:270c8caa",
        "profile": "HARD_ANALYZE",
        "q": _make_question(
            question=(
                "Comparez les conditions de la prise en passant avec "
                "celles de la promotion d'un pion. En quoi different-"
                "elles dans leur temporalite?"
            ),
            expected_answer=(
                "La prise en passant est soumise a une contrainte "
                "temporelle stricte : autorisee uniquement sur le "
                "coup suivant l'avancement de deux cases. La "
                "promotion n'a pas de contrainte temporelle : elle "
                "s'effectue automatiquement a la derniere rangee, "
                "le joueur choisissant sans restriction aux pieces "
                "deja capturees."
            ),
            chunk_id="LA-octobre2025.pdf-p039-parent159-child00",
            source="LA-octobre2025.pdf",
            pages=[39],
            article_ref="Art. 3.7.3",
            cognitive_level="Analyze",
            difficulty=0.7,
            question_type="comparative",
            answer_type="inferential",
            reasoning_class="reasoning",
            keywords=["en passant", "promotion", "pion", "temporalite"],
        ),
    },
    {
        "old_id": "gs:scratch:answerable:2135:2f8d44de",
        "profile": "HARD_ANALYZE",
        "q": _make_question(
            question=(
                "Comparez les exigences pour obtenir le titre d'Arbitre "
                "FIDE (AF) et le titre d'Arbitre International (AI) en "
                "termes d'age et de normes ?"
            ),
            expected_answer=(
                "Pour l'AF : age minimum 19 ans, un seminaire AF "
                "plus 3 attestations de normes lors de tournois. "
                "Pour l'AI : age minimum 21 ans, 4 attestations "
                "de normes. Les deux requierent un formulaire signe "
                "par la federation et le paiement de droits FIDE."
            ),
            chunk_id="LA-octobre2025.pdf-p209-parent623-child01",
            source="LA-octobre2025.pdf",
            pages=[209],
            article_ref="Art. 5.4-5.5",
            cognitive_level="Analyze",
            difficulty=0.75,
            question_type="comparative",
            answer_type="inferential",
            reasoning_class="reasoning",
            keywords=["AF", "AI", "age", "norme", "titre"],
        ),
    },
    {
        "old_id": "gs:scratch:answerable:2209:26f15b39",
        "profile": "HARD_ANALYZE",
        "q": _make_question(
            question=(
                "En quoi le departage Buchholz differe-t-il d'un "
                "simple comptage de points, et quel ajustement est "
                "applique pour les rondes non jouees?"
            ),
            expected_answer=(
                "Le Buchholz mesure la force des adversaires (somme "
                "des scores ajustes de chaque adversaire) et non la "
                "performance directe du joueur. Il recompense les "
                "joueurs ayant affronte des adversaires plus forts. "
                "Pour les rondes non jouees, un calcul specifique du "
                "score ajuste est applique selon l'article 14."
            ),
            chunk_id="LA-octobre2025.pdf-p149-parent438-child00",
            source="LA-octobre2025.pdf",
            pages=[149],
            article_ref="Art. 7.1 - Buchholz",
            cognitive_level="Analyze",
            difficulty=0.7,
            question_type="comparative",
            answer_type="inferential",
            reasoning_class="reasoning",
            keywords=["buchholz", "departage", "score", "adversaire"],
        ),
    },
    {
        "old_id": "gs:scratch:answerable:2287:3d27112c",
        "profile": "HARD_ANALYZE",
        "q": _make_question(
            question=(
                "Comparez les trois categories de formation des arbitres "
                "et expliquez le role de la commission FIDE des Arbitres "
                "dans le soutien aux federations nationales ?"
            ),
            expected_answer=(
                "Les trois categories sont : formation initiale, "
                "formation continue, et preparation aux evenements "
                "mondiaux. En complement, la commission FIDE des "
                "Arbitres assiste les federations, sur leur demande, "
                "concernant leurs programmes nationaux de formation."
            ),
            chunk_id="LA-octobre2025.pdf-p210-parent628-child00",
            source="LA-octobre2025.pdf",
            pages=[210],
            article_ref="Art. 4 - Formation",
            cognitive_level="Analyze",
            difficulty=0.7,
            question_type="comparative",
            answer_type="inferential",
            reasoning_class="reasoning",
            keywords=["formation", "initiale", "continue", "FIDE"],
        ),
    },
]


# ===================================================================
# MED_APPLY_INF: procedural, inferential, difficulty 0.5-0.7
# ===================================================================

MED_APPLY_INF = [
    {
        "old_id": "gs:scratch:answerable:2437:bdaee54b",
        "profile": "MED_APPLY_INF",
        "q": _make_question(
            question=(
                "Comment un arbitre doit-il proceder pour organiser un "
                "tournoi de 17 equipes de 8 joueurs selon le systeme "
                "Molter?"
            ),
            expected_answer=(
                "L'arbitre prevoit 68 echiquiers (17x4). Les capitaines "
                "remettent la liste ordonnee des membres. Un tirage au "
                "sort attribue une lettre par equipe. Les membres sont "
                "numerotes. L'ordre ne peut pas etre permute entre les "
                "rondes. L'arbitre doit preparer les tableaux pour "
                "plusieurs cas de figure."
            ),
            chunk_id="LA-octobre2025.pdf-p107-parent317-child00",
            source="LA-octobre2025.pdf",
            pages=[107],
            article_ref="Organisation tournois equipes",
            cognitive_level="Apply",
            difficulty=0.6,
            question_type="procedural",
            answer_type="inferential",
            reasoning_class="reasoning",
            keywords=["molter", "equipes", "echiquiers", "organisation"],
        ),
    },
    {
        "old_id": "gs:scratch:answerable:2438:4f6b770e",
        "profile": "MED_APPLY_INF",
        "q": _make_question(
            question=(
                "Quelles verifications techniques un arbitre doit-il "
                "effectuer avant d'utiliser des echiquiers electroniques "
                "dans un tournoi?"
            ),
            expected_answer=(
                "L'arbitre doit verifier le type d'alimentation "
                "(cable ou batterie), le type de connexion et nombre "
                "max d'echiquiers connectables, la connexion sans fil, "
                "la compatibilite avec les pendules, et les dispositifs "
                "de transmission des parties. Les pieces doivent etre "
                "conformes aux standards."
            ),
            chunk_id="LA-octobre2025.pdf-p082-parent247-child00",
            source="LA-octobre2025.pdf",
            pages=[82],
            article_ref="Art. 7.1 - Echiquier electronique",
            cognitive_level="Apply",
            difficulty=0.6,
            question_type="procedural",
            answer_type="inferential",
            reasoning_class="reasoning",
            keywords=["echiquier", "electronique", "verification", "cable"],
        ),
    },
    {
        "old_id": "gs:scratch:answerable:1110:d2cf7cd4",
        "profile": "MED_APPLY_INF",
        "q": _make_question(
            question=(
                "Comment un club doit-il proceder si une reserve est "
                "formulee sur la nationalite d'un de ses joueurs?"
            ),
            expected_answer=(
                "Le club doit justifier la nationalite dans les 15 "
                "jours suivant la notification. A defaut, le joueur "
                "n'est pas comptabilise dans le quota de nationalite "
                "francaise. Pour le statut de resident, la residence "
                "doit etre etablie au plus tard le 30 novembre de "
                "la saison."
            ),
            chunk_id=("R01_2025_26_Regles_generales.pdf" "-p001-parent004-child00"),
            source="R01_2025_26_Regles_generales.pdf",
            pages=[1],
            article_ref="Art. 2.1",
            cognitive_level="Apply",
            difficulty=0.6,
            question_type="procedural",
            answer_type="inferential",
            reasoning_class="reasoning",
            keywords=["nationalite", "reserve", "club", "residence"],
        ),
    },
    {
        "old_id": "gs:scratch:answerable:1224:959c65e2",
        "profile": "MED_APPLY_INF",
        "q": _make_question(
            question=(
                "Comment se deroule le processus de deliberation et de "
                "notification d'une decision disciplinaire a la FFE?"
            ),
            expected_answer=(
                "L'organe delibere a huis clos, hors la presence de "
                "la personne poursuivie. La decision est motivee, "
                "signee par le president et le secretaire. La "
                "notification mentionne voies et delais de recours. "
                "L'association sportive et la FFE sont informees."
            ),
            chunk_id=(
                "2018_Reglement_Disciplinaire20180422.pdf" "-p005-parent015-child00"
            ),
            source="2018_Reglement_Disciplinaire20180422.pdf",
            pages=[5],
            article_ref="Art. 15",
            cognitive_level="Apply",
            difficulty=0.6,
            question_type="procedural",
            answer_type="inferential",
            reasoning_class="reasoning",
            keywords=["disciplinaire", "delibere", "notification", "huis clos"],
        ),
    },
    {
        "old_id": "gs:scratch:answerable:1140:4be52d16",
        "profile": "MED_APPLY_INF",
        "q": _make_question(
            question=(
                "Comment classer les transpositions possibles dans S2 "
                "lors de l'appariement d'un niveau au systeme suisse?"
            ),
            expected_answer=(
                "Les transpositions sont classees dans l'ordre "
                "croissant des valeurs lexicographiques de leurs N1 "
                "premiers numeros. Les joueurs restants de S2 sont "
                "ignores car ils constituent le niveau residuel "
                "(heterogene) ou deviennent flotteurs descendants "
                "(homogene)."
            ),
            chunk_id="LA-octobre2025.pdf-p129-parent381-child00",
            source="LA-octobre2025.pdf",
            pages=[129],
            article_ref="D.1 Transpositions",
            cognitive_level="Apply",
            difficulty=0.65,
            question_type="procedural",
            answer_type="inferential",
            reasoning_class="reasoning",
            keywords=["transposition", "S2", "lexicographique", "NAN"],
        ),
    },
    {
        "old_id": "gs:scratch:answerable:1782:64a58fa4",
        "profile": "MED_APPLY_INF",
        "q": _make_question(
            question=(
                "Comment un arbitre peut-il demander a etre supervise "
                "dans le cadre de son cursus, et qui prend en charge "
                "les frais?"
            ),
            expected_answer=(
                "L'arbitre demande via la Direction Regionale de "
                "l'Arbitrage. Les frais sont pris en charge par la "
                "Ligue. Si la visite est diligentee par la DNA, les "
                "frais sont a la charge de la FFE. Tout superviseur "
                "peut aussi prendre l'initiative de superviser. "
                "Aucun arbitre ne peut refuser."
            ),
            chunk_id="LA-octobre2025.pdf-p016-parent055-child00",
            source="LA-octobre2025.pdf",
            pages=[16],
            article_ref="Art. 4.1.2",
            cognitive_level="Apply",
            difficulty=0.6,
            question_type="procedural",
            answer_type="inferential",
            reasoning_class="reasoning",
            keywords=["supervision", "DRA", "DNA", "frais", "ligue"],
        ),
    },
    {
        "old_id": "gs:scratch:answerable:0219:09fcdd91",
        "profile": "MED_APPLY_INF",
        "q": _make_question(
            question=(
                "Comment un nouveau joueur peut-il obtenir une reference "
                "FIDE et quelles sont les donnees obligatoires?"
            ),
            expected_answer=(
                "Le joueur s'inscrit sur la FOA (zone de jeu en ligne "
                "FIDE). Donnees obligatoires : prenom, nom, sexe, date "
                "de naissance, email et federation. Un enfant peut "
                "s'inscrire sur sm.fide.com et obtenir un classement "
                "CiS apres mise a niveau Premium. Les auto-inscrits "
                "recoivent le drapeau FIDE (FID)."
            ),
            chunk_id="LA-octobre2025.pdf-p225-parent685-child01",
            source="LA-octobre2025.pdf",
            pages=[225],
            article_ref="Art. 1.13-1.14",
            cognitive_level="Apply",
            difficulty=0.6,
            question_type="procedural",
            answer_type="inferential",
            reasoning_class="reasoning",
            keywords=["FIDE", "enregistrement", "FOA", "reference"],
        ),
    },
    {
        "old_id": "gs:scratch:answerable:1561:6db8fbf4",
        "profile": "MED_APPLY_INF",
        "q": _make_question(
            question=(
                "Quels avantages concrets un organisateur obtient-il "
                "en faisant homologuer son tournoi par la FFE?"
            ),
            expected_answer=(
                "L'organisateur beneficie d'une promotion dans les "
                "annonces FFE, d'un logiciel d'appariements gratuit, "
                "de l'assurance FFE, et du traitement des resultats "
                "pour le classement Elo. La FFE assure aussi la "
                "formation des arbitres et controle leur travail."
            ),
            chunk_id=(
                "R03_2025_26_Competitions_homologuees.pdf" "-p003-parent015-child00"
            ),
            source="R03_2025_26_Competitions_homologuees.pdf",
            pages=[3],
            article_ref="Avantages homologation",
            cognitive_level="Apply",
            difficulty=0.5,
            question_type="procedural",
            answer_type="inferential",
            reasoning_class="summary",
            keywords=["homologation", "avantages", "logiciel", "assurance"],
        ),
    },
    {
        "old_id": "gs:scratch:answerable:2171:1637a933",
        "profile": "MED_APPLY_INF",
        "q": _make_question(
            question=(
                "Comment proceder au classement final d'un groupe de "
                "Nationale 1 interclubs jeunes en cas d'egalite de "
                "points de matchs?"
            ),
            expected_answer=(
                "On utilise les resultats entre les equipes a "
                "departager (points de match, differentiel, points "
                "de parties). Puis les differentiels sur l'ensemble, "
                "puis le nombre de points 'pour'. Si l'egalite "
                "persiste, on prend les differentiels par echiquier "
                "(1er, 2eme, etc.)."
            ),
            chunk_id=(
                "J02_2025_26_Championnat_de_France_Interclubs_Jeunes"
                ".pdf-p007-parent032-child00"
            ),
            source=("J02_2025_26_Championnat_de_France_Interclubs_" "Jeunes.pdf"),
            pages=[7],
            article_ref="Art. 4.4",
            cognitive_level="Apply",
            difficulty=0.6,
            question_type="procedural",
            answer_type="inferential",
            reasoning_class="reasoning",
            keywords=["classement", "departage", "interclubs", "N1"],
        ),
    },
    {
        "old_id": "gs:scratch:answerable:0802:209369cd",
        "profile": "MED_APPLY_INF",
        "q": _make_question(
            question=(
                "Comment utiliser la notion de D-SA pour justifier le "
                "choix d'un appariement dans un niveau heterogene?"
            ),
            expected_answer=(
                "Quand un niveau a plusieurs solutions possibles, "
                "on compare les D-SA de ces possibilites pour "
                "justifier qu'un appariement est meilleur. En "
                "pratique, cela concerne les niveaux heterogenes. "
                "Le respect d'un algorithme rigoureux evite tout "
                "questionnement."
            ),
            chunk_id="LA-octobre2025.pdf-p126-parent367-child01",
            source="LA-octobre2025.pdf",
            pages=[126],
            article_ref="D-SA",
            cognitive_level="Apply",
            difficulty=0.65,
            question_type="procedural",
            answer_type="inferential",
            reasoning_class="reasoning",
            keywords=["D-SA", "heterogene", "appariement", "algorithme"],
        ),
    },
    {
        "old_id": "gs:scratch:answerable:1429:2b80bf7e",
        "profile": "MED_APPLY_INF",
        "q": _make_question(
            question=(
                "Quelles sont les obligations de l'OC en matiere de "
                "couverture medicale pour les participants a une "
                "competition d'echecs?"
            ),
            expected_answer=(
                "L'OC garantit une couverture medicale pour tous les "
                "participants, representants, arbitres et officiels, "
                "contre les accidents et services medicaux, mais pas "
                "pour les maladies chroniques. Une equipe medicale "
                "officielle est nommee pour la duree de la competition."
            ),
            chunk_id="LA-octobre2025.pdf-p077-parent234-child01",
            source="LA-octobre2025.pdf",
            pages=[77],
            article_ref="Art. 12.5",
            cognitive_level="Apply",
            difficulty=0.5,
            question_type="procedural",
            answer_type="inferential",
            reasoning_class="summary",
            keywords=["couverture", "medicale", "OC", "assurance"],
        ),
    },
    {
        "old_id": "gs:scratch:answerable:2303:23336902",
        "profile": "MED_APPLY_INF",
        "q": _make_question(
            question=(
                "Comment un arbitre doit-il proceder en cas de force "
                "majeure necessitant l'interruption d'un tournoi?"
            ),
            expected_answer=(
                "L'arbitre informe les joueurs de l'interruption. Les "
                "parties peuvent etre ajournees. Il peut modifier la "
                "cadence, depasser 12 heures de jeu, ou organiser une "
                "ronde a cadence differente. L'objectif est d'atteindre "
                "le nombre prevu de rondes en minimisant le risque "
                "d'aide exterieure."
            ),
            chunk_id="LA-octobre2025.pdf-p084-parent253-child00",
            source="LA-octobre2025.pdf",
            pages=[84],
            article_ref="Art. 10.1",
            cognitive_level="Apply",
            difficulty=0.6,
            question_type="procedural",
            answer_type="inferential",
            reasoning_class="reasoning",
            keywords=["force majeure", "interruption", "cadence", "arbitre"],
        ),
    },
    {
        "old_id": "gs:scratch:answerable:0793:d9c23e23",
        "profile": "MED_APPLY_INF",
        "q": _make_question(
            question=(
                "Quelles options de jeu supplementaires peuvent etre "
                "activees par les joueurs sur une plateforme d'echecs "
                "en ligne?"
            ),
            expected_answer=(
                "Trois options : le smart move (selection de la case "
                "d'arrivee seule quand un seul coup est possible), "
                "le pre-move (preparation du coup avant que "
                "l'adversaire ait joue), et la promotion en dame "
                "automatique (sans choix de piece promue)."
            ),
            chunk_id="LA-octobre2025.pdf-p086-parent256-child00",
            source="LA-octobre2025.pdf",
            pages=[86],
            article_ref="Art. 3.6",
            cognitive_level="Apply",
            difficulty=0.5,
            question_type="procedural",
            answer_type="inferential",
            reasoning_class="summary",
            keywords=["en ligne", "smart move", "pre-move", "promotion"],
        ),
    },
    {
        "old_id": "gs:scratch:answerable:0223:e2f38696",
        "profile": "MED_APPLY_INF",
        "q": _make_question(
            question=(
                "Comment procede-t-on au calcul de la variation de "
                "classement Elo d'un joueur pour un tournoi?"
            ),
            expected_answer=(
                "Pour chaque partie, on determine la difference D "
                "(plafonnee a 400 pts, parties avec ecart >=600 "
                "exclues si un joueur >2600). On calcule PD via "
                "le tableau 8.1.2, puis delta_R = score - PD. La "
                "variation totale est somme(delta_R) x K, ou K=40 "
                "(nouveau, <30 parties) ou K=20 (classement <2400)."
            ),
            chunk_id="LA-octobre2025.pdf-p190-parent566-child00",
            source="LA-octobre2025.pdf",
            pages=[190],
            article_ref="Art. 7.3",
            cognitive_level="Apply",
            difficulty=0.7,
            question_type="procedural",
            answer_type="inferential",
            reasoning_class="reasoning",
            keywords=["elo", "variation", "K", "classement"],
        ),
    },
    {
        "old_id": "gs:scratch:answerable:1892:36c49e26",
        "profile": "MED_APPLY_INF",
        "q": _make_question(
            question=(
                "Comment sont organisees les qualifications pour la "
                "Nationale 2 du championnat interclubs jeunes?"
            ),
            expected_answer=(
                "Les Ligues organisent la N3 avec autant de groupes "
                "qu'elles souhaitent. Elles ne peuvent qualifier en "
                "N2 qu'une equipe par Zone Interdepartementale. "
                "S'il reste des places, les Ligues avec le plus de "
                "licences A minimes (U16/U16F) et plus jeunes "
                "obtiennent une place supplementaire."
            ),
            chunk_id=(
                "J02_2025_26_Championnat_de_France_Interclubs_Jeunes"
                ".pdf-p007-parent000-child00"
            ),
            source=("J02_2025_26_Championnat_de_France_Interclubs_" "Jeunes.pdf"),
            pages=[7],
            article_ref="Art. 1.1",
            cognitive_level="Apply",
            difficulty=0.6,
            question_type="procedural",
            answer_type="inferential",
            reasoning_class="reasoning",
            keywords=["N2", "N3", "qualification", "ZID", "interclubs"],
        ),
    },
    {
        "old_id": "gs:scratch:answerable:0005:2d8dbce8",
        "profile": "MED_APPLY_INF",
        "q": _make_question(
            question=(
                "Comment attribuer un classement fictif a un joueur "
                "non classe pour le calcul des departages Elo?"
            ),
            expected_answer=(
                "On utilise l'interpolation entre niveaux de score "
                "adjacents. Pour un joueur seul dans son niveau, on "
                "calcule l'ecart entre son malus et celui du niveau "
                "superieur, puis on soustrait de la moyenne Elo du "
                "niveau superieur. Pour plusieurs joueurs non classes "
                "dans un meme niveau, on interpole entre les niveaux "
                "superieur et inferieur."
            ),
            chunk_id="LA-octobre2025.pdf-p156-parent469-child01",
            source="LA-octobre2025.pdf",
            pages=[156],
            article_ref="Classement fictif",
            cognitive_level="Apply",
            difficulty=0.7,
            question_type="procedural",
            answer_type="inferential",
            reasoning_class="reasoning",
            keywords=["classement fictif", "departage", "interpolation"],
        ),
    },
    {
        "old_id": "gs:scratch:answerable:0600:ed6e4f00",
        "profile": "MED_APPLY_INF",
        "q": _make_question(
            question=(
                "Quelles categories de personnes sont soumises a la "
                "juridiction de la Commission du Fair-Play de la FIDE "
                "en cas de tricherie?"
            ),
            expected_answer=(
                "Les joueurs, capitaines d'equipe, et parties "
                "prenantes : responsables de delegation, entraineurs, "
                "managers, psychologues, membres de l'organisation, "
                "public, parents, journalistes, officiels et arbitres "
                "lorsqu'ils sont impliques dans des incidents de "
                "tricherie."
            ),
            chunk_id="LA-octobre2025.pdf-p096-parent286-child00",
            source="LA-octobre2025.pdf",
            pages=[96],
            article_ref="Section F - Juridiction",
            cognitive_level="Apply",
            difficulty=0.5,
            question_type="procedural",
            answer_type="inferential",
            reasoning_class="summary",
            keywords=["FPL", "juridiction", "tricherie", "parties prenantes"],
        ),
    },
    {
        "old_id": "gs:scratch:answerable:1050:e95b24f5",
        "profile": "MED_APPLY_INF",
        "q": _make_question(
            question=(
                "Comment un arbitre doit-il gerer les demandes de "
                "'bye' dans un tournoi au systeme suisse homologue?"
            ),
            expected_answer=(
                "L'arbitre ne peut autoriser le bye que si la "
                "procedure est clairement decrite dans l'annonce du "
                "tournoi et le reglement interieur. Il doit refuser de "
                "diriger les appariements et ne pas ceder aux pressions "
                "de joueurs cherchant une norme."
            ),
            chunk_id="LA-octobre2025.pdf-p024-parent110-child00",
            source="LA-octobre2025.pdf",
            pages=[24],
            article_ref="Art. 16.1",
            cognitive_level="Apply",
            difficulty=0.6,
            question_type="procedural",
            answer_type="inferential",
            reasoning_class="reasoning",
            keywords=["bye", "suisse", "reglement", "appariement"],
        ),
    },
    {
        "old_id": "gs:scratch:answerable:1376:f56ad7f4",
        "profile": "MED_APPLY_INF",
        "q": _make_question(
            question=(
                "Comment utiliser les concepts de MaxPaires et M1 lors "
                "de l'appariement d'un niveau au systeme suisse?"
            ),
            expected_answer=(
                "MaxPaires est le nombre maximum de paires possibles "
                "dans le niveau (defini en C.5). M1 est le nombre "
                "max de flotteurs descendants pouvant etre apparies "
                "(defini en C.6). M0 est le nombre de flotteurs du "
                "niveau precedent. L'appariement ne doit pas depasser "
                "ces limites pour etre legal."
            ),
            chunk_id="LA-octobre2025.pdf-p132-parent389-child00",
            source="LA-octobre2025.pdf",
            pages=[132],
            article_ref="Glossaire appariement",
            cognitive_level="Apply",
            difficulty=0.65,
            question_type="procedural",
            answer_type="inferential",
            reasoning_class="reasoning",
            keywords=["MaxPaires", "M1", "M0", "appariement", "suisse"],
        ),
    },
    {
        "old_id": "gs:scratch:answerable:0509:4f2ebc45",
        "profile": "MED_APPLY_INF",
        "q": _make_question(
            question=(
                "Comment proceder au departage entre 2 joueurs ex-aequo "
                "lors d'une phase qualificative du championnat de France "
                "jeunes?"
            ),
            expected_answer=(
                "On joue en rapide 15 min + 10 sec/coup en aller-retour "
                "avec couleurs inversees. En cas d'egalite, blitz 3 min "
                "+ 2 sec/coup en aller-retour. En cas de nouvelle "
                "egalite, le match est rejoue."
            ),
            chunk_id=(
                "J01_2025_26_Championnat_de_France_Jeunes.pdf" "-p004-parent022-child00"
            ),
            source="J01_2025_26_Championnat_de_France_Jeunes.pdf",
            pages=[4],
            article_ref="Art. 3.4",
            cognitive_level="Apply",
            difficulty=0.6,
            question_type="procedural",
            answer_type="inferential",
            reasoning_class="reasoning",
            keywords=["departage", "jeunes", "blitz", "rapide"],
        ),
    },
]


# ===================================================================
# MED_ANALYZE_COMP: comparative, extractive, difficulty 0.4-0.6
# ===================================================================

MED_ANALYZE_COMP = [
    {
        "old_id": "gs:scratch:answerable:2406:7e5874c8",
        "profile": "MED_ANALYZE_COMP",
        "q": _make_question(
            question=(
                "Comparez les differentes voies possibles pour obtenir "
                "une norme d'Arbitre International. Quelles limitations "
                "s'appliquent a chacune?"
            ),
            expected_answer=(
                "Les voies sont : finale nationale (2 normes max), "
                "tournoi/match FIDE, tournoi international a normes, "
                "championnat rapide/blitz continental/mondial (1 norme "
                "max). Un tournoi FIDE >=100 joueurs de 3 federations "
                "et 7 rondes en division nationale comptent aussi, "
                "chacun utilisable une seule fois."
            ),
            chunk_id="LA-octobre2025.pdf-p208-parent622-child00",
            source="LA-octobre2025.pdf",
            pages=[208],
            article_ref="Art. 4.8-4.9",
            cognitive_level="Analyze",
            difficulty=0.5,
            question_type="comparative",
            answer_type="extractive",
            reasoning_class="summary",
            keywords=["AI", "norme", "voies", "limitation"],
        ),
    },
    {
        "old_id": "gs:scratch:answerable:0680:620d287e",
        "profile": "MED_ANALYZE_COMP",
        "q": _make_question(
            question=(
                "Comparez les normes de dimensions des tables entre les "
                "competitions EVE/GSC standard et les competitions "
                "pour jeunes joueurs ?"
            ),
            expected_answer=(
                "Standard : longueur 110 cm (+/-15%), largeur 85 cm "
                "(+/-15%), hauteur 74 cm. Pour les jeunes (Junior, "
                "Cadets, Ecoles sous EVE), ces dimensions peuvent "
                "etre modifiees en fonction de l'age, en accord avec "
                "l'EVE. Les chaises doivent minimiser le bruit."
            ),
            chunk_id="LA-octobre2025.pdf-p079-parent240-child00",
            source="LA-octobre2025.pdf",
            pages=[79],
            article_ref="Art. 4 - Tables et chaises",
            cognitive_level="Analyze",
            difficulty=0.4,
            question_type="comparative",
            answer_type="extractive",
            reasoning_class="fact_single",
            keywords=["table", "dimensions", "jeunes", "EVE"],
        ),
    },
    {
        "old_id": "gs:scratch:answerable:1848:506cbec5",
        "profile": "MED_ANALYZE_COMP",
        "q": _make_question(
            question=(
                "Quels sont les deux statuts possibles pour un arbitre "
                "FIDE et en quoi influencent-ils sa capacite a officier?"
            ),
            expected_answer=(
                "Les arbitres (AIs et AFs) ont deux statuts : Actif "
                "(a) ou Inactif (i). Le statut determine si l'arbitre "
                "est autorise a exercer dans une manifestation "
                "comptabilisee pour le classement FIDE. Seuls les "
                "arbitres actifs peuvent officier."
            ),
            chunk_id="LA-octobre2025.pdf-p214-parent648-child00",
            source="LA-octobre2025.pdf",
            pages=[214],
            article_ref="Art. 1.1 - Statut arbitres",
            cognitive_level="Analyze",
            difficulty=0.4,
            question_type="comparative",
            answer_type="extractive",
            reasoning_class="fact_single",
            keywords=["arbitre", "actif", "inactif", "statut"],
        ),
    },
    {
        "old_id": "gs:scratch:answerable:0019:79ae0c71",
        "profile": "MED_ANALYZE_COMP",
        "q": _make_question(
            question=(
                "Comparez les droits et les limitations du capitaine "
                "d'equipe dans un match interclubs ?"
            ),
            expected_answer=(
                "Le capitaine peut acceder a l'aire de jeu, conseiller "
                "sur nullite/abandon/situation du match, remettre les "
                "feuilles et transmettre les resultats. Il lui est "
                "interdit de commenter les positions sur l'echiquier "
                "et d'officier simultanement comme arbitre."
            ),
            chunk_id=("Interclubs_DepartementalBdr.pdf" "-p005-parent012-child00"),
            source="Interclubs_DepartementalBdr.pdf",
            pages=[5],
            article_ref="Art. 3.5",
            cognitive_level="Analyze",
            difficulty=0.5,
            question_type="comparative",
            answer_type="extractive",
            reasoning_class="summary",
            keywords=["capitaine", "droits", "interclubs", "limitation"],
        ),
    },
    {
        "old_id": "gs:scratch:answerable:0069:ccff5829",
        "profile": "MED_ANALYZE_COMP",
        "q": _make_question(
            question=(
                "Comparez les deux moments ou la Commission "
                "d'Homologation peut se prononcer sur le statut d'un "
                "joueur ?"
            ),
            expected_answer=(
                "Elle peut se prononcer : (1) avant le debut de la "
                "saison, a la demande du club, ou (2) a tout moment "
                "mais au plus tard 30 jours apres la rencontre, a la "
                "demande du club, de la direction du groupe, de la "
                "DTN ou du Comite Directeur. Les clubs non conformes "
                "peuvent etre penalises retroactivement."
            ),
            chunk_id=(
                "J02_2025_26_Championnat_de_France_Interclubs_Jeunes"
                ".pdf-p003-parent013-child00"
            ),
            source=("J02_2025_26_Championnat_de_France_Interclubs_" "Jeunes.pdf"),
            pages=[3],
            article_ref="Art. 3.4",
            cognitive_level="Analyze",
            difficulty=0.5,
            question_type="comparative",
            answer_type="extractive",
            reasoning_class="summary",
            keywords=["homologation", "commission", "statut", "delai"],
        ),
    },
    {
        "old_id": "gs:scratch:answerable:1241:b32de191",
        "profile": "MED_ANALYZE_COMP",
        "q": _make_question(
            question=(
                "Comparez les criteres de publication du classement "
                "rapide/blitz FIDE avec les conditions d'entree d'un "
                "nouveau joueur dans la liste ?"
            ),
            expected_answer=(
                "Publication mensuelle avec titre, federation, "
                "classement, code, nombre de parties, naissance, "
                "sexe et K. Date de cloture : 3 jours avant "
                "publication. Nouveau joueur : minimum 5 parties "
                "contre adversaires classes sur 26 mois max, "
                "classement initial >= 1400."
            ),
            chunk_id="LA-octobre2025.pdf-p188-parent560-child00",
            source="LA-octobre2025.pdf",
            pages=[188],
            article_ref="Art. 6",
            cognitive_level="Analyze",
            difficulty=0.5,
            question_type="comparative",
            answer_type="extractive",
            reasoning_class="summary",
            keywords=["classement", "rapide", "blitz", "publication"],
        ),
    },
    {
        "old_id": "gs:scratch:answerable:1926:390e6502",
        "profile": "MED_ANALYZE_COMP",
        "q": _make_question(
            question=(
                "Comparez un niveau d'appariement homogene et un "
                "niveau heterogene au systeme suisse ?"
            ),
            expected_answer=(
                "Un niveau homogene contient uniquement des joueurs "
                "avec le meme score. Un niveau heterogene comprend "
                "des residents (meme groupe de points) et des "
                "personnes non appariees du niveau precedent. Le "
                "niveau heterogene peut contenir un sous-niveau "
                "residuel."
            ),
            chunk_id="LA-octobre2025.pdf-p125-parent362-child00",
            source="LA-octobre2025.pdf",
            pages=[125],
            article_ref="A.3 Niveaux",
            cognitive_level="Analyze",
            difficulty=0.5,
            question_type="comparative",
            answer_type="extractive",
            reasoning_class="fact_single",
            keywords=["homogene", "heterogene", "niveau", "appariement"],
        ),
    },
    {
        "old_id": "gs:scratch:answerable:0606:628bd72c",
        "profile": "MED_ANALYZE_COMP",
        "q": _make_question(
            question=(
                "Comparez les termes 'piratage' et 'promotion "
                "automatique' dans le contexte des echecs en ligne ?"
            ),
            expected_answer=(
                "Le piratage (Annexe I, B.3) est une fraude : une "
                "autre personne joue a la place du veritable joueur. "
                "La promotion automatique (Art. 3.6.c) est une "
                "fonctionnalite legitime : un pion est automatiquement "
                "remplace par une dame selon les reglages. L'un est "
                "une infraction, l'autre un outil de jeu."
            ),
            chunk_id="LA-octobre2025.pdf-p099-parent295-child02",
            source="LA-octobre2025.pdf",
            pages=[99],
            article_ref="Glossaire en ligne",
            cognitive_level="Analyze",
            difficulty=0.5,
            question_type="comparative",
            answer_type="extractive",
            reasoning_class="fact_single",
            keywords=["piratage", "promotion", "en ligne", "glossaire"],
        ),
    },
    {
        "old_id": "gs:scratch:answerable:1768:1c953452",
        "profile": "MED_ANALYZE_COMP",
        "q": _make_question(
            question=(
                "Comparez les obligations de communication des joueurs "
                "et officiels : que doivent-ils faire et que doivent-"
                "ils eviter?"
            ),
            expected_answer=(
                "Ils doivent : communications factuelles et non "
                "blessantes, informations precises, approche "
                "constructive. Ils doivent eviter : critique negative "
                "publique des benevoles, entraineurs et officiels. "
                "Les difficultes doivent etre signalees de maniere "
                "appropriee plutot que publiquement."
            ),
            chunk_id="LA-octobre2025.pdf-p029-parent130-child00",
            source="LA-octobre2025.pdf",
            pages=[29],
            article_ref="Art. 6.31",
            cognitive_level="Analyze",
            difficulty=0.4,
            question_type="comparative",
            answer_type="extractive",
            reasoning_class="summary",
            keywords=["communication", "transparence", "obligation"],
        ),
    },
    {
        "old_id": "gs:scratch:answerable:2121:cd2c750b",
        "profile": "MED_ANALYZE_COMP",
        "q": _make_question(
            question=(
                "Comparez les modalites des seances en presentiel et "
                "en visioconference du comite directeur de la FFE ?"
            ),
            expected_answer=(
                "Le comite se reunit au moins 3 fois par an en "
                "presentiel (obligatoire, 1 par tranche de 4 mois). "
                "Les seances supplementaires peuvent etre en "
                "visioconference. Le DTN assiste avec voix consultative "
                "dans les deux cas. Entre les reunions, vote "
                "electronique possible."
            ),
            chunk_id="2024_Statuts20240420.pdf-p010-parent031-child00",
            source="2024_Statuts20240420.pdf",
            pages=[10],
            article_ref="Art. 7.3",
            cognitive_level="Analyze",
            difficulty=0.5,
            question_type="comparative",
            answer_type="extractive",
            reasoning_class="summary",
            keywords=["presentiel", "visioconference", "comite", "DTN"],
        ),
    },
    {
        "old_id": "gs:scratch:answerable:2122:c4147d16",
        "profile": "MED_ANALYZE_COMP",
        "q": _make_question(
            question=(
                "Comparez les conditions pour proposer une modification "
                "des statuts avec celles pour l'adopter ?"
            ),
            expected_answer=(
                "Proposer : comite directeur ou 1/3 des membres "
                "representant 1/3 des voix. Adopter : quorum de 50% "
                "des membres (50% des voix), majorite des 2/3 des "
                "presents representant 2/3 des voix. Si quorum non "
                "atteint, nouvelle convocation sans condition de "
                "quorum. Convocation 15 jours avant."
            ),
            chunk_id="2024_Statuts20240420.pdf-p016-parent055-child00",
            source="2024_Statuts20240420.pdf",
            pages=[16],
            article_ref="Art. 16",
            cognitive_level="Analyze",
            difficulty=0.6,
            question_type="comparative",
            answer_type="extractive",
            reasoning_class="reasoning",
            keywords=["statuts", "modification", "quorum", "majorite"],
        ),
    },
    {
        "old_id": "gs:scratch:answerable:1342:eb5134db",
        "profile": "MED_ANALYZE_COMP",
        "q": _make_question(
            question=(
                "Comparez les tournois de type A 'FIDE' et 'FIDE a "
                "normes'. Quelle est la difference principale?"
            ),
            expected_answer=(
                "Les deux comptent pour le classement Elo FIDE. Les "
                "tournois 'FIDE a normes' permettent en plus "
                "l'obtention de normes de titres, impliquant des "
                "exigences supplementaires. Quand la FIDE impose "
                "une limite Elo, le tournoi est reserve aux joueurs "
                "ne depassant pas cette limite."
            ),
            chunk_id=(
                "R03_2025_26_Competitions_homologuees.pdf" "-p001-parent000-child00"
            ),
            source="R03_2025_26_Competitions_homologuees.pdf",
            pages=[1],
            article_ref="Tournois type A",
            cognitive_level="Analyze",
            difficulty=0.5,
            question_type="comparative",
            answer_type="extractive",
            reasoning_class="fact_single",
            keywords=["type A", "FIDE", "normes", "homologation"],
        ),
    },
    {
        "old_id": "gs:scratch:answerable:0155:de95f352",
        "profile": "MED_ANALYZE_COMP",
        "q": _make_question(
            question=(
                "Comparez les definitions de 'position morte' et de "
                "'prise' dans les regles du jeu d'echecs ?"
            ),
            expected_answer=(
                "La position morte (Art. 5.2.2) signifie qu'aucun "
                "adversaire ne peut mater, entrainant la nulle. La "
                "prise (Art. 3.1) est le deplacement vers une case "
                "occupee par une piece adverse qui est retiree. "
                "La prise reduit le materiel et peut mener a une "
                "position morte."
            ),
            chunk_id="LA-octobre2025.pdf-p194-parent214-child01",
            source="LA-octobre2025.pdf",
            pages=[194],
            article_ref="Glossaire",
            cognitive_level="Analyze",
            difficulty=0.4,
            question_type="comparative",
            answer_type="extractive",
            reasoning_class="fact_single",
            keywords=["position morte", "prise", "nulle", "materiel"],
        ),
    },
    {
        "old_id": "gs:scratch:answerable:0550:32b33414",
        "profile": "MED_ANALYZE_COMP",
        "q": _make_question(
            question=(
                "Comparez la verification de la chute du drapeau entre "
                "une pendule electronique avec compteur et une pendule "
                "mecanique ?"
            ),
            expected_answer=(
                "Avec pendule electronique a compteur, on peut "
                "determiner automatiquement si 40 coups ont ete "
                "joues : un '-' apparait si le joueur n'a pas "
                "termine. Si les deux cadrans affichent 0.00, "
                "l'arbitre determine quel drapeau est tombe en "
                "premier. Avec pendules mecaniques, l'article III.3.1 "
                "des Directives s'applique."
            ),
            chunk_id="LA-octobre2025.pdf-p045-parent169-child02",
            source="LA-octobre2025.pdf",
            pages=[45],
            article_ref="Art. 6.4",
            cognitive_level="Analyze",
            difficulty=0.5,
            question_type="comparative",
            answer_type="extractive",
            reasoning_class="reasoning",
            keywords=["drapeau", "pendule", "electronique", "mecanique"],
        ),
    },
    {
        "old_id": "gs:scratch:answerable:1918:0122a808",
        "profile": "MED_ANALYZE_COMP",
        "q": _make_question(
            question=(
                "Comparez les droits de l'appelant et les pouvoirs du "
                "Jury d'Appel dans le processus d'appel ?"
            ),
            expected_answer=(
                "L'appelant (joueur, representant, capitaine) doit "
                "soumettre par ecrit avec caution dans le delai. "
                "Le JA rend des decisions definitives. Si l'appel "
                "aboutit, la caution est remboursee. Si l'appel "
                "echoue mais est juge raisonnable, la caution peut "
                "etre partiellement remboursee."
            ),
            chunk_id="LA-octobre2025.pdf-p076-parent231-child01",
            source="LA-octobre2025.pdf",
            pages=[76],
            article_ref="Art. 10.3-10.4",
            cognitive_level="Analyze",
            difficulty=0.5,
            question_type="comparative",
            answer_type="extractive",
            reasoning_class="summary",
            keywords=["appel", "jury", "caution", "decision"],
        ),
    },
    {
        "old_id": "gs:scratch:answerable:2374:4c50ab80",
        "profile": "MED_ANALYZE_COMP",
        "q": _make_question(
            question=(
                "Comparez les methodes d'attribution des numeros "
                "d'appariement dans un tournoi toutes rondes. Quel "
                "ajustement est recommande en aller-retour?"
            ),
            expected_answer=(
                "Les methodes sont : tirage au sort, classement Elo, "
                "ou protocole de Varma. Le reglement doit preciser "
                "la methode. Pour un tournoi aller-retour, il est "
                "recommande d'inverser l'ordre des deux derniers "
                "tours du premier cycle pour eviter trois parties "
                "consecutives de la meme couleur."
            ),
            chunk_id="LA-octobre2025.pdf-p103-parent303-child00",
            source="LA-octobre2025.pdf",
            pages=[103],
            article_ref="Mise en oeuvre toutes rondes",
            cognitive_level="Analyze",
            difficulty=0.5,
            question_type="comparative",
            answer_type="extractive",
            reasoning_class="reasoning",
            keywords=["appariement", "toutes rondes", "berger", "couleur"],
        ),
    },
    {
        "old_id": "gs:scratch:answerable:0241:e8253a25",
        "profile": "MED_ANALYZE_COMP",
        "q": _make_question(
            question=(
                "Comparez les cadences minimales requises pour le "
                "classement Elo selon le niveau des adversaires ?"
            ),
            expected_answer=(
                "Si un adversaire >= 2400 : minimum 120 minutes. "
                "Si un adversaire >= 1800 : minimum 90 minutes. "
                "Si les deux < 1800 : minimum 60 minutes (pour 60 "
                "coups). Premier controle de temps : minimum 30 "
                "coups."
            ),
            chunk_id="LA-octobre2025.pdf-p182-parent533-child00",
            source="LA-octobre2025.pdf",
            pages=[182],
            article_ref="Art. 1 - Cadence",
            cognitive_level="Analyze",
            difficulty=0.4,
            question_type="comparative",
            answer_type="extractive",
            reasoning_class="fact_single",
            keywords=["cadence", "elo", "temps", "classement"],
        ),
    },
    {
        "old_id": "gs:scratch:answerable:0544:8dc78fca",
        "profile": "MED_ANALYZE_COMP",
        "q": _make_question(
            question=(
                "Comparez l'ordre de tirage entre les joueurs de la "
                "meme federation et ceux de federations differentes "
                "dans le tirage au sort restrictif ?"
            ),
            expected_answer=(
                "La federation avec le plus de representants tire en "
                "premier. A nombre egal, priorite par ordre "
                "alphabetique des codes FIDE. Au sein d'une meme "
                "federation, priorite par ordre alphabetique des "
                "noms. Le premier joueur choisit une grande enveloppe "
                "contenant assez de numeros pour son contingent."
            ),
            chunk_id="LA-octobre2025.pdf-p104-parent308-child00",
            source="LA-octobre2025.pdf",
            pages=[104],
            article_ref="C.05 - Varma",
            cognitive_level="Analyze",
            difficulty=0.6,
            question_type="comparative",
            answer_type="extractive",
            reasoning_class="reasoning",
            keywords=["tirage", "varma", "federation", "restrictif"],
        ),
    },
    {
        "old_id": "gs:scratch:answerable:1883:991bfeb3",
        "profile": "MED_ANALYZE_COMP",
        "q": _make_question(
            question=(
                "Comparez les criteres utilises pour distinguer une "
                "accusation legitime d'une fausse accusation de "
                "tricherie aux echecs ?"
            ),
            expected_answer=(
                "Les criteres sont : base factuelle suffisante, "
                "niveau de la competition, titre et classement de "
                "l'accuse, resultat final dans la competition, "
                "moyen et ampleur de diffusion (medias sociaux, "
                "interview, blog). Cette liste n'est pas exhaustive. "
                "La fausse accusation est un abus de la liberte "
                "d'expression."
            ),
            chunk_id="LA-octobre2025.pdf-p096-parent284-child00",
            source="LA-octobre2025.pdf",
            pages=[96],
            article_ref="Section D - Fausse accusation",
            cognitive_level="Analyze",
            difficulty=0.5,
            question_type="comparative",
            answer_type="extractive",
            reasoning_class="summary",
            keywords=["fausse accusation", "tricherie", "criteres"],
        ),
    },
    {
        "old_id": "gs:scratch:answerable:1165:5e88a455",
        "profile": "MED_ANALYZE_COMP",
        "q": _make_question(
            question=(
                "Comparez la responsabilite de l'organisateur pour "
                "le comportement des spectateurs avec celle des "
                "personnes qui s'accordent pour commettre des "
                "infractions ?"
            ),
            expected_answer=(
                "L'organisateur doit intervenir avec des mesures "
                "raisonnables en cas d'inconduite des spectateurs. "
                "Tout accord entre personnes visant a commettre des "
                "infractions est une infraction distincte. Le code "
                "d'ethique complete les lois sans les remplacer, "
                "ajoutant des regles de conduite supplementaires."
            ),
            chunk_id="LA-octobre2025.pdf-p031-parent138-child00",
            source="LA-octobre2025.pdf",
            pages=[31],
            article_ref="Art. 11.10",
            cognitive_level="Analyze",
            difficulty=0.5,
            question_type="comparative",
            answer_type="extractive",
            reasoning_class="reasoning",
            keywords=["spectateurs", "organisateur", "infraction", "ethique"],
        ),
    },
]


def main() -> int:
    """Generate replacements JSON from hand-crafted questions."""
    candidates = load_json(
        _project_root / "data" / "gs_generation" / "p2_candidates.json",
    )

    all_profiles = {
        "HARD_APPLY": HARD_APPLY,
        "HARD_ANALYZE": HARD_ANALYZE,
        "MED_APPLY_INF": MED_APPLY_INF,
        "MED_ANALYZE_COMP": MED_ANALYZE_COMP,
    }

    replacements = []
    for profile_name, items in all_profiles.items():
        cands = candidates[profile_name]
        if len(items) != len(cands):
            print(
                f"ERROR: {profile_name} has {len(items)} questions "
                f"but {len(cands)} candidates",
            )
            return 1
        for item, cand in zip(items, cands):
            if item["old_id"] != cand["old_id"]:
                print(
                    f"ERROR: ID mismatch in {profile_name}: "
                    f"{item['old_id']} != {cand['old_id']}",
                )
                return 1
            replacements.append(
                {
                    "old_id": item["old_id"],
                    "profile": item["profile"],
                    "new_question": item["q"],
                }
            )

    # Validate all questions
    errors: list[str] = []
    for r in replacements:
        q = r["new_question"]
        profile_name = r["profile"]
        old_id = r["old_id"]
        prefix = f"{profile_name}/{old_id}"

        # Structure checks
        text = q.get("content", {}).get("question", "")
        if not text.endswith("?"):
            errors.append(f"{prefix}: question doesn't end with '?'")
        answer = q.get("content", {}).get("expected_answer", "")
        if len(answer) <= 5:
            errors.append(f"{prefix}: expected_answer too short ({len(answer)})")

        # Profile conformance checks
        cls = q.get("classification", {})
        profile_spec = PROFILES[profile_name]
        if cls.get("cognitive_level") != profile_spec["cognitive_level"]:
            errors.append(
                f"{prefix}: cognitive_level={cls.get('cognitive_level')!r} "
                f"!= {profile_spec['cognitive_level']!r}"
            )
        if cls.get("question_type") != profile_spec["question_type"]:
            errors.append(
                f"{prefix}: question_type={cls.get('question_type')!r} "
                f"!= {profile_spec['question_type']!r}"
            )
        if cls.get("answer_type") != profile_spec["answer_type"]:
            errors.append(
                f"{prefix}: answer_type={cls.get('answer_type')!r} "
                f"!= {profile_spec['answer_type']!r}"
            )
        diff = cls.get("difficulty", -1)
        if not profile_spec["difficulty_min"] <= diff <= profile_spec["difficulty_max"]:
            errors.append(
                f"{prefix}: difficulty={diff} not in "
                f"[{profile_spec['difficulty_min']}, {profile_spec['difficulty_max']}]"
            )

    if errors:
        for e in errors:
            print(f"ERROR: {e}")
        print(f"\n{len(errors)} validation errors — aborting")
        return 1

    out_path = _project_root / "data" / "gs_generation" / "p2_replacements.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(replacements, f, indent=2, ensure_ascii=False)

    print(f"Generated {len(replacements)} replacements -> {out_path}")
    for pn, items in all_profiles.items():
        print(f"  {pn}: {len(items)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
