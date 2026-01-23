#!/usr/bin/env python3
"""
Crée des questions VALIDÉES pour le Gold Standard FR.
Chaque question est vérifiée contre le corpus avant ajout.
"""

import json
from pathlib import Path

def load_data():
    """Charge le GS et les chunks."""
    with open("corpus/processed/chunks_mode_b_fr.json", encoding="utf-8") as f:
        chunks_data = json.load(f)
    with open("tests/data/gold_standard_fr.json", encoding="utf-8") as f:
        gs = json.load(f)
    return gs, chunks_data

def build_index(chunks_data):
    """Index chunks par (source, page)."""
    index = {}
    for c in chunks_data["chunks"]:
        key = (c["source"], c["page"])
        if key not in index:
            index[key] = ""
        index[key] += " " + c["text"].lower()
    return index

def validate_question(index, source, pages, keywords):
    """Vérifie que les keywords existent sur les pages."""
    for page in pages:
        text = index.get((source, page), "")
        found = [kw for kw in keywords if kw.lower() in text]
        if len(found) >= 2:
            return True, found
    return False, []

def get_validated_questions():
    """Retourne toutes les questions validées."""
    return [
        # LA-octobre2025.pdf - Règles du jeu
        {"q": "Comment fonctionne le déplacement du pion ?", "s": "LA-octobre2025.pdf", "p": [39], "k": ["pion", "déplacer", "case", "colonne"], "cat": "regles_jeu", "at": "PROCEDURAL", "cl": "UNDERSTAND", "rt": "MULTI_SENTENCE"},
        {"q": "Comment fonctionne la prise en passant aux échecs ?", "s": "LA-octobre2025.pdf", "p": [39], "k": ["prise", "passant", "pion", "avance"], "cat": "regles_jeu", "at": "PROCEDURAL", "cl": "UNDERSTAND", "rt": "MULTI_SENTENCE"},
        {"q": "Comment fonctionne la promotion du pion ?", "s": "LA-octobre2025.pdf", "p": [40], "k": ["promotion", "pion", "pièce", "remplacement"], "cat": "regles_jeu", "at": "PROCEDURAL", "cl": "UNDERSTAND", "rt": "SINGLE_SENTENCE"},
        {"q": "Quand le roque est-il définitivement interdit ?", "s": "LA-octobre2025.pdf", "p": [40], "k": ["roque", "interdit", "roi", "tour"], "cat": "regles_jeu", "at": "LIST", "cl": "REMEMBER", "rt": "MULTI_SENTENCE"},
        {"q": "Quand le roque est-il temporairement interdit ?", "s": "LA-octobre2025.pdf", "p": [40], "k": ["roque", "temporairement", "attaqué", "pièce"], "cat": "regles_jeu", "at": "LIST", "cl": "REMEMBER", "rt": "MULTI_SENTENCE"},
        {"q": "Que signifie la chute du drapeau aux échecs ?", "s": "LA-octobre2025.pdf", "p": [44], "k": ["chute", "drapeau", "temps", "fin"], "cat": "temps", "at": "FACTUAL", "cl": "REMEMBER", "rt": "LEXICAL_MATCH"},
        {"q": "Comment fonctionne la pendule aux échecs ?", "s": "LA-octobre2025.pdf", "p": [44], "k": ["pendule", "horloge", "afficheurs", "drapeau"], "cat": "temps", "at": "PROCEDURAL", "cl": "UNDERSTAND", "rt": "MULTI_SENTENCE"},
        {"q": "Comment se déplace la tour aux échecs ?", "s": "LA-octobre2025.pdf", "p": [38], "k": ["tour", "déplacer", "colonne", "rangée"], "cat": "regles_jeu", "at": "PROCEDURAL", "cl": "REMEMBER", "rt": "SINGLE_SENTENCE"},
        {"q": "Comment se déplace la dame aux échecs ?", "s": "LA-octobre2025.pdf", "p": [38], "k": ["dame", "déplacer", "colonne", "diagonale"], "cat": "regles_jeu", "at": "PROCEDURAL", "cl": "REMEMBER", "rt": "SINGLE_SENTENCE"},
        {"q": "Quelle est la position initiale des pièces aux échecs ?", "s": "LA-octobre2025.pdf", "p": [37], "k": ["position", "initiale", "pièces", "échiquier"], "cat": "regles_jeu", "at": "PROCEDURAL", "cl": "REMEMBER", "rt": "MULTI_SENTENCE"},
        {"q": "Quand considère-t-on que le roi est en échec ?", "s": "LA-octobre2025.pdf", "p": [41], "k": ["roi", "échec", "attaqué", "pièce"], "cat": "regles_jeu", "at": "FACTUAL", "cl": "REMEMBER", "rt": "SINGLE_SENTENCE"},
        {"q": "Comment gagne-t-on une partie aux échecs ?", "s": "LA-octobre2025.pdf", "p": [43], "k": ["gagné", "mat", "roi", "partie"], "cat": "regles_jeu", "at": "FACTUAL", "cl": "REMEMBER", "rt": "SINGLE_SENTENCE"},
        {"q": "Comment abandonner une partie aux échecs ?", "s": "LA-octobre2025.pdf", "p": [43], "k": ["abandonner", "perdue", "partie", "fin"], "cat": "regles_jeu", "at": "PROCEDURAL", "cl": "UNDERSTAND", "rt": "SINGLE_SENTENCE"},
        {"q": "Comment fonctionne la cadence Fischer ?", "s": "LA-octobre2025.pdf", "p": [45], "k": ["Fischer", "incrément", "temps", "coup"], "cat": "temps", "at": "PROCEDURAL", "cl": "UNDERSTAND", "rt": "MULTI_SENTENCE"},
        {"q": "Comment fonctionne le temps différé Bronstein ?", "s": "LA-octobre2025.pdf", "p": [45], "k": ["Bronstein", "différé", "temps", "délai"], "cat": "temps", "at": "PROCEDURAL", "cl": "UNDERSTAND", "rt": "MULTI_SENTENCE"},
        {"q": "Quels sont les engagements de l'arbitre FFE ?", "s": "LA-octobre2025.pdf", "p": [8], "k": ["arbitre", "engage", "règles", "compétition"], "cat": "arbitrage", "at": "LIST", "cl": "REMEMBER", "rt": "MULTI_SENTENCE"},
        {"q": "Quels frais sont inclus dans le défraiement des arbitres ?", "s": "LA-octobre2025.pdf", "p": [10], "k": ["défraiement", "arbitre", "indemnités", "frais"], "cat": "arbitrage", "at": "LIST", "cl": "REMEMBER", "rt": "MULTI_SENTENCE"},
        {"q": "Quel système de notation est reconnu par la FIDE ?", "s": "LA-octobre2025.pdf", "p": [59], "k": ["FIDE", "notation", "algébrique", "système"], "cat": "notation", "at": "FACTUAL", "cl": "REMEMBER", "rt": "LEXICAL_MATCH"},
        {"q": "Quelles adaptations pour les joueurs handicapés visuels ?", "s": "LA-octobre2025.pdf", "p": [61], "k": ["aveugle", "échiquier", "spécial", "handicap"], "cat": "handicap", "at": "LIST", "cl": "REMEMBER", "rt": "MULTI_SENTENCE"},
        {"q": "Comment fonctionne l'échiquier pour personnes aveugles ?", "s": "LA-octobre2025.pdf", "p": [61], "k": ["échiquier", "handicap", "visuel", "cases"], "cat": "handicap", "at": "PROCEDURAL", "cl": "UNDERSTAND", "rt": "MULTI_SENTENCE"},
        # Discipline
        {"q": "Les débats disciplinaires de la FFE sont-ils publics ?", "s": "2018_Reglement_Disciplinaire20180422.pdf", "p": [3], "k": ["débats", "public", "audience", "salle"], "cat": "discipline", "at": "FACTUAL", "cl": "REMEMBER", "rt": "SINGLE_SENTENCE"},
        {"q": "Quel délai pour la convocation à l'audience disciplinaire ?", "s": "2018_Reglement_Disciplinaire20180422.pdf", "p": [4, 5], "k": ["convocation", "audience", "dix", "jours"], "cat": "discipline", "at": "FACTUAL", "cl": "REMEMBER", "rt": "SINGLE_SENTENCE"},
        {"q": "Quelles obligations de confidentialité en discipline FFE ?", "s": "2018_Reglement_Disciplinaire20180422.pdf", "p": [4], "k": ["confidentialité", "faits", "informations", "fonctions"], "cat": "discipline", "at": "LIST", "cl": "REMEMBER", "rt": "MULTI_SENTENCE"},
        {"q": "Comment se déroule l'instruction disciplinaire FFE ?", "s": "2018_Reglement_Disciplinaire20180422.pdf", "p": [4], "k": ["instruction", "rapport", "organe", "disciplinaire"], "cat": "discipline", "at": "PROCEDURAL", "cl": "UNDERSTAND", "rt": "MULTI_SENTENCE"},
        {"q": "Qu'est-ce qu'une mesure conservatoire en discipline FFE ?", "s": "2018_Reglement_Disciplinaire20180422.pdf", "p": [4], "k": ["mesure", "conservatoire", "Bureau", "Fédéral"], "cat": "discipline", "at": "PROCEDURAL", "cl": "UNDERSTAND", "rt": "MULTI_SENTENCE"},
        # Règles générales
        {"q": "Quand débute et se termine la saison FFE ?", "s": "R01_2025_26_Regles_generales.pdf", "p": [1], "k": ["saison", "septembre", "août", "débute"], "cat": "tournoi", "at": "FACTUAL", "cl": "REMEMBER", "rt": "SINGLE_SENTENCE"},
        {"q": "Quelles sont les compétitions fédérales FFE ?", "s": "R01_2025_26_Regles_generales.pdf", "p": [1], "k": ["compétitions", "fédérales", "Coupes", "Championnats"], "cat": "tournoi", "at": "LIST", "cl": "REMEMBER", "rt": "MULTI_SENTENCE"},
        {"q": "Qu'est-ce qu'un joueur muté à la FFE ?", "s": "R01_2025_26_Regles_generales.pdf", "p": [2], "k": ["muté", "joueur", "club", "licencié"], "cat": "tournoi", "at": "PROCEDURAL", "cl": "UNDERSTAND", "rt": "MULTI_SENTENCE"},
        {"q": "Qu'est-ce qu'un forfait administratif à la FFE ?", "s": "R01_2025_26_Regles_generales.pdf", "p": [3], "k": ["forfait", "administratif", "sanction", "sportif"], "cat": "tournoi", "at": "PROCEDURAL", "cl": "UNDERSTAND", "rt": "MULTI_SENTENCE"},
        {"q": "Qui peut prononcer un forfait administratif FFE ?", "s": "R01_2025_26_Regles_generales.pdf", "p": [3], "k": ["forfait", "administratif", "direction", "groupe"], "cat": "tournoi", "at": "FACTUAL", "cl": "REMEMBER", "rt": "SINGLE_SENTENCE"},
        # Jeunes
        {"q": "Qui peut participer aux Championnats de France Jeunes ?", "s": "J01_2025_26_Championnat_de_France_Jeunes.pdf", "p": [1], "k": ["jeunes", "français", "licenciés", "scolaris"], "cat": "jeunes", "at": "LIST", "cl": "REMEMBER", "rt": "MULTI_SENTENCE"},
        {"q": "Quelles conditions de licence pour jeunes étrangers aux championnats ?", "s": "J01_2025_26_Championnat_de_France_Jeunes.pdf", "p": [1], "k": ["étrangers", "licences", "novembre", "FFE"], "cat": "jeunes", "at": "PROCEDURAL", "cl": "UNDERSTAND", "rt": "MULTI_SENTENCE"},
        {"q": "Comment fonctionne la phase zone interdépartementale jeunes ?", "s": "J01_2025_26_Championnat_de_France_Jeunes.pdf", "p": [2], "k": ["Ligue", "règlement", "ZID", "qualification"], "cat": "jeunes", "at": "PROCEDURAL", "cl": "UNDERSTAND", "rt": "MULTI_SENTENCE"},
        {"q": "Quelles qualifications d'office aux Championnats Jeunes ?", "s": "J01_2025_26_Championnat_de_France_Jeunes.pdf", "p": [3], "k": ["qualifiés", "score", "catégorie", "Elo"], "cat": "jeunes", "at": "LIST", "cl": "REMEMBER", "rt": "MULTI_SENTENCE"},
        {"q": "Quel score minimal pour qualification d'office jeunes ?", "s": "J01_2025_26_Championnat_de_France_Jeunes.pdf", "p": [3], "k": ["score", "minimal", "championnat", "précédent"], "cat": "jeunes", "at": "FACTUAL", "cl": "REMEMBER", "rt": "SINGLE_SENTENCE"},
        # Championnat de France
        {"q": "Quels tournois au Championnat de France individuel ?", "s": "A01_2025_26_Championnat_de_France.pdf", "p": [1], "k": ["tournois", "National", "Accession", "Open"], "cat": "tournoi", "at": "LIST", "cl": "REMEMBER", "rt": "MULTI_SENTENCE"},
        {"q": "Combien de joueuses au Tournoi National Féminin ?", "s": "A01_2025_26_Championnat_de_France.pdf", "p": [2], "k": ["National", "Féminin", "16", "joueuses"], "cat": "feminin", "at": "FACTUAL", "cl": "REMEMBER", "rt": "LEXICAL_MATCH"},
        {"q": "Comment se qualifier au Tournoi National Féminin ?", "s": "A01_2025_26_Championnat_de_France.pdf", "p": [2], "k": ["qualifiées", "National", "Féminin", "Elo"], "cat": "feminin", "at": "LIST", "cl": "REMEMBER", "rt": "MULTI_SENTENCE"},
        {"q": "Quel âge minimum pour le tournoi Vétérans FFE ?", "s": "A01_2025_26_Championnat_de_France.pdf", "p": [3], "k": ["Vétérans", "65", "ans", "janvier"], "cat": "tournoi", "at": "FACTUAL", "cl": "REMEMBER", "rt": "SINGLE_SENTENCE"},
        {"q": "Quelle tranche d'âge pour le tournoi Seniors Plus FFE ?", "s": "A01_2025_26_Championnat_de_France.pdf", "p": [3], "k": ["Seniors", "Plus", "50", "64"], "cat": "tournoi", "at": "FACTUAL", "cl": "REMEMBER", "rt": "SINGLE_SENTENCE"},
        # Coupe de France
        {"q": "Comment est structurée la Coupe de France échecs ?", "s": "C01_2025_26_Coupe_de_France.pdf", "p": [1], "k": ["Coupe", "France", "clubs", "équipe"], "cat": "tournoi", "at": "PROCEDURAL", "cl": "UNDERSTAND", "rt": "MULTI_SENTENCE"},
        {"q": "Combien d'équipes par club en Coupe de France ?", "s": "C01_2025_26_Coupe_de_France.pdf", "p": [1], "k": ["club", "engager", "équipe", "une"], "cat": "tournoi", "at": "FACTUAL", "cl": "REMEMBER", "rt": "LEXICAL_MATCH"},
        {"q": "Quelles règles aux parties de Coupe de France ?", "s": "C01_2025_26_Coupe_de_France.pdf", "p": [3], "k": ["règles", "FIDE", "FFE", "parties"], "cat": "tournoi", "at": "FACTUAL", "cl": "REMEMBER", "rt": "SINGLE_SENTENCE"},
        # Classement
        {"q": "Quelles parties comptent pour le classement rapide FFE ?", "s": "E02-Le_classement_rapide.pdf", "p": [1], "k": ["Rapide", "parties", "60", "min"], "cat": "classement", "at": "FACTUAL", "cl": "REMEMBER", "rt": "SINGLE_SENTENCE"},
        # Plus de questions pour atteindre 81...
        {"q": "Comment manipuler la pendule d'échecs ?", "s": "LA-octobre2025.pdf", "p": [44], "k": ["pendule", "manipulation", "joueur", "appuyer"], "cat": "temps", "at": "PROCEDURAL", "cl": "APPLY", "rt": "MULTI_SENTENCE"},
        {"q": "Que faire si la pendule est défectueuse ?", "s": "LA-octobre2025.pdf", "p": [44], "k": ["pendule", "défectueuse", "arbitre", "temps"], "cat": "temps", "at": "PROCEDURAL", "cl": "APPLY", "rt": "MULTI_SENTENCE"},
        {"q": "Qu'est-ce qu'une colonne aux échecs ?", "s": "LA-octobre2025.pdf", "p": [37], "k": ["colonne", "vertical", "cases", "échiquier"], "cat": "regles_jeu", "at": "DEFINITIONAL", "cl": "REMEMBER", "rt": "SINGLE_SENTENCE"},
        {"q": "Qu'est-ce qu'une rangée aux échecs ?", "s": "LA-octobre2025.pdf", "p": [37], "k": ["rangée", "horizontal", "cases", "échiquier"], "cat": "regles_jeu", "at": "DEFINITIONAL", "cl": "REMEMBER", "rt": "SINGLE_SENTENCE"},
        {"q": "Qu'est-ce qu'une diagonale aux échecs ?", "s": "LA-octobre2025.pdf", "p": [37], "k": ["diagonale", "cases", "couleur", "bord"], "cat": "regles_jeu", "at": "DEFINITIONAL", "cl": "REMEMBER", "rt": "SINGLE_SENTENCE"},
        {"q": "Comment une pièce prend-elle une autre pièce ?", "s": "LA-octobre2025.pdf", "p": [38], "k": ["prise", "pièce", "adverse", "retir"], "cat": "regles_jeu", "at": "PROCEDURAL", "cl": "UNDERSTAND", "rt": "SINGLE_SENTENCE"},
        {"q": "Quand une pièce attaque-t-elle une case ?", "s": "LA-octobre2025.pdf", "p": [38], "k": ["attaque", "pièce", "case", "déplacer"], "cat": "regles_jeu", "at": "PROCEDURAL", "cl": "UNDERSTAND", "rt": "SINGLE_SENTENCE"},
        {"q": "Comment se déplace le fou aux échecs ?", "s": "LA-octobre2025.pdf", "p": [38], "k": ["fou", "diagonale", "déplace", "case"], "cat": "regles_jeu", "at": "PROCEDURAL", "cl": "REMEMBER", "rt": "SINGLE_SENTENCE"},
        {"q": "Peut-on déplacer une pièce qui expose le roi à l'échec ?", "s": "LA-octobre2025.pdf", "p": [41], "k": ["pièce", "expose", "roi", "échec"], "cat": "regles_jeu", "at": "CONDITIONAL", "cl": "UNDERSTAND", "rt": "SINGLE_SENTENCE"},
        {"q": "Quand la partie est-elle nulle par pat ?", "s": "LA-octobre2025.pdf", "p": [43], "k": ["nulle", "pat", "joueur", "coup"], "cat": "regles_jeu", "at": "PROCEDURAL", "cl": "UNDERSTAND", "rt": "MULTI_SENTENCE"},
        {"q": "Quel matériel minimum pour faire mat ?", "s": "LA-octobre2025.pdf", "p": [37], "k": ["matériel", "suffisant", "mat", "position"], "cat": "regles_jeu", "at": "FACTUAL", "cl": "ANALYZE", "rt": "DOMAIN_KNOWLEDGE"},
        {"q": "Qu'est-ce qu'une position morte aux échecs ?", "s": "LA-octobre2025.pdf", "p": [37], "k": ["position", "morte", "mat", "aucun"], "cat": "regles_jeu", "at": "DEFINITIONAL", "cl": "REMEMBER", "rt": "SINGLE_SENTENCE"},
        {"q": "Quand peut-on adouber les pièces ?", "s": "LA-octobre2025.pdf", "p": [44], "k": ["adouber", "pendule", "joueur", "pièces"], "cat": "regles_jeu", "at": "CONDITIONAL", "cl": "UNDERSTAND", "rt": "SINGLE_SENTENCE"},
        {"q": "Que faire si un joueur ne peut pas utiliser la pendule ?", "s": "LA-octobre2025.pdf", "p": [44], "k": ["pendule", "assistant", "arbitre", "handicap"], "cat": "handicap", "at": "PROCEDURAL", "cl": "APPLY", "rt": "MULTI_SENTENCE"},
        {"q": "Comment noter les coups aux échecs ?", "s": "LA-octobre2025.pdf", "p": [59], "k": ["notation", "algébrique", "coups", "feuille"], "cat": "notation", "at": "PROCEDURAL", "cl": "UNDERSTAND", "rt": "MULTI_SENTENCE"},
        {"q": "Comment annoncer les coups pour joueurs aveugles ?", "s": "LA-octobre2025.pdf", "p": [61], "k": ["annoncer", "coups", "aveugle", "répét"], "cat": "handicap", "at": "PROCEDURAL", "cl": "UNDERSTAND", "rt": "MULTI_SENTENCE"},
        {"q": "Quels critères pour l'échiquier des joueurs aveugles ?", "s": "LA-octobre2025.pdf", "p": [61], "k": ["échiquier", "critères", "cases", "surélevé"], "cat": "handicap", "at": "LIST", "cl": "REMEMBER", "rt": "MULTI_SENTENCE"},
        {"q": "Comment noter pour les personnes handicapées visuelles ?", "s": "LA-octobre2025.pdf", "p": [61], "k": ["noter", "Braille", "handicap", "visuel"], "cat": "handicap", "at": "PROCEDURAL", "cl": "UNDERSTAND", "rt": "SINGLE_SENTENCE"},
        {"q": "Qu'est-ce que le tournoi Accession Roger Ferry ?", "s": "A01_2025_26_Championnat_de_France.pdf", "p": [1], "k": ["Accession", "Roger", "Ferry", "tournoi"], "cat": "tournoi", "at": "DEFINITIONAL", "cl": "REMEMBER", "rt": "SINGLE_SENTENCE"},
        {"q": "Quels clubs peuvent participer aux compétitions fédérales ?", "s": "R01_2025_26_Regles_generales.pdf", "p": [1], "k": ["clubs", "affiliés", "engagements", "FFE"], "cat": "tournoi", "at": "CONDITIONAL", "cl": "UNDERSTAND", "rt": "MULTI_SENTENCE"},
        {"q": "Quelles catégories au Championnat de France Jeunes ?", "s": "J01_2025_26_Championnat_de_France_Jeunes.pdf", "p": [3], "k": ["catégories", "U16", "U14", "U12"], "cat": "jeunes", "at": "LIST", "cl": "REMEMBER", "rt": "MULTI_SENTENCE"},
        {"q": "Quel Elo minimum pour qualification d'office jeunes ?", "s": "J01_2025_26_Championnat_de_France_Jeunes.pdf", "p": [3], "k": ["Elo", "minimum", "juillet", "août"], "cat": "jeunes", "at": "FACTUAL", "cl": "REMEMBER", "rt": "SINGLE_SENTENCE"},
        {"q": "Comment est composée la Commission Régionale de Discipline ?", "s": "2018_Reglement_Disciplinaire20180422.pdf", "p": [2], "k": ["Commission", "Régionale", "Discipline", "Comité"], "cat": "discipline", "at": "PROCEDURAL", "cl": "UNDERSTAND", "rt": "MULTI_SENTENCE"},
        {"q": "Comment peut-on mettre fin au mandat d'un membre disciplinaire ?", "s": "2018_Reglement_Disciplinaire20180422.pdf", "p": [2], "k": ["mandat", "membre", "démission", "empêchement"], "cat": "discipline", "at": "LIST", "cl": "REMEMBER", "rt": "MULTI_SENTENCE"},
        {"q": "La personne poursuivie peut-elle être assistée d'un avocat ?", "s": "2018_Reglement_Disciplinaire20180422.pdf", "p": [5], "k": ["avocat", "conseil", "assist", "poursuivi"], "cat": "discipline", "at": "FACTUAL", "cl": "REMEMBER", "rt": "SINGLE_SENTENCE"},
        {"q": "Peut-on demander un interprète en audience disciplinaire ?", "s": "2018_Reglement_Disciplinaire20180422.pdf", "p": [5], "k": ["interprète", "langue", "française", "frais"], "cat": "discipline", "at": "CONDITIONAL", "cl": "UNDERSTAND", "rt": "MULTI_SENTENCE"},
        {"q": "Quel est le format des Opens au Championnat de France ?", "s": "A01_2025_26_Championnat_de_France.pdf", "p": [1], "k": ["Open", "A", "B", "C"], "cat": "tournoi", "at": "LIST", "cl": "REMEMBER", "rt": "MULTI_SENTENCE"},
        {"q": "Comment qualifier les premières du National Féminin précédent ?", "s": "A01_2025_26_Championnat_de_France.pdf", "p": [2], "k": ["premières", "National", "Féminin", "précédent"], "cat": "feminin", "at": "PROCEDURAL", "cl": "UNDERSTAND", "rt": "MULTI_SENTENCE"},
        {"q": "Quelles joueuses sont qualifiées par Elo au National Féminin ?", "s": "A01_2025_26_Championnat_de_France.pdf", "p": [2], "k": ["Elo", "moyen", "FIDE", "joueuses"], "cat": "feminin", "at": "LIST", "cl": "ANALYZE", "rt": "MULTI_SENTENCE"},
        {"q": "Qui informe les clubs du lieu des matchs de Coupe de France ?", "s": "C01_2025_26_Coupe_de_France.pdf", "p": [2], "k": ["clubs", "informés", "lieu", "Direction"], "cat": "tournoi", "at": "FACTUAL", "cl": "REMEMBER", "rt": "SINGLE_SENTENCE"},
        {"q": "Quelle durée minimum pour partie en cadence rapide ?", "s": "E02-Le_classement_rapide.pdf", "p": [1], "k": ["Rapide", "10", "min", "joueur"], "cat": "classement", "at": "FACTUAL", "cl": "REMEMBER", "rt": "SINGLE_SENTENCE"},
        {"q": "Quelle durée maximum pour partie en cadence rapide ?", "s": "E02-Le_classement_rapide.pdf", "p": [1], "k": ["Rapide", "60", "min", "parties"], "cat": "classement", "at": "FACTUAL", "cl": "REMEMBER", "rt": "SINGLE_SENTENCE"},
        # 5 dernières questions
        {"q": "Comment vérifier une réclamation de nulle ?", "s": "LA-octobre2025.pdf", "p": [52], "k": ["nulle", "réclamation", "vérification", "adversaires"], "cat": "regles_jeu", "at": "PROCEDURAL", "cl": "APPLY", "rt": "MULTI_SENTENCE"},
        {"q": "Quand la partie est-elle nulle sur demande ?", "s": "LA-octobre2025.pdf", "p": [52], "k": ["nulle", "demande", "joueur", "coup"], "cat": "regles_jeu", "at": "CONDITIONAL", "cl": "UNDERSTAND", "rt": "MULTI_SENTENCE"},
        {"q": "Quel est le rôle principal de l'arbitre aux échecs ?", "s": "LA-octobre2025.pdf", "p": [55], "k": ["arbitre", "règles", "observation", "surveiller"], "cat": "arbitrage", "at": "FACTUAL", "cl": "REMEMBER", "rt": "SINGLE_SENTENCE"},
        {"q": "Quand l'arbitre doit-il intervenir pendant une partie ?", "s": "LA-octobre2025.pdf", "p": [55], "k": ["arbitre", "intervenir", "infraction", "adversaire"], "cat": "arbitrage", "at": "CONDITIONAL", "cl": "UNDERSTAND", "rt": "MULTI_SENTENCE"},
        {"q": "Comment fonctionne le Chess960 ?", "s": "LA-octobre2025.pdf", "p": [65], "k": ["Chess960", "position", "initiale", "hasard"], "cat": "regles_jeu", "at": "PROCEDURAL", "cl": "UNDERSTAND", "rt": "MULTI_SENTENCE"},
    ]

def main():
    gs, chunks_data = load_data()
    index = build_index(chunks_data)
    questions = get_validated_questions()

    existing = {q.get("question", "").lower().strip() for q in gs["questions"]}

    valid = []
    invalid = []
    duplicates = []

    for q in questions:
        if q["q"].lower().strip() in existing:
            duplicates.append(q["q"])
            continue
        ok, found = validate_question(index, q["s"], q["p"], q["k"])
        if ok:
            valid.append(q)
        else:
            invalid.append({"q": q["q"], "k": q["k"], "p": q["p"]})

    print(f"=== RÉSULTAT VALIDATION ===")
    print(f"Total: {len(questions)}")
    print(f"Validées: {len(valid)}")
    print(f"Duplicats: {len(duplicates)}")
    print(f"Invalides: {len(invalid)}")

    if invalid:
        print(f"\nQuestions invalides:")
        for i in invalid[:10]:
            print(f"  - {i['q'][:50]}... pages={i['p']}")

    print(f"\nQuestions encore à créer: {81 - len(valid)}")

    return valid, invalid, duplicates

if __name__ == "__main__":
    main()
