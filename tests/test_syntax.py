"""Синтаксический слой на вручную размеченных деревьях (эталонных разборах)."""

from app.nlp.parsers import Parsed, fix_lemma, sanitize_tree
from app.nlp.syntax import SentenceAnalysis
from app.nlp.tree import sentence_from_rows

M_SG_NOM = "Case=Nom|Gender=Masc|Number=Sing"
PAST_M = "Gender=Masc|Number=Sing|Tense=Past|VerbForm=Fin"


def rels(an, i, rel_type):
    return {(r.related_lemma, r.scope) for r in an.relations(i) if r.rel_type == rel_type}


def lemmas(an, i, rel_type):
    return {r.related_lemma for r in an.relations(i) if r.rel_type == rel_type}


LILAC = sentence_from_rows([
    ("Сладкий", "сладкий", "ADJ", M_SG_NOM, 4, "amod"),
    (",", ",", "PUNCT", "", 3, "punct"),
    ("тяжёлый", "тяжёлый", "ADJ", M_SG_NOM, 4, "amod"),
    ("запах", "запах", "NOUN", M_SG_NOM, 12, "nsubj"),
    ("сирени", "сирень", "NOUN", "Case=Gen|Gender=Fem|Number=Sing", 4, "nmod"),
    (",", ",", "PUNCT", "", 8, "punct"),
    ("который", "который", "PRON", M_SG_NOM, 8, "nsubj"),
    ("шёл", "идти", "VERB", PAST_M, 4, "acl:relcl"),
    ("из", "из", "ADP", "", 10, "case"),
    ("сада", "сад", "NOUN", "Case=Gen|Gender=Masc|Number=Sing", 8, "obl"),
    (",", ",", "PUNCT", "", 8, "punct"),
    ("наполнял", "наполнять", "VERB", PAST_M, 0, "root"),
    ("комнату", "комната", "NOUN", "Case=Acc|Gender=Fem|Number=Sing", 12, "obj"),
    ("и", "и", "CCONJ", "", 15, "cc"),
    ("кружил", "кружить", "VERB", PAST_M, 12, "conj"),
    ("голову", "голова", "NOUN", "Case=Acc|Gender=Fem|Number=Sing", 15, "obj"),
    (".", ".", "PUNCT", "", 12, "punct"),
])


def test_lilac_full_profile():
    an = SentenceAnalysis(LILAC)
    assert an.role(4).role == "subject"
    assert lemmas(an, 4, "attribute") == {"сладкий", "тяжёлый", "сирень", "идти"}
    preds = rels(an, 4, "predicate")
    # однородное сказуемое «кружил» без правила распространения терялось бы
    assert ("наполнять", "direct") in preds
    assert ("кружить", "shared") in preds
    assert ("идти", "direct") in preds  # через «который»
    m = an.metrics()
    assert m["sent_type"] == "complex" and m["has_homogeneous"]


def test_copula_predicate():
    s = sentence_from_rows([
        ("Запах", "запах", "NOUN", M_SG_NOM, 3, "nsubj"),
        ("был", "быть", "AUX", PAST_M, 3, "cop"),
        ("сладким", "сладкий", "ADJ", "Case=Ins|Gender=Masc|Number=Sing", 0, "root"),
        (".", ".", "PUNCT", "", 3, "punct"),
    ])
    an = SentenceAnalysis(s)
    preds = [r for r in an.relations(1) if r.rel_type == "predicate"]
    assert preds[0].related_lemma == "сладкий"
    assert "быть" in preds[0].detail
    assert an.role(3).role == "predicate"


SHARED_ADJ = sentence_from_rows([
    ("Сладкий", "сладкий", "ADJ", M_SG_NOM, 2, "amod"),
    ("запах", "запах", "NOUN", M_SG_NOM, 5, "nsubj"),
    ("и", "и", "CCONJ", "", 4, "cc"),
    ("аромат", "аромат", "NOUN", M_SG_NOM, 2, "conj"),
    ("стояли", "стоять", "VERB", "Number=Plur|Tense=Past|VerbForm=Fin", 0, "root"),
    ("в", "в", "ADP", "", 7, "case"),
    ("комнате", "комната", "NOUN", "Case=Loc|Gender=Fem|Number=Sing", 5, "obl"),
    (".", ".", "PUNCT", "", 5, "punct"),
])


def test_shared_left_attribute_and_inherited_role():
    an = SentenceAnalysis(SHARED_ADJ)
    assert an.role(4).role == "subject"
    assert ("сладкий", "shared") in rels(an, 4, "attribute")
    assert ("стоять", "shared") in rels(an, 4, "predicate")
    assert lemmas(an, 4, "adverbial") == {"комната"}
    assert lemmas(an, 4, "conj") == {"запах"}
    assert an.groups[0].conj_type == "copulative"


def test_parallelism_makes_private():
    s = sentence_from_rows([
        ("Сладкий", "сладкий", "ADJ", M_SG_NOM, 2, "amod"),
        ("запах", "запах", "NOUN", M_SG_NOM, 6, "nsubj"),
        ("и", "и", "CCONJ", "", 5, "cc"),
        ("горький", "горький", "ADJ", M_SG_NOM, 5, "amod"),
        ("аромат", "аромат", "NOUN", M_SG_NOM, 2, "conj"),
        ("стояли", "стоять", "VERB", "Number=Plur|Tense=Past|VerbForm=Fin", 0, "root"),
    ])
    an = SentenceAnalysis(s)
    assert lemmas(an, 5, "attribute") == {"горький"}
    assert lemmas(an, 2, "attribute") == {"сладкий"}


def test_plural_adjective_is_shared_by_agreement():
    s = sentence_from_rows([
        ("Сладкие", "сладкий", "ADJ", "Case=Nom|Number=Plur", 2, "amod"),
        ("запах", "запах", "NOUN", M_SG_NOM, 0, "root"),
        ("и", "и", "CCONJ", "", 4, "cc"),
        ("аромат", "аромат", "NOUN", M_SG_NOM, 2, "conj"),
    ])
    an = SentenceAnalysis(s)
    r = [x for x in an.relations(4) if x.rel_type == "attribute"][0]
    assert (r.scope, r.rule) == ("shared", "agreement")


def test_shared_object_across_verbs_and_subject_inheritance():
    s = sentence_from_rows([
        ("Он", "он", "PRON", "Case=Nom|Gender=Masc|Number=Sing|Person=3", 2, "nsubj"),
        ("почувствовал", "почувствовать", "VERB", PAST_M, 0, "root"),
        ("и", "и", "CCONJ", "", 4, "cc"),
        ("вдохнул", "вдохнуть", "VERB", PAST_M, 2, "conj"),
        ("запах", "запах", "NOUN", "Case=Acc|Gender=Masc|Number=Sing", 4, "obj"),
        (".", ".", "PUNCT", "", 2, "punct"),
    ])
    an = SentenceAnalysis(s)
    gov = rels(an, 5, "governor")
    assert ("вдохнуть", "direct") in gov
    assert ("почувствовать", "shared") in gov
    assert lemmas(an, 5, "subject_of_pred") == {"он"}


def test_intransitive_verb_does_not_share_object():
    s = sentence_from_rows([
        ("Он", "он", "PRON", "Case=Nom|Number=Sing", 2, "nsubj"),
        ("подошёл", "подойти", "VERB", PAST_M, 0, "root"),
        ("и", "и", "CCONJ", "", 4, "cc"),
        ("понюхал", "понюхать", "VERB", PAST_M, 2, "conj"),
        ("цветок", "цветок", "NOUN", "Case=Acc|Number=Sing", 4, "obj"),
    ])
    an = SentenceAnalysis(s)
    assert lemmas(an, 5, "governor") == {"понюхать"}


def test_false_homogeneity_pair():
    s = sentence_from_rows([
        ("Он", "он", "PRON", "Case=Nom|Number=Sing", 2, "nsubj"),
        ("пошёл", "пойти", "VERB", PAST_M, 0, "root"),
        ("и", "и", "CCONJ", "", 4, "cc"),
        ("купил", "купить", "VERB", PAST_M, 2, "conj"),
        ("духи", "духи", "NOUN", "Case=Acc|Number=Plur", 4, "obj"),
    ])
    an = SentenceAnalysis(s)
    assert an.groups == []
    assert lemmas(an, 5, "governor") == {"купить"}


def test_adversative_negation():
    s = sentence_from_rows([
        ("Не", "не", "PART", "", 2, "advmod"),
        ("запах", "запах", "NOUN", M_SG_NOM, 6, "nsubj"),
        (",", ",", "PUNCT", "", 5, "punct"),
        ("а", "а", "CCONJ", "", 5, "cc"),
        ("вонь", "вонь", "NOUN", "Case=Nom|Gender=Fem|Number=Sing", 2, "conj"),
        ("стояла", "стоять", "VERB", "Gender=Fem|Number=Sing|Tense=Past|VerbForm=Fin", 0, "root"),
    ])
    an = SentenceAnalysis(s)
    assert an.groups[0].conj_type == "adversative"
    p_neg = [r for r in an.relations(2) if r.rel_type == "predicate"][0]
    p_pos = [r for r in an.relations(5) if r.rel_type == "predicate"][0]
    assert p_neg.negated and not p_pos.negated


def test_control_and_gerund_subjects():
    s = sentence_from_rows([
        ("Он", "он", "PRON", "Case=Nom|Number=Sing", 2, "nsubj"),
        ("хотел", "хотеть", "VERB", PAST_M, 0, "root"),
        ("почувствовать", "почувствовать", "VERB", "VerbForm=Inf", 2, "xcomp"),
        ("запах", "запах", "NOUN", "Case=Acc|Number=Sing", 3, "obj"),
    ])
    an = SentenceAnalysis(s)
    subj = [r for r in an.relations(4) if r.rel_type == "subject_of_pred"][0]
    assert (subj.related_lemma, subj.rule) == ("он", "control")

    s2 = sentence_from_rows([
        ("Почувствовав", "почувствовать", "VERB", "VerbForm=Conv", 5, "advcl"),
        ("запах", "запах", "NOUN", "Case=Acc|Number=Sing", 1, "obj"),
        (",", ",", "PUNCT", "", 1, "punct"),
        ("он", "он", "PRON", "Case=Nom|Number=Sing", 5, "nsubj"),
        ("остановился", "остановиться", "VERB", PAST_M, 0, "root"),
    ])
    an2 = SentenceAnalysis(s2)
    subj = [r for r in an2.relations(2) if r.rel_type == "subject_of_pred"][0]
    assert subj.rule == "gerund"
    assert an2.metrics()["sent_type"] == "simple"


def test_obl_classification_and_case():
    s = sentence_from_rows([
        ("Пахло", "пахнуть", "VERB", "Gender=Neut|Number=Sing|Tense=Past|VerbForm=Fin", 0, "root"),
        ("дымом", "дым", "NOUN", "Case=Ins|Number=Sing", 1, "obl"),
    ])
    an = SentenceAnalysis(s)
    assert an.role(2).role == "object"
    assert lemmas(an, 2, "governor") == {"пахнуть"}

    s2 = sentence_from_rows([
        ("Он", "он", "PRON", "Case=Nom|Number=Sing", 2, "nsubj"),
        ("проснулся", "проснуться", "VERB", PAST_M, 0, "root"),
        ("от", "от", "ADP", "", 4, "case"),
        ("запаха", "запах", "NOUN", "Case=Gen|Number=Sing", 2, "obl"),
    ])
    an2 = SentenceAnalysis(s2)
    role = an2.role(4)
    assert role.role == "adverbial" and "причины" in role.subtype
    assert lemmas(an2, 4, "case") == {"от+Gen"}


def test_phase_verb_compound_predicate():
    s = sentence_from_rows([
        ("Запах", "запах", "NOUN", M_SG_NOM, 2, "nsubj"),
        ("начал", "начать", "VERB", PAST_M, 0, "root"),
        ("усиливаться", "усиливаться", "VERB", "VerbForm=Inf", 2, "xcomp"),
    ])
    an = SentenceAnalysis(s)
    assert lemmas(an, 1, "predicate") == {"начать", "усиливаться"}


INS_SG_F = "Case=Ins|Gender=Fem|Number=Sing"
INS_PL = "Case=Ins|Number=Plur"
DEMIAN = sentence_from_rows([  # Гессе, «Демиан»: разбор Stanza
    ("Пахнет", "пахнуть", "VERB", "Number=Sing|Person=3|Tense=Pres|VerbForm=Fin", 0, "root"),
    ("теплой", "тёплый", "ADJ", INS_SG_F, 3, "amod"),
    ("теснотой", "теснота", "NOUN", INS_SG_F, 1, "iobj"),
    (",", ",", "PUNCT", "", 5, "punct"),
    ("кроликами", "кролик", "NOUN", INS_PL, 3, "conj"),
    ("и", "и", "CCONJ", "", 7, "cc"),
    ("служанками", "служанка", "NOUN", INS_PL, 3, "conj"),
    (",", ",", "PUNCT", "", 10, "punct"),
    ("домашними", "домашний", "ADJ", INS_PL, 10, "amod"),
    ("снадобьями", "снадобье", "NOUN", INS_PL, 3, "conj"),
    ("и", "и", "CCONJ", "", 13, "cc"),
    ("сушеными", "сушёный", "ADJ", INS_PL, 13, "amod"),
    ("фруктами", "фрукт", "NOUN", INS_PL, 3, "conj"),
    (".", ".", "PUNCT", "", 1, "punct"),
])


def test_verb_profile_has_all_dependents():
    """У глагола в профиле — все зависимые, а не только однородные сказуемые."""
    an = SentenceAnalysis(DEMIAN)
    assert rels(an, 1, "object") == {("теснота", "direct"), ("кролик", "shared"), ("служанка", "shared"),
                                     ("снадобье", "shared"), ("фрукт", "shared")}
    assert lemmas(an, 2, "head") == {"теснота"}
    assert lemmas(an, 9, "head") == {"снадобье"}  # внутри ряда — только своё слово
    assert lemmas(an, 13, "governor") == {"пахнуть"}
    assert lemmas(an, 10, "attribute") == {"домашний"}
    # «теплой» (ед. ч.) не согласуется с «кроликами» (мн. ч.) — не общее определение ряда
    assert lemmas(an, 3, "attribute") == {"тёплый"}
    assert not lemmas(an, 5, "attribute")


def test_verb_subject_adverbials_and_shared_head():
    s = sentence_from_rows([
        ("Ночью", "ночь", "NOUN", "Case=Ins|Gender=Fem|Number=Sing", 4, "obl"),
        ("сильно", "сильно", "ADV", "", 4, "advmod"),
        ("не", "не", "PART", "", 4, "advmod"),
        ("пахли", "пахнуть", "VERB", "Number=Plur|Tense=Past|VerbForm=Fin", 0, "root"),
        ("белые", "белый", "ADJ", "Case=Nom|Number=Plur", 6, "amod"),
        ("сирень", "сирень", "NOUN", "Case=Nom|Gender=Fem|Number=Sing", 4, "nsubj"),
        ("и", "и", "CCONJ", "", 8, "cc"),
        ("жасмин", "жасмин", "NOUN", M_SG_NOM, 6, "conj"),
        (".", ".", "PUNCT", "", 4, "punct"),
    ])
    an = SentenceAnalysis(s)
    assert rels(an, 4, "subject") == {("сирень", "direct"), ("жасмин", "shared")}
    assert lemmas(an, 4, "adverbial") == {"ночь", "сильно"}  # частица «не» — не обстоятельство
    assert lemmas(an, 5, "head") == {"сирень", "жасмин"}  # «белые» мн. ч. — общее определение ряда


def test_sanitize_tree_fixes_cycles_and_roots():
    words = [
        Parsed("запах", "запах", "NOUN", head=1, deprel="conj"),  # петля
        Parsed("шёл", "идти", "VERB", head=3, deprel="acl"),
        Parsed("из", "из", "ADP", head=2, deprel="case"),  # цикл 2↔3
        Parsed(".", ".", "PUNCT", head=1, deprel="punct"),
    ]
    fixed = sanitize_tree(words)
    roots = [i for i, w in enumerate(fixed, 1) if w.head == 0]
    assert len(roots) == 1
    for i in range(1, len(fixed) + 1):
        seen, cur = set(), i
        while cur:
            assert cur not in seen
            seen.add(cur)
            cur = fixed[cur - 1].head


def test_relative_pronoun_subject_resolves_to_antecedent():
    an = SentenceAnalysis(LILAC)
    # «из сада»: субъект действия «шёл» — не «который», а «запах»
    subj = [r for r in an.relations(10) if r.rel_type == "subject_of_pred"][0]
    assert (subj.related_lemma, subj.rule) == ("запах", "relcl")


def test_fix_lemma_replaces_nonexistent_lemma_only():
    past = {"Gender": "Masc", "Number": "Sing", "Tense": "Past", "VerbForm": "Fin"}
    assert fix_lemma("шёл", "ийти", "VERB", past) == "идти"  # ошибка лемматизатора Stanza
    assert fix_lemma("наполнял", "наполнять", "VERB", past) == "наполнять"  # верная лемма не меняется
    assert fix_lemma("Rome", "Rome", "PROPN", {}) == "Rome"  # латиница и имена собственные не трогаем


def test_association_measures():
    from app.services.stats import association, juilland_d, tf_idf

    # сочетание вдвое чаще ожидаемого: PMI = 1
    m = association(c=20, fx=100, fy=1000, n=10_000, span=1)
    assert m["expected"] == 10.0 and m["pmi"] == 1.0 and m["g2"] > 0
    assert 0 < m["npmi"] < 1 and m["mi3"] > m["pmi"] and m["tscore"] > 0
    assert m["dp_fwd"] > 0 and m["dp_bwd"] > 0
    # реже ожидаемого: отрицательные PMI и G²
    m = association(c=2, fx=100, fy=1000, n=10_000)
    assert m["pmi"] < 0 and m["g2"] < 0
    # направленность: редкий коллокат почти всегда рядом с частым узлом → ΔP(узел|коллокат) велика
    m = association(c=9, fx=1000, fy=10, n=100_000)
    assert m["dp_bwd"] > m["dp_fwd"]
    assert association(0, 10, 10, 100)["pmi"] is None
    assert juilland_d([5, 5, 5, 5]) == 1.0 and juilland_d([20, 0, 0, 0]) == 0.0
    assert tf_idf(5, 100, 1, 10) > tf_idf(5, 100, 10, 10) > 0


def test_entity_normalization():
    from app.services.entities import normalize

    assert normalize([("Андрею", "PROPN", {"Case": "Dat", "Gender": "Masc"}), ("Болконскому", "PROPN", {"Case": "Dat", "Gender": "Masc"})], "PER") == "Андрей Болконский"
    assert normalize([("Наташи", "PROPN", {"Case": "Gen", "Gender": "Fem"}), ("Ростовой", "PROPN", {"Case": "Gen", "Gender": "Fem"})], "PER") == "Наташа Ростова"
    assert normalize([("Лысых", "ADJ", {"Case": "Loc", "Number": "Plur"}), ("Горах", "PROPN", {"Case": "Loc", "Number": "Plur"})], "LOC") == "Лысые Горы"
    assert normalize([("Москву", "PROPN", {"Case": "Acc"})], "LOC") == "Москва"
    assert normalize([("газета", "NOUN", {"Case": "Nom"}), ("«", "PUNCT", {}), ("Правда", "NOUN", {}), ("»", "PUNCT", {})], "ORG") == "газета «Правда»"


def test_semantic_math():
    import numpy as np

    from app.services.entities import spring_layout
    from app.services.semantic import kmeans, pca2

    x = np.array([[1, 0, 0], [0.9, 0.1, 0], [0, 1, 0], [0, 0.9, 0.1]], dtype=np.float32)
    lab = kmeans(x, 2)
    assert lab[0] == lab[1] and lab[2] == lab[3] and lab[0] != lab[2]
    assert pca2(x).shape == (4, 2)
    pos = spring_layout(["a", "b", "c"], [("a", "b", 3)])
    assert set(pos) == {"a", "b", "c"} and all(abs(v) <= 1 for p in pos.values() for v in p)


def test_stylometry_math_and_ai_features():
    import numpy as np

    from app.services.aicheck import _logreg, _mattr, features
    from app.services.stylometry import dendrogram_coords, distance_matrix, mds2, upgma, zscores

    words = [["и", "в", "не", "что"] * 50, ["и", "в", "не", "что"] * 49 + ["он"] * 4, ["он", "она", "был"] * 60]
    z, feats = zscores(words, 5)
    d = distance_matrix(z, "cosine")
    assert d.shape == (3, 3) and d[0, 1] < d[0, 2] and abs(d[0, 0]) < 1e-9
    assert distance_matrix(z, "burrows")[0, 1] < distance_matrix(z, "burrows")[0, 2]
    m = upgma(d)
    assert m[0][:2] == (0, 1) and len(dendrogram_coords(m, 3)["segments"]) == 6 and mds2(d).shape == (3, 2)
    x = np.array([[0.0], [0.1], [0.9], [1.0]])
    w, b = _logreg((x - 0.5) / 0.4, np.array([0, 0, 1, 1]))
    assert w[0] > 0
    assert _mattr(["а", "б"] * 100, window=10) == 0.2
    s = [{"paragraph": 0, "text": "Важно отметить, что запах играет ключевую роль.",
          "tokens": [("Важно", "важно", "ADV", {}), ("отметить", "отметить", "VERB", {}), (",", ",", "PUNCT", {}), ("что", "что", "SCONJ", {}),
                     ("запах", "запах", "NOUN", {}), ("играет", "играть", "VERB", {}), ("ключевую", "ключевой", "ADJ", {}),
                     ("роль", "роль", "NOUN", {}), (".", ".", "PUNCT", {})]}]
    f = features(s)
    assert f["n_words"] == 7 and f["markers"] > 0 and f["commas"] > 0
