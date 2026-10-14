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
