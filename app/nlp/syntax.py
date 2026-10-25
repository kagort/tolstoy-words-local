"""Синтаксический слой поверх UD (§5 ТЗ).

* сочинительные ряды и решение «общий / частный зависимый» (§5.6, правила 1–7);
* члены предложения в академической традиции (подлежащее, сказуемое, …);
* связи токена: определения, сказуемое, управляющий глагол, субъект, обстоятельства,
  однородные члены, предлог+падеж (§5.3);
* структурные метрики предложения.

Модуль чистый: работает только с объектом ``Sentence`` и не обращается к БД,
поэтому покрыт модульными тестами на вручную построенных деревьях.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.nlp.tree import Sentence, base_rel

SYNTAX_VERSION = "1.1"

# ------------------------------------------------------------------ словари правил

COPULATIVE = {"и", "да", "ни", "также", "тоже", "плюс"}
ADVERSATIVE = {"а", "но", "однако", "зато", "же", "хотя"}
DISJUNCTIVE = {"или", "либо", "то", "иль", "ли"}

# «пошёл и купил», «взял да и ушёл» — одно сказуемое (правило 6)
FALSE_PAIR_FIRST = {
    "пойти", "идти", "взять", "брать", "сесть", "садиться", "встать", "вставать",
    "прийти", "приходить", "подойти", "подходить", "уйти", "давай", "давать",
}

# глаголы, которые не берут прямого дополнения (правило 3, дополняется по корпусу)
INTRANSITIVE = {
    "идти", "пойти", "прийти", "подойти", "уйти", "выйти", "войти", "бежать", "стоять", "сидеть",
    "лежать", "спать", "жить", "быть", "стать", "становиться", "оказаться", "казаться", "пахнуть",
    "вонять", "благоухать", "смердеть", "веять", "тянуть", "нести", "расти", "умереть", "плакать",
    "смеяться", "молчать", "улыбнуться", "улыбаться", "остановиться", "подняться", "появиться",
}

# фазовые и модальные глаголы → составное глагольное сказуемое (xcomp)
PHASE_MODAL = {
    "начать", "начинать", "стать", "становиться", "продолжать", "продолжить", "перестать",
    "переставать", "кончить", "закончить", "прекратить", "мочь", "смочь", "хотеть", "захотеть",
    "уметь", "суметь", "пытаться", "попытаться", "стараться", "постараться", "собираться",
    "собраться", "любить", "желать", "решить", "решать", "думать", "надеяться", "успеть",
}

RELATIVE_PRON = {"который", "что", "чей", "кто"}

OBJECT_PREPS = {"о", "об", "обо", "про"}
PLACE_PREPS = {
    "в", "во", "на", "под", "подо", "над", "надо", "за", "перед", "передо", "около", "у", "возле",
    "вокруг", "среди", "между", "вдоль", "мимо", "напротив", "из", "изо", "по", "сквозь", "вне", "внутри",
}
TIME_PREPS = {"до", "после", "через", "накануне", "спустя", "во время", "в течение"}
CAUSE_PREPS = {"из-за", "благодаря", "вследствие", "ввиду", "от"}
PURPOSE_PREPS = {"для", "ради"}
# «ночью», «зимой»: творительный без предлога — обстоятельство времени, а не дополнение
TIME_NOUNS = {"утро", "день", "вечер", "ночь", "зима", "весна", "лето", "осень", "рассвет", "закат", "сумерки", "полдень", "полночь"}

ATTR_RELS = {"amod", "det", "nmod", "acl", "nummod", "appos"}
HEAD_RELS = {"amod", "det", "nmod", "acl", "nummod"}  # определение → определяемое слово
DEP_RELS = {"obj", "iobj", "obl", "advmod", "advcl"}  # зависимые глагола: дополнения и обстоятельства
SUBJ_RELS = {"nsubj", "csubj"}
OBJ_RELS = {"obj", "iobj"}
SKIP_DEP_RELS = {"conj", "cc", "punct", "orphan", "fixed", "flat", "goeswith", "reparandum", "dep"}

ROLE_LABELS = {
    "subject": "подлежащее",
    "predicate": "сказуемое",
    "main_member": "главный член (односоставное)",
    "attribute": "определение",
    "object": "дополнение",
    "adverbial": "обстоятельство",
    "apposition": "приложение",
    "vocative": "обращение",
    "other": "вне структуры",
}

# Метки UD, из которых выводится член предложения (показываются рядом с русским названием)
ROLE_UD = {
    "subject": "nsubj, csubj",
    "predicate": "root, cop, xcomp",
    "main_member": "root",
    "attribute": "amod, det, nmod, acl",
    "object": "obj, iobj, obl",
    "adverbial": "advmod, obl, advcl",
    "apposition": "appos, flat",
    "vocative": "vocative",
    "other": "прочие",
}

REL_UD = {
    "attribute": "amod, nmod, acl",
    "head": "голова amod / nmod / acl",
    "subject": "nsubj, csubj",
    "object": "obj, iobj, obl",
    "predicate": "голова nsubj; cop, xcomp",
    "governor": "голова obj / iobj / obl",
    "subject_of_pred": "nsubj",
    "adverbial": "advmod, obl, advcl",
    "conj": "conj",
    "case": "case",
}


def role_label_ud(role: str, deprel: str | None = None) -> str:
    """«подлежащее (nsubj)» — с конкретной меткой UD, если она известна, иначе с типичными."""
    label = ROLE_LABELS.get(role, role)
    if deprel == "conj":  # роль унаследована от вершины однородного ряда
        return f"{label} (однородный член, conj)"
    return f"{label} ({deprel or ROLE_UD.get(role, '')})"


def rel_label_ud(rel_type: str) -> str:
    return f"{REL_LABELS.get(rel_type, rel_type)} ({REL_UD.get(rel_type, '')})"


REL_LABELS = {
    "attribute": "определения",
    "head": "определяемое слово",
    "subject": "подлежащее при сказуемом",
    "object": "дополнения при глаголе",
    "predicate": "сказуемое при подлежащем",
    "governor": "управляющее слово",
    "subject_of_pred": "субъект действия",
    "adverbial": "обстоятельства при сказуемом",
    "conj": "однородные члены",
    "case": "предлог + падеж",
}


# ------------------------------------------------------------------ структуры


@dataclass
class CoordGroup:
    members: list[int]
    head: int
    conj_type: str
    conj_lemmas: list[str] = field(default_factory=list)
    negated: set[int] = field(default_factory=set)

    @property
    def first(self) -> int:
        return min(self.members)

    @property
    def last(self) -> int:
        return max(self.members)


@dataclass
class Relation:
    rel_type: str
    direction: str  # down | up | sibling
    related_idx: int
    related_lemma: str
    related_upos: str = ""
    detail: str = ""
    path: str = ""
    source: str = "basic"  # basic | enhanced
    scope: str = "direct"  # direct | shared | private | uncertain
    rule: str = ""
    confidence: float = 1.0
    negated: bool = False


@dataclass
class Role:
    role: str
    subtype: str = ""
    detail: str = ""

    @property
    def label(self) -> str:
        return ROLE_LABELS.get(self.role, self.role)


# ------------------------------------------------------------------ анализ предложения


class SentenceAnalysis:
    def __init__(self, sent: Sentence) -> None:
        self.s = sent
        self.groups = self._build_groups()
        self.group_by_member: dict[int, CoordGroup] = {}
        for g in self.groups:
            for m in g.members:
                self.group_by_member[m] = g

    # ---------------------------------------------- вспомогательное

    def t(self, i: int):
        return self.s.tok(i)

    def rel(self, i: int) -> str:
        return base_rel(self.t(i).deprel)

    def lemma(self, i: int) -> str:
        return self.t(i).lemma.lower()

    def _case_prep(self, i: int) -> str:
        preps = [self.lemma(c) for c in self.s.children(i, {"case"})]
        if not preps:
            return ""
        # многословные предлоги: «из-за», «во время»
        return " ".join(preps)

    def case_detail(self, i: int) -> str:
        prep = self._case_prep(i)
        case = self.t(i).feat("Case") or ""
        if prep and case:
            return f"{prep}+{case}"
        return prep or case

    # ---------------------------------------------- сочинительные ряды (§5.6)

    def _build_groups(self) -> list[CoordGroup]:
        s = self.s
        groups = []
        for t in s.tokens:
            if t.deprel == "conj":
                continue
            conj_kids = s.children(t.idx, {"conj"})
            if not conj_kids:
                continue
            members = [t.idx]
            queue = list(conj_kids)
            while queue:
                c = queue.pop(0)
                if c in members:
                    continue
                # гэппинг (orphan) — две клаузы, а не однородные члены (правило 6)
                if s.children(c, {"orphan"}):
                    continue
                members.append(c)
                queue.extend(s.children(c, {"conj"}))
            if len(members) < 2:
                continue
            members.sort()
            cc_lemmas = []
            for m in members:
                cc_lemmas += [self.lemma(c) for c in s.children(m, {"cc"})]
            members = self._drop_false_homogeneity(members)
            if len(members) < 2:
                continue
            g = CoordGroup(members=members, head=t.idx, conj_type=self._conj_type(members, cc_lemmas), conj_lemmas=cc_lemmas)
            if g.conj_type in ("adversative", "gradational"):
                for m in members:
                    if any(self.lemma(c) == "не" for c in s.children(m)):
                        g.negated.add(m)
            groups.append(g)
        return groups

    def _conj_type(self, members: list[int], cc: list[str]) -> str:
        cc_set = set(cc)
        if "но" in cc_set and any(self.lemma(c) == "только" for m in members for c in self.s.children(m)):
            return "gradational"
        if cc_set & ADVERSATIVE:
            return "adversative"
        if cc_set & DISJUNCTIVE:
            return "disjunctive"
        if cc_set & COPULATIVE:
            return "copulative"
        return "asyndetic" if not cc_set else "copulative"

    def _drop_false_homogeneity(self, members: list[int]) -> list[int]:
        s = self.s
        keep = list(members)
        for a, b in zip(members, members[1:]):
            ta, tb = s.tok(a), s.tok(b)
            if ta.upos not in ("VERB", "AUX") or tb.upos not in ("VERB", "AUX"):
                continue
            between = [s.tok(k) for k in range(a + 1, b)]
            # повтор «ходил-ходил»
            if ta.lemma == tb.lemma and all(x.form in "-–" for x in between):
                keep.remove(b)
                continue
            # «пошёл и купил», «взял да ушёл»: союз и/да вплотную
            if ta.lemma.lower() in FALSE_PAIR_FIRST and between and all(
                x.lemma.lower() in ("и", "да") for x in between
            ):
                keep.remove(b)
                continue
            # «сидит читает»: два глагола подряд без союза и запятой
            if b == a + 1 and ta.feat("Tense") == tb.feat("Tense"):
                keep.remove(b)
        return keep

    def group_of(self, i: int) -> CoordGroup | None:
        return self.group_by_member.get(i)

    def struct_head(self, i: int) -> int:
        """Вершина ряда, если токен — однородный член; иначе сам токен."""
        g = self.group_of(i)
        return g.head if g else i

    def decide_scope(self, d: int, m: int, x: int) -> tuple[str, str, float]:
        """Относится ли зависимый d члена ряда m также к члену x (правила 1–4)."""
        s = self.s
        g = self.group_of(m)
        drel = self.rel(d)
        td, tm, tx = s.tok(d), s.tok(m), s.tok(x)
        if g and g.first < d < g.last:
            return "private", "inside_span", 1.0
        # правило 1: параллелизм
        own = [c for c in s.children(x) if c != d and self.rel(c) == drel and c not in g.members]
        if own:
            return "private", "parallelism", 0.95
        # правило 2: согласование
        if drel in ("amod", "det"):
            if td.feat("Number") == "Plur" and tm.feat("Number") == "Sing" and tx.feat("Number") == "Sing":
                return "shared", "agreement", 0.95
            # не согласуется с членом x: «тёплой теснотой, кроликами» — «тёплой» не относится к «кроликами»
            dc, xc = td.feat("Case"), tx.feat("Case")
            if (dc and xc and dc != xc) or (td.feat("Number") == "Sing" and tx.feat("Number") == "Plur"):
                return "private", "agreement", 0.9
            dg, xg = td.feat("Gender"), tx.feat("Gender")
            if td.feat("Number") == "Sing" and tx.feat("Number") == "Sing" and dg and xg and dg != xg:
                return "private", "agreement", 0.9
        if drel == "nsubj" and tx.upos in ("VERB", "AUX", "ADJ"):
            ok = self._agrees(d, x)
            if ok is False:
                return "private", "agreement", 0.9
            if ok is True:
                return "shared", "agreement", 0.9
        # правило 3: одинаковое управление
        if drel in OBJ_RELS:
            if tx.upos != "VERB":
                return "private", "government", 0.8
            lx = tx.lemma.lower()
            if lx.endswith(("ся", "сь")) or lx in INTRANSITIVE:
                return "private", "government", 0.8
            return "shared", "government", 0.75
        if drel == "obl":
            if tx.upos != tm.upos:
                return "private", "government", 0.7
            return "uncertain", "government", 0.6
        # правило 4: линейная позиция
        if g and d < g.first:
            return "shared", "left_position", 0.85
        return "uncertain", "right_position", 0.6

    def _agrees(self, subj: int, verb: int) -> bool | None:
        ts, tv = self.t(subj), self.t(verb)
        sn, vn = ts.feat("Number"), tv.feat("Number")
        if sn and vn and sn != vn:
            return False
        if tv.feat("Tense") == "Past" and vn == "Sing":
            sg, vg = ts.feat("Gender"), tv.feat("Gender")
            if sg and vg and sg != vg:
                return False
        sp, vp = ts.feat("Person"), tv.feat("Person")
        if sp and vp and sp != vp:
            return False
        if vn or tv.feat("Gender") or vp:
            return True
        return None

    def deps_for(self, i: int, rels: set[str]) -> list[tuple[int, str, str, float, str]]:
        """Зависимые токена i заданных типов: собственные + общие от членов ряда.

        Возвращает (dep, scope, rule, confidence, source).
        """
        out = [(d, "direct", "", 1.0, "basic") for d in self.s.children(i) if self.rel(d) in rels]
        g = self.group_of(i)
        if g:
            for m in g.members:
                if m == i:
                    continue
                for d in self.s.children(m):
                    if self.rel(d) not in rels or d in g.members:
                        continue
                    scope, rule, conf = self.decide_scope(d, m, i)
                    if scope in ("shared", "uncertain"):
                        out.append((d, scope, rule, conf, "enhanced"))
        return out

    # ---------------------------------------------- члены предложения

    def role(self, i: int) -> Role:
        s = self.s
        h = self.struct_head(i)
        t = s.tok(i)
        th = s.tok(h)
        r = base_rel(th.deprel)
        full = th.deprel
        head_tok = s.tok(th.head) if th.head else None
        has_cop = bool(s.children(i, {"cop"}))
        if has_cop:
            return Role("predicate", "составное именное", f"связка: {self.lemma(s.children(i, {'cop'})[0])}")
        if r in SUBJ_RELS:
            sub = "пассивное" if full.endswith(":pass") else ""
            if t.feat("Case") == "Gen":
                sub = "генитивное"
            return Role("subject", sub)
        if r == "root":
            if t.upos in ("VERB", "AUX") or s.children(i, {"nsubj", "csubj"}):
                return Role("predicate", "глагольное" if t.upos in ("VERB", "AUX") else "именное")
            if t.upos in ("NOUN", "PROPN", "PRON", "NUM"):
                return Role("main_member", "номинативное")
            return Role("predicate", "")
        if r == "xcomp":
            return Role("predicate", "составное глагольное (часть)")
        if r in ("ccomp", "parataxis"):
            if t.upos in ("VERB", "AUX") or s.children(i, {"nsubj"}):
                return Role("predicate", "сказуемое другой клаузы")
            return Role("other", r)
        if r in ("amod", "det", "nummod"):
            return Role("attribute", "согласованное")
        if r == "acl":
            return Role("attribute", "придаточное определительное" if full == "acl:relcl" else "обособленное (оборот)")
        if r == "nmod":
            if head_tok is not None and head_tok.upos in ("NOUN", "PROPN", "PRON", "NUM", "ADJ"):
                return Role("attribute", "несогласованное", self.case_detail(i))
            return Role("object", "косвенное", self.case_detail(i))
        if r in OBJ_RELS:
            return Role("object", "прямое" if r == "obj" else "косвенное", self.case_detail(i))
        if r == "obl":
            return self._classify_obl(i)
        if r == "advmod":
            return Role("adverbial", "образа действия / меры")
        if r == "advcl":
            if t.feat("VerbForm") == "Conv":
                return Role("adverbial", "деепричастный оборот")
            return Role("adverbial", "придаточное обстоятельственное")
        if r in ("appos", "flat"):
            return Role("apposition")
        if r == "vocative":
            return Role("vocative")
        return Role("other", r)

    def _classify_obl(self, i: int) -> Role:
        prep = self._case_prep(i)
        case = self.t(i).feat("Case") or ""
        detail = self.case_detail(i)
        if not prep:
            if case == "Ins" and self.lemma(i) in TIME_NOUNS:
                return Role("adverbial", "времени", detail)
            if case in ("Ins", "Dat", "Gen", "Acc"):
                return Role("object", "косвенное", detail)
            return Role("adverbial", "", detail)
        if prep in OBJECT_PREPS:
            return Role("object", "косвенное", detail)
        if prep in ("с", "со") and case == "Ins":
            return Role("object", "косвенное", detail)
        if prep in TIME_PREPS:
            return Role("adverbial", "времени", detail)
        if prep in PURPOSE_PREPS:
            return Role("adverbial", "цели", detail)
        if prep in CAUSE_PREPS:
            return Role("adverbial", "причины / источника", detail)
        if prep in PLACE_PREPS:
            return Role("adverbial", "места", detail)
        return Role("adverbial", "", detail)

    # ---------------------------------------------- связи токена (§5.3)

    def relations(self, i: int) -> list[Relation]:
        s = self.s
        out: list[Relation] = []
        T = s.tok(i)
        g = self.group_of(i)
        h = self.struct_head(i)
        via = h != i
        role = self.role(i)

        def add(rel_type, direction, j, **kw):
            tj = s.tok(j)
            kw.setdefault("related_lemma", tj.lemma.lower())
            out.append(Relation(rel_type=rel_type, direction=direction, related_idx=j, related_upos=tj.upos, **kw))

        # 1. определения
        if T.upos in ("NOUN", "PROPN", "PRON", "NUM", "X") or role.role in ("subject", "object", "main_member"):
            for d, scope, rule, conf, src in self.deps_for(i, ATTR_RELS):
                td = s.tok(d)
                if td.lemma.lower() in RELATIVE_PRON:
                    continue
                full = td.deprel
                r = base_rel(full)
                if r in ("amod", "det", "nummod"):
                    detail = "согласованное"
                elif r == "nmod":
                    detail = "несогласованное " + self.case_detail(d)
                elif full == "acl:relcl":
                    detail = "придаточное"
                elif r == "acl":
                    detail = "оборот"
                else:
                    detail = r
                add("attribute", "down", d, detail=detail.strip(), path=full, source=src, scope=scope, rule=rule, confidence=conf)

        # 1б. определяемое слово (для определения): «тёплой теснотой» → «теснота»
        if role.role == "attribute" and base_rel(s.tok(h).deprel) in HEAD_RELS and s.tok(h).head:
            H = s.tok(h).head
            add("head", "up", H, detail=role.subtype, path=s.tok(h).deprel, source="enhanced" if via else "basic",
                scope="shared" if via else "direct", rule="conj_inherit" if via else "")
            gh = self.group_of(H)
            if gh:  # общее определение однородного ряда: «домашними снадобьями и фруктами»
                for Q in gh.members:
                    if Q == H:
                        continue
                    scope, rule, conf = self.decide_scope(h, H, Q)
                    if scope in ("shared", "uncertain"):
                        add("head", "up", Q, detail=role.subtype, path="conj", source="enhanced", scope=scope, rule=rule, confidence=conf)

        # 2. сказуемое при подлежащем
        if role.role == "subject":
            P = s.tok(h).head
            if P:
                self._add_predicate(out, P, i, via, add)
        # 2б. относительное придаточное: «запах, который шёл»
        for R in s.children(i):
            if s.tok(R).deprel != "acl:relcl":
                continue
            for pr in s.children(R):
                if s.tok(pr).lemma.lower() in RELATIVE_PRON:
                    prel = self.rel(pr)
                    if prel in SUBJ_RELS:
                        add("predicate", "down", R, detail="в придаточном (через «который»)", path="acl:relcl>nsubj", source="enhanced", rule="relcl")
                    elif prel in OBJ_RELS or prel == "obl":
                        add("governor", "down", R, detail=f"в придаточном ({prel})", path=f"acl:relcl>{prel}", source="enhanced", rule="relcl")

        # 3. управляющее слово (для дополнения / обстоятельства)
        if role.role in ("object", "adverbial") and s.tok(h).head:
            G = s.tok(h).head
            detail = self.case_detail(i)
            add("governor", "up", G, detail=detail, path=s.tok(h).deprel,
                source="enhanced" if via else "basic", scope="shared" if via else "direct",
                rule="conj_inherit" if via else "")
            gg = self.group_of(G)
            if gg:
                for Q in gg.members:
                    if Q == G:
                        continue
                    scope, rule, conf = self.decide_scope(h, G, Q)
                    if scope in ("shared", "uncertain"):
                        add("governor", "up", Q, detail=detail, path="conj", source="enhanced", scope=scope, rule=rule, confidence=conf)
            # 4. субъект действия
            subj = self._find_subject(G)
            if subj:
                S, rule = subj
                add("subject_of_pred", "up", S, path="nsubj", source="enhanced" if rule else "basic", rule=rule)

        # 5. обстоятельства при сказуемом (для подлежащего)
        if role.role == "subject" and s.tok(h).head:
            P = s.tok(h).head
            for d in s.children(P):
                if d == h:
                    continue
                r = self.rel(d)
                if r in ("advmod", "advcl") or (r == "obl" and self.role(d).role == "adverbial"):
                    add("adverbial", "up", d, detail=self.role(d).subtype + (" " + self.case_detail(d) if r == "obl" else ""), path=r)

        # 5б. зависимые глагола / сказуемого: подлежащее, дополнения, обстоятельства
        if T.upos in ("VERB", "AUX") or role.role == "predicate":
            self._add_dependents(i, add)

        # 6. однородные члены
        if g:
            for m in g.members:
                if m != i:
                    add("conj", "sibling", m, detail=g.conj_type, path="conj", negated=m in g.negated)

        # 7. предлог + падеж
        cd = self.case_detail(i)
        preps = s.children(i, {"case"})
        if preps:
            out.append(Relation("case", "down", preps[0], cd, s.tok(preps[0]).upos, detail=cd, path="case"))

        if g and i in g.negated:
            for r in out:
                if r.rel_type != "conj":
                    r.negated = True
        return out

    def _add_dependents(self, i: int, add) -> None:
        """«Пахнет тёплой теснотой, кроликами…» → подлежащее, дополнения (с однородными членами), обстоятельства."""
        s = self.s
        subj = self._find_subject(i)
        if subj:
            S, rule = subj
            inherited = rule in ("conj_inherit", "control", "gerund")
            add("subject", "down", S, path="nsubj", source="enhanced" if rule else "basic",
                scope="shared" if inherited else "direct", rule=rule)
            self._add_members(S, "subject", "", "nsubj>conj", "direct", add)
        for d, scope, rule, conf, src in self.deps_for(i, DEP_RELS):
            td = s.tok(d)
            if td.upos == "PART" or td.lemma.lower() in RELATIVE_PRON:
                continue
            drole = self.role(d)
            if drole.role == "object":
                rt, detail = "object", f"{drole.subtype} {self.case_detail(d)}".strip()
            elif drole.role == "adverbial":
                rt = "adverbial"
                detail = (drole.subtype + (" " + self.case_detail(d) if self.rel(d) == "obl" else "")).strip()
            else:
                continue
            add(rt, "down", d, detail=detail, path=td.deprel, source=src, scope=scope, rule=rule, confidence=conf)
            self._add_members(d, rt, detail, td.deprel + ">conj", scope, add)

    def _add_members(self, d: int, rt: str, detail: str, path: str, scope: str, add) -> None:
        """Остальные члены однородного ряда, вершина которого d: «теснотой, кроликами и служанками»."""
        g = self.group_of(d)
        if not g or g.head != d:
            return
        for m in g.members:
            if m != d:
                add(rt, "down", m, detail=detail, path=path, source="enhanced",
                    scope="shared" if scope == "direct" else scope, rule="conj_expand", negated=m in g.negated)

    def _add_predicate(self, out, P, i, via, add) -> None:
        s = self.s
        tp = s.tok(P)
        cops = s.children(P, {"cop"})
        if cops:
            detail = f"составное именное (связка «{self.lemma(cops[0])}»)"
        elif tp.upos in ("NOUN", "PROPN", "ADJ", "NUM", "PRON"):
            detail = "составное именное (нулевая связка)"
        else:
            detail = "глагольное"
        add("predicate", "up", P, detail=detail, path="nsubj^", source="enhanced" if via else "basic",
            scope="shared" if via else "direct", rule="conj_inherit" if via else "")
        if tp.lemma.lower() in PHASE_MODAL:
            for X in s.children(P, {"xcomp"}):
                add("predicate", "up", X, detail="составное глагольное (инфинитив)", path="nsubj^>xcomp", source="enhanced", rule="xcomp")
        gp = self.group_of(P)
        if gp:
            for Q in gp.members:
                if Q == P:
                    continue
                if any(c != i for c in s.children(Q, {"nsubj", "csubj"})):
                    continue
                ok = self._agrees(i, Q)
                if ok is False:
                    continue
                add("predicate", "up", Q, detail="однородное сказуемое", path="nsubj^>conj", source="enhanced",
                    scope="shared", rule="agreement" if ok else "conj_inherit", confidence=0.9 if ok else 0.7)

    def _find_subject(self, G: int) -> tuple[int, str] | None:
        found = self._find_subject_raw(G)
        if found is None:
            return None
        S, rule = found
        # «который» → антецедент: «запах, который шёл из сада» — субъект действия = «запах»
        if self.lemma(S) in RELATIVE_PRON:
            R = self.s.tok(S).head
            while R and self.s.tok(R).deprel == "conj":
                R = self.s.tok(R).head
            if R and self.s.tok(R).deprel == "acl:relcl" and self.s.tok(R).head:
                return self.s.tok(R).head, "relcl"
        return S, rule

    def _find_subject_raw(self, G: int) -> tuple[int, str] | None:
        s = self.s
        subj = s.children(G, {"nsubj", "csubj"})
        if subj:
            return subj[0], ""
        tg = s.tok(G)
        if tg.deprel == "xcomp" and tg.head:
            sub = s.children(tg.head, {"nsubj"})
            if sub:
                return sub[0], "control"
        if tg.deprel.startswith("advcl") and tg.feat("VerbForm") == "Conv" and tg.head:
            sub = s.children(tg.head, {"nsubj"})
            if sub:
                return sub[0], "gerund"
        g = self.group_of(G)
        if g and g.head != G:
            sub = s.children(g.head, {"nsubj"})
            if sub and self._agrees(sub[0], G) is not False:
                return sub[0], "conj_inherit"
        return None

    # ---------------------------------------------- метрики предложения

    def metrics(self) -> dict:
        s = self.s
        words = [t for t in s.tokens if t.upos != "PUNCT"]
        clauses, subord, coord, asyn = 1, 0, 0, 0
        for t in s.tokens:
            r = base_rel(t.deprel)
            finite = t.feat("VerbForm") == "Fin" or bool(s.children(t.idx, {"nsubj", "cop"}))
            if t.deprel == "acl:relcl" or (r in ("ccomp", "csubj") and finite) or (r == "advcl" and t.feat("VerbForm") != "Conv" and finite):
                clauses += 1
                subord += 1
            elif r == "parataxis" and finite:
                clauses += 1
                asyn += 1
            elif r == "conj" and s.children(t.idx, {"nsubj"}) and t.upos in ("VERB", "AUX", "ADJ", "NOUN"):
                clauses += 1
                if s.children(t.idx, {"cc"}):
                    coord += 1
                else:
                    asyn += 1
        kinds = [k for k, v in (("complex", subord), ("compound", coord), ("asyndetic", asyn)) if v]
        sent_type = "simple" if clauses == 1 else (kinds[0] if len(kinds) == 1 else "mixed")
        return {
            "n_words": len(words),
            "tree_depth": s.depth(),
            "n_clauses": clauses,
            "sent_type": sent_type,
            "has_homogeneous": bool(self.groups),
            "has_participial": any(base_rel(t.deprel) == "acl" and t.feat("VerbForm") == "Part" for t in s.tokens),
            "has_gerund": any(t.feat("VerbForm") == "Conv" for t in s.tokens),
        }


SENT_TYPE_LABELS = {
    "simple": "простое",
    "complex": "сложноподчинённое",
    "compound": "сложносочинённое",
    "asyndetic": "бессоюзное",
    "mixed": "с разными видами связи",
}
