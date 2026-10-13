"""Структуры данных для разобранного предложения (близко к CoNLL-U)."""

from __future__ import annotations

from dataclasses import dataclass, field


def norm(s: str) -> str:
    """Нормализованная форма для поиска: нижний регистр, ё → е."""
    return s.lower().replace("ё", "е")


def parse_feats(s: str | None) -> dict[str, str]:
    if not s or s == "_":
        return {}
    out = {}
    for part in s.split("|"):
        if "=" in part:
            k, v = part.split("=", 1)
            out[k] = v
    return out


def format_feats(d: dict[str, str]) -> str:
    return "|".join(f"{k}={d[k]}" for k in sorted(d)) if d else "_"


@dataclass
class Token:
    idx: int  # 1..N внутри предложения
    form: str
    lemma: str
    upos: str
    feats: dict[str, str] = field(default_factory=dict)
    head: int = 0  # 0 = корень
    deprel: str = "root"
    start: int = 0  # смещение начала в тексте предложения
    end: int = 0

    @property
    def lemma_norm(self) -> str:
        return norm(self.lemma)

    @property
    def form_norm(self) -> str:
        return norm(self.form)

    def feat(self, name: str) -> str | None:
        return self.feats.get(name)


@dataclass
class Sentence:
    tokens: list[Token]
    text: str = ""

    def __post_init__(self) -> None:
        self._children: dict[int, list[int]] | None = None

    def tok(self, idx: int) -> Token:
        return self.tokens[idx - 1]

    def __len__(self) -> int:
        return len(self.tokens)

    def children(self, idx: int, rels: set[str] | None = None) -> list[int]:
        if self._children is None:
            ch: dict[int, list[int]] = {}
            for t in self.tokens:
                ch.setdefault(t.head, []).append(t.idx)
            self._children = ch
        out = self._children.get(idx, [])
        if rels is None:
            return out
        return [c for c in out if base_rel(self.tok(c).deprel) in rels or self.tok(c).deprel in rels]

    def root(self) -> int:
        for t in self.tokens:
            if t.head == 0:
                return t.idx
        return 1

    def depth(self) -> int:
        best = 0
        for t in self.tokens:
            d, cur, seen = 0, t.idx, set()
            while cur != 0 and cur not in seen:
                seen.add(cur)
                cur = self.tok(cur).head
                d += 1
            best = max(best, d)
        return best

    def to_conllu(self, sent_id: str | int | None = None) -> str:
        lines = []
        if sent_id is not None:
            lines.append(f"# sent_id = {sent_id}")
        if self.text:
            lines.append(f"# text = {self.text}")
        for t in self.tokens:
            lines.append(
                "\t".join(
                    [str(t.idx), t.form, t.lemma, t.upos, "_", format_feats(t.feats), str(t.head), t.deprel, "_", "_"]
                )
            )
        return "\n".join(lines) + "\n"


def base_rel(rel: str) -> str:
    """nsubj:pass → nsubj, acl:relcl → acl (подтип отбрасывается)."""
    return rel.split(":", 1)[0]


def sentence_from_rows(rows: list[tuple], text: str = "") -> Sentence:
    """Быстрое построение предложения (используется в тестах).

    rows: (form, lemma, upos, feats_str, head, deprel)
    """
    toks = []
    for i, (form, lemma, upos, feats, head, deprel) in enumerate(rows, start=1):
        toks.append(Token(i, form, lemma, upos, parse_feats(feats), head, deprel))
    return Sentence(toks, text or " ".join(r[0] for r in rows))
