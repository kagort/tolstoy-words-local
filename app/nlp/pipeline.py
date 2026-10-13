"""Конвейер: нормализованный текст → разобранные предложения с метриками."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass

from app.nlp.normalize import paragraphs
from app.nlp.parsers import Parser
from app.nlp.segment import SegSentence, split_paragraph
from app.nlp.syntax import SentenceAnalysis
from app.nlp.tree import Sentence, Token

MAX_TOKENS_PER_SENTENCE = 300  # защита от «предложений» без точек на целую страницу


@dataclass
class ProcessedSentence:
    paragraph_idx: int
    sentence: Sentence
    is_direct_speech: bool
    metrics: dict


def _segments(text: str) -> Iterator[tuple[int, SegSentence]]:
    for p_idx, par in enumerate(paragraphs(text)):
        for seg in split_paragraph(par):
            toks = seg.tokens
            if len(toks) <= MAX_TOKENS_PER_SENTENCE:
                yield p_idx, seg
                continue
            # слишком длинное «предложение» режем на куски
            for k in range(0, len(toks), MAX_TOKENS_PER_SENTENCE):
                part = toks[k : k + MAX_TOKENS_PER_SENTENCE]
                base = part[0].start
                text_part = seg.text[base : part[-1].end]
                for t in part:
                    t.start -= base
                    t.end -= base
                yield p_idx, SegSentence(text_part, part, seg.is_direct_speech)


def count_segments(text: str) -> int:
    return sum(1 for _ in _segments(text))


def process_text(
    text: str,
    parser: Parser,
    batch_size: int = 64,
    on_batch: Callable[[int], None] | None = None,
) -> Iterator[list[ProcessedSentence]]:
    """Разбирает текст пачками; отдаёт списки обработанных предложений."""
    batch: list[tuple[int, SegSentence]] = []
    done = 0

    def flush() -> list[ProcessedSentence]:
        parsed = parser.parse([[t.text for t in seg.tokens] for _, seg in batch])
        out = []
        for (p_idx, seg), words in zip(batch, parsed):
            tokens = []
            for i, (st, w) in enumerate(zip(seg.tokens, words), start=1):
                tokens.append(Token(i, st.text, w.lemma, w.upos, w.feats, w.head, w.deprel, st.start, st.end))
            sent = Sentence(tokens, seg.text)
            metrics = SentenceAnalysis(sent).metrics()
            out.append(ProcessedSentence(p_idx, sent, seg.is_direct_speech, metrics))
        return out

    for item in _segments(text):
        batch.append(item)
        if len(batch) >= batch_size:
            result = flush()
            done += len(batch)
            batch = []
            if on_batch:
                on_batch(done)
            yield result
    if batch:
        result = flush()
        done += len(batch)
        if on_batch:
            on_batch(done)
        yield result
