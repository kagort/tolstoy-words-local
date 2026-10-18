"""Модель данных (§7 ТЗ, упрощённая для MVP)."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    BigInteger,
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    SmallInteger,
    String,
    Table,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db import Base

corpus_texts = Table(
    "corpus_texts",
    Base.metadata,
    Column("corpus_id", ForeignKey("corpora.id", ondelete="CASCADE"), primary_key=True),
    Column("text_id", ForeignKey("texts.id", ondelete="CASCADE"), primary_key=True),
)


class TextDoc(Base):
    __tablename__ = "texts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    title: Mapped[str] = mapped_column(String(500))
    author: Mapped[str | None] = mapped_column(String(500))
    year_from: Mapped[int | None] = mapped_column(Integer)
    year_to: Mapped[int | None] = mapped_column(Integer)
    genre: Mapped[str | None] = mapped_column(String(255))
    source_type: Mapped[str] = mapped_column(String(32), default="natural")  # natural | generated | other
    generator: Mapped[str | None] = mapped_column(Text)  # модель/промпт для сгенерированных текстов
    meta: Mapped[dict] = mapped_column(JSONB, default=dict)
    source_filename: Mapped[str | None] = mapped_column(String(500))
    raw_text: Mapped[str] = mapped_column(Text)
    norm_text: Mapped[str | None] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(32), default="uploaded")  # uploaded|processing|ready|error
    error: Mapped[str | None] = mapped_column(Text)
    pipeline_version: Mapped[str | None] = mapped_column(String(100))
    n_sentences: Mapped[int] = mapped_column(Integer, default=0)
    n_tokens: Mapped[int] = mapped_column(Integer, default=0)
    n_words: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    corpora: Mapped[list[Corpus]] = relationship(secondary=corpus_texts, back_populates="texts")

    @property
    def years(self) -> str:
        if self.year_from and self.year_to and self.year_to != self.year_from:
            return f"{self.year_from}–{self.year_to}"
        return str(self.year_from or "")


class Corpus(Base):
    __tablename__ = "corpora"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(255), unique=True)
    description: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    texts: Mapped[list[TextDoc]] = relationship(secondary=corpus_texts, back_populates="corpora", order_by="TextDoc.id")


# уровни разбора предложения (двухступенчатая обработка, волна 1)
PARSE_FAST = 1  # предварительный разбор быстрым парсером (Natasha) — для статистики всего текста
PARSE_PRECISE = 2  # разбор основным парсером (NLP_BACKEND, обычно Stanza) — по нему ищутся вхождения словарей


class SentenceRow(Base):
    __tablename__ = "sentences"
    __table_args__ = (UniqueConstraint("text_id", "idx"), Index("ix_sentences_text_level", "text_id", "parse_level"))

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    text_id: Mapped[int] = mapped_column(ForeignKey("texts.id", ondelete="CASCADE"), index=True)
    idx: Mapped[int] = mapped_column(Integer)
    paragraph_idx: Mapped[int] = mapped_column(Integer)
    text: Mapped[str] = mapped_column(Text)
    n_words: Mapped[int] = mapped_column(Integer)
    tree_depth: Mapped[int] = mapped_column(SmallInteger)
    n_clauses: Mapped[int] = mapped_column(SmallInteger)
    sent_type: Mapped[str] = mapped_column(String(16))
    is_direct_speech: Mapped[bool] = mapped_column(Boolean, default=False)
    has_homogeneous: Mapped[bool] = mapped_column(Boolean, default=False)
    has_participial: Mapped[bool] = mapped_column(Boolean, default=False)
    has_gerund: Mapped[bool] = mapped_column(Boolean, default=False)
    parse_level: Mapped[int] = mapped_column(SmallInteger, default=PARSE_PRECISE, server_default=str(PARSE_PRECISE))


class TokenRow(Base):
    __tablename__ = "tokens"
    __table_args__ = (
        UniqueConstraint("sentence_id", "idx"),
        Index("ix_tokens_lemma_norm", "lemma_norm"),
        Index("ix_tokens_form_norm", "form_norm"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    sentence_id: Mapped[int] = mapped_column(ForeignKey("sentences.id", ondelete="CASCADE"))
    text_id: Mapped[int] = mapped_column(Integer, index=True)
    idx: Mapped[int] = mapped_column(SmallInteger)
    form: Mapped[str] = mapped_column(Text)
    form_norm: Mapped[str] = mapped_column(Text)
    lemma: Mapped[str] = mapped_column(Text)
    lemma_norm: Mapped[str] = mapped_column(Text)
    upos: Mapped[str] = mapped_column(String(8))
    feats: Mapped[str] = mapped_column(Text)  # формат CoNLL-U: Case=Nom|Number=Sing
    head: Mapped[int] = mapped_column(SmallInteger)
    deprel: Mapped[str] = mapped_column(String(32))
    start_char: Mapped[int] = mapped_column(Integer)
    end_char: Mapped[int] = mapped_column(Integer)


class Dictionary(Base):
    __tablename__ = "dictionaries"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(255), unique=True)
    description: Mapped[str | None] = mapped_column(Text)
    # include_participles / include_gerunds / merge_propn
    settings: Mapped[dict] = mapped_column(JSONB, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    matched_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    entries: Mapped[list[DictionaryEntry]] = relationship(
        back_populates="dictionary", cascade="all, delete-orphan", order_by="DictionaryEntry.id"
    )


class DictionaryEntry(Base):
    __tablename__ = "dictionary_entries"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    dictionary_id: Mapped[int] = mapped_column(ForeignKey("dictionaries.id", ondelete="CASCADE"), index=True)
    surface: Mapped[str] = mapped_column(String(255))
    lemma: Mapped[str] = mapped_column(String(255))
    lemma_norm: Mapped[str] = mapped_column(String(255), index=True)
    upos: Mapped[str | None] = mapped_column(String(8))
    match_mode: Mapped[str] = mapped_column(String(16), default="lemma")  # lemma | forms
    category: Mapped[str | None] = mapped_column(String(255))
    comment: Mapped[str | None] = mapped_column(Text)
    verified: Mapped[bool] = mapped_column(Boolean, default=False)
    ambiguous: Mapped[bool] = mapped_column(Boolean, default=False)
    options: Mapped[list] = mapped_column(JSONB, default=list)

    dictionary: Mapped[Dictionary] = relationship(back_populates="entries")


class Hit(Base):
    """Вхождение слова словаря в корпус (с ролью токена)."""

    __tablename__ = "dictionary_hits"
    __table_args__ = (
        UniqueConstraint("entry_id", "token_id"),
        Index("ix_hits_dict_text", "dictionary_id", "text_id"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    entry_id: Mapped[int] = mapped_column(ForeignKey("dictionary_entries.id", ondelete="CASCADE"), index=True)
    dictionary_id: Mapped[int] = mapped_column(Integer)
    token_id: Mapped[int] = mapped_column(ForeignKey("tokens.id", ondelete="CASCADE"))
    sentence_id: Mapped[int] = mapped_column(BigInteger, index=True)
    text_id: Mapped[int] = mapped_column(Integer)
    token_idx: Mapped[int] = mapped_column(SmallInteger)
    form: Mapped[str] = mapped_column(Text)
    role: Mapped[str] = mapped_column(String(32))
    role_subtype: Mapped[str | None] = mapped_column(String(64))
    role_detail: Mapped[str | None] = mapped_column(String(64))
    in_coord_group: Mapped[bool] = mapped_column(Boolean, default=False)
    is_valid: Mapped[bool | None] = mapped_column(Boolean)
    syntax_version: Mapped[str] = mapped_column(String(16))


class TokenRelation(Base):
    """Связь вхождения с другим словом предложения (§5.3, §5.6)."""

    __tablename__ = "token_relations"
    __table_args__ = (
        Index("ix_rel_entry_type", "entry_id", "rel_type", "related_lemma"),
        Index("ix_rel_dict_text", "dictionary_id", "text_id"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    hit_id: Mapped[int] = mapped_column(ForeignKey("dictionary_hits.id", ondelete="CASCADE"), index=True)
    dictionary_id: Mapped[int] = mapped_column(Integer)
    entry_id: Mapped[int] = mapped_column(Integer)
    text_id: Mapped[int] = mapped_column(Integer)
    rel_type: Mapped[str] = mapped_column(String(32))
    direction: Mapped[str] = mapped_column(String(8))
    related_idx: Mapped[int] = mapped_column(SmallInteger)
    related_lemma: Mapped[str] = mapped_column(Text)
    related_upos: Mapped[str | None] = mapped_column(String(8))
    detail: Mapped[str | None] = mapped_column(Text)
    path: Mapped[str | None] = mapped_column(String(64))
    source: Mapped[str] = mapped_column(String(16))
    scope: Mapped[str] = mapped_column(String(16))
    rule: Mapped[str | None] = mapped_column(String(32))
    confidence: Mapped[float] = mapped_column(Float, default=1.0)
    negated: Mapped[bool] = mapped_column(Boolean, default=False)


class Entity(Base):
    """Упоминание имени собственного (NER, Natasha/Slovnet): персона, место, организация или иное название."""

    __tablename__ = "entities"
    __table_args__ = (Index("ix_entities_text_type", "text_id", "etype"), Index("ix_entities_norm", "norm"))

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    text_id: Mapped[int] = mapped_column(ForeignKey("texts.id", ondelete="CASCADE"))
    sentence_id: Mapped[int] = mapped_column(ForeignKey("sentences.id", ondelete="CASCADE"), index=True)
    etype: Mapped[str] = mapped_column(String(8))  # PER | LOC | ORG
    start_char: Mapped[int] = mapped_column(Integer)
    end_char: Mapped[int] = mapped_column(Integer)
    surface: Mapped[str] = mapped_column(Text)
    norm: Mapped[str] = mapped_column(Text)  # начальная форма: «Андрею Болконскому» → «Андрей Болконский»


class EntityOverride(Base):
    """Ручная правка имени: другой класс, слияние с другим именем (варианты одного героя) или скрытие ошибки."""

    __tablename__ = "entity_overrides"

    norm: Mapped[str] = mapped_column(Text, primary_key=True)
    etype: Mapped[str | None] = mapped_column(String(8))
    canonical: Mapped[str | None] = mapped_column(Text)
    hidden: Mapped[bool] = mapped_column(Boolean, default=False, server_default="false")


class StopList(Base):
    """Свой список стоп-слов (встроенный хранится в файле app/data/stopwords_ru.txt)."""

    __tablename__ = "stopword_lists"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(255), unique=True)
    description: Mapped[str | None] = mapped_column(Text)
    words: Mapped[list] = mapped_column(JSONB, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class Job(Base):
    __tablename__ = "jobs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    kind: Mapped[str] = mapped_column(String(32))  # ingest | match
    status: Mapped[str] = mapped_column(String(16), default="queued", index=True)  # queued|running|done|error|cancelled
    params: Mapped[dict] = mapped_column(JSONB, default=dict)
    title: Mapped[str] = mapped_column(String(500), default="")
    progress: Mapped[float] = mapped_column(Float, default=0.0)
    message: Mapped[str | None] = mapped_column(Text)
    error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    # «пульс» обработчика: обновляется при каждом шаге; по нему находят задачи упавших обработчиков
    heartbeat_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    attempts: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    cancel_requested: Mapped[bool] = mapped_column(Boolean, default=False, server_default="false")
