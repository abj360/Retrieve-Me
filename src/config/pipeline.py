#!/usr/bin/env python3
"""
pipeline.py --- config-driven YAML pipeline loader

Contains:
    ConfigError: raised for missing or invalid pipeline configuration
    EmbedderSection: dense embedder settings
    SparseSection: BM25 sparse index settings
    DenseSection: dense vector index settings
    FusionSection: RRF fusion settings
    RerankerSection: cross-encoder reranker settings
    ChunkSection: chunking settings
    CacheSection: query cache settings
    StrategySection: retrieval strategy selection settings
    GenerationSection: citation-grounded generation settings
    EvalSection: evaluation harness settings
    PipelineConfig: typed view over the YAML pipeline definition
    load_pipeline_config(): loads, validates, and types a pipeline YAML file
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

REQUIRED_SECTIONS = ("embedder", "sparse", "dense", "fusion", "reranker")


class ConfigError(ValueError):
    """Raised when the pipeline configuration is missing or invalid."""


@dataclass(frozen=True)
class EmbedderSection:
    """Carries dense embedder settings.

    Attributes:
        model_name: Sentence-transformers model identifier.
        batch_size: Texts encoded per forward pass.
    """

    model_name: str
    batch_size: int = 32


@dataclass(frozen=True)
class SparseSection:
    """Carries BM25 sparse index settings.

    Attributes:
        enabled: Whether the sparse leg runs.
        k1: BM25 term-frequency saturation parameter.
        b: BM25 length-normalization parameter.
    """

    enabled: bool = True
    k1: float = 1.5
    b: float = 0.75


@dataclass(frozen=True)
class DenseSection:
    """Carries dense vector index settings.

    Attributes:
        enabled: Whether the dense leg runs.
        collection: Qdrant collection name.
    """

    enabled: bool = True
    collection: str = "chunks"


@dataclass(frozen=True)
class FusionSection:
    """Carries RRF fusion settings.

    Attributes:
        rrf_k: Reciprocal-rank-fusion smoothing constant.
        sparse_weight: Weight applied to the sparse leg.
        dense_weight: Weight applied to the dense leg.
        normalize_scores: Whether to min-max normalize each leg pre-fusion.
    """

    rrf_k: int = 60
    sparse_weight: float = 1.0
    dense_weight: float = 1.0
    normalize_scores: bool = True


@dataclass(frozen=True)
class RerankerSection:
    """Carries cross-encoder reranker settings.

    Attributes:
        model_name: Cross-encoder model identifier.
        top_k: Candidates kept after reranking.
    """

    model_name: str
    top_k: int = 20


@dataclass(frozen=True)
class ChunkSection:
    """Carries chunking settings.

    Attributes:
        max_tokens: Maximum tokens per chunk.
        overlap_tokens: Tokens shared between consecutive chunks.
    """

    max_tokens: int = 512
    overlap_tokens: int = 64


@dataclass(frozen=True)
class CacheSection:
    """Carries query cache settings.

    Attributes:
        ttl_seconds: Time-to-live for cached responses.
        key_version: Cache key namespace; bump to invalidate old entries.
    """

    ttl_seconds: int = 300
    key_version: int = 1


@dataclass(frozen=True)
class StrategySection:
    """Carries retrieval strategy selection settings.

    Attributes:
        type: Strategy name resolved through the strategy registry.
        candidate_k: Candidates fused before reranking.
    """

    type: str = "hybrid"
    candidate_k: int = 50


@dataclass(frozen=True)
class GenerationSection:
    """Carries citation-grounded generation settings.

    Attributes:
        model_name: LLM identifier for grounded answers.
        max_tokens: Maximum tokens per generated answer.
        temperature: Sampling temperature; zero for deterministic output.
    """

    model_name: str = "gpt-4o-mini"
    max_tokens: int = 512
    temperature: float = 0.0


@dataclass(frozen=True)
class EvalSection:
    """Carries evaluation harness settings.

    Attributes:
        judge_model: LLM identifier used by the judge.
        ndcg_k: Cutoff for nDCG scoring.
        recall_k: Cutoff for recall scoring.
    """

    judge_model: str = "gpt-4o-mini"
    ndcg_k: int = 10
    recall_k: int = 50


@dataclass(frozen=True)
class PipelineConfig:
    """Carries the typed pipeline definition.

    Attributes:
        embedder: Dense embedder settings.
        sparse: Sparse index settings.
        dense: Dense index settings.
        fusion: Fusion settings.
        reranker: Reranker settings.
        chunking: Chunking settings.
        cache: Cache settings.
        strategy: Strategy selection settings.
        generation: Generation settings.
        eval: Evaluation harness settings.
    """

    embedder: EmbedderSection
    sparse: SparseSection
    dense: DenseSection
    fusion: FusionSection
    reranker: RerankerSection
    chunking: ChunkSection
    cache: CacheSection
    strategy: StrategySection
    generation: GenerationSection
    eval: EvalSection


def load_pipeline_config(path: str | Path) -> PipelineConfig:
    """Loads and validates a pipeline YAML file.

    Args:
        path: Location of the pipeline YAML definition.

    Returns:
        config: Typed pipeline configuration.
    """
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ConfigError(f"pipeline config at {path} is not a mapping")
    missing = [section for section in REQUIRED_SECTIONS if section not in raw]
    if missing:
        raise ConfigError(f"pipeline config at {path} is missing sections: {missing}")
    known = set(REQUIRED_SECTIONS) | {"pipeline", "chunking", "cache", "strategy", "generation", "eval", "logging"}
    unknown = sorted(set(raw) - known)
    if unknown:
        logger.warning("ignoring unknown pipeline config sections: %s", unknown)
    return PipelineConfig(
        embedder=EmbedderSection(**raw["embedder"]),
        sparse=SparseSection(**raw.get("sparse", {})),
        dense=DenseSection(**raw.get("dense", {})),
        fusion=FusionSection(**raw.get("fusion", {})),
        reranker=RerankerSection(**raw["reranker"]),
        chunking=ChunkSection(**raw.get("chunking", {})),
        cache=CacheSection(**raw.get("cache", {})),
        strategy=StrategySection(**raw.get("strategy", {})),
        generation=GenerationSection(**raw.get("generation", {})),
        eval=EvalSection(**raw.get("eval", {})),
    )
