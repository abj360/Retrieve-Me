#!/usr/bin/env python3
"""
test_config_loader.py --- unit tests for the YAML pipeline config loader

Contains:
    write_config(): writes a config dict to a temp YAML file
    test_loads_full_config(): asserts a full YAML loads into typed sections
    test_missing_required_section_raises(): asserts validation fails closed
"""

import pytest
import yaml

from src.config.pipeline import ConfigError, load_pipeline_config

FULL_CONFIG = {
    "pipeline": {"name": "test-pipeline"},
    "embedder": {"model_name": "test-model", "batch_size": 8},
    "sparse": {"enabled": True, "k1": 1.2, "b": 0.6},
    "dense": {"enabled": True, "collection": "test-chunks"},
    "fusion": {"rrf_k": 42, "sparse_weight": 1.1, "dense_weight": 0.9},
    "reranker": {"model_name": "test-reranker", "top_k": 7},
    "chunking": {"max_tokens": 128, "overlap_tokens": 16},
    "cache": {"ttl_seconds": 60},
}


def write_config(tmp_path, config: dict):
    """Writes a config dict to a temp YAML file.

    Args:
        tmp_path: Pytest temp directory.
        config: Config mapping to serialize.

    Returns:
        path: Path of the written YAML file.
    """
    target = tmp_path / "pipeline.yaml"
    target.write_text(yaml.safe_dump(config), encoding="utf-8")
    return target


def test_loads_full_config(tmp_path) -> None:
    """Asserts a full YAML loads into typed sections."""
    config = load_pipeline_config(write_config(tmp_path, FULL_CONFIG))
    assert config.embedder.model_name == "test-model"
    assert config.fusion.rrf_k == 42
    assert config.reranker.top_k == 7


def test_missing_required_section_raises(tmp_path) -> None:
    """Asserts a missing required section fails validation."""
    broken = {key: value for key, value in FULL_CONFIG.items() if key != "reranker"}
    with pytest.raises(ConfigError):
        load_pipeline_config(write_config(tmp_path, broken))


def test_optional_sections_use_defaults(tmp_path) -> None:
    """Asserts omitted optional sections fall back to defaults."""
    minimal = {
        "embedder": {"model_name": "m"},
        "reranker": {"model_name": "r"},
        "sparse": {},
        "dense": {},
        "fusion": {},
    }
    config = load_pipeline_config(write_config(tmp_path, minimal))
    assert config.chunking.max_tokens == 512
    assert config.cache.ttl_seconds == 300
