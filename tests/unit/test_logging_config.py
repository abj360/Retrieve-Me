#!/usr/bin/env python3
"""
test_logging_config.py --- covers RETRIEVAL_LOG_LEVEL reaching the loggers

Contains:
    restore_levels(): puts the touched loggers back after each test
    test_level_is_applied_to_service_loggers(): asserts the setting takes effect
    test_debug_level_is_applied(): asserts a lower level also lands
    test_unknown_level_leaves_levels_untouched(): asserts a typo fails safe
    test_unknown_level_warns(): asserts the typo is reported, not swallowed
"""

import logging

import pytest

from src.api.main import configure_logging


@pytest.fixture(autouse=True)
def restore_levels():
    """Restores the log levels this module changes."""
    names = ("retrieval", "src")
    before = {name: logging.getLogger(name).level for name in names}
    root_before = logging.getLogger().level
    yield
    for name, level in before.items():
        logging.getLogger(name).setLevel(level)
    logging.getLogger().setLevel(root_before)


def test_level_is_applied_to_service_loggers() -> None:
    """Asserts the configured level reaches the loggers under src/."""
    logging.getLogger("src").setLevel(logging.WARNING)
    configure_logging("INFO")
    assert logging.getLogger("src").isEnabledFor(logging.INFO)
    assert logging.getLogger("retrieval").isEnabledFor(logging.INFO)


def test_debug_level_is_applied() -> None:
    """Asserts a level below INFO lands too."""
    configure_logging("DEBUG")
    assert logging.getLogger("src.ingest.loader").isEnabledFor(logging.DEBUG)


def test_unknown_level_leaves_levels_untouched() -> None:
    """Asserts a typo'd level does not silently drop everything."""
    logging.getLogger("src").setLevel(logging.INFO)
    configure_logging("VERBOSE")
    assert logging.getLogger("src").isEnabledFor(logging.INFO)


def test_unknown_level_warns(caplog) -> None:
    """Asserts an unknown level is reported rather than passing quietly."""
    with caplog.at_level(logging.WARNING, logger="retrieval.startup"):
        configure_logging("VERBOSE")
    assert "unknown RETRIEVAL_LOG_LEVEL" in caplog.text
