#!/usr/bin/env python3
"""
test_generate.py --- unit tests for citation-grounded generation

Contains:
    StubLLM: returns a canned completion for prompt assertions
    make_results(): builds retrieved chunks for generation tests
    test_generate_returns_answer_text(): asserts the answer passes through
    test_generate_parses_citation_markers(): asserts [n] markers become citations
    test_generate_drops_out_of_range_citations(): asserts invalid citations drop
"""

from src.generation.generate import CitationGenerator
from src.retrieval.fusion import RankedResult


class StubLLM:
    """Returns a canned completion and records prompts."""

    def __init__(self, completion: str = "The clause requires indemnification [1].") -> None:
        """Stores the canned completion.

        Args:
            completion: Text returned for every prompt.
        """
        self.completion = completion
        self.prompts: list[str] = []
        self.max_tokens_seen: list[int] = []

    def __call__(self, prompt: str, max_tokens: int = 512) -> str:
        """Records the prompt and returns the canned completion.

        Args:
            prompt: Prompt text to record.
            max_tokens: Token budget to record.

        Returns:
            completion: Canned completion text.
        """
        self.prompts.append(prompt)
        self.max_tokens_seen.append(max_tokens)
        return self.completion


def make_results(count: int = 2) -> list[RankedResult]:
    """Builds retrieved chunks for generation tests.

    Args:
        count: Number of chunks to build.

    Returns:
        results: Fused ranked results with clause text.
    """
    return [
        RankedResult(
            chunk_id=f"doc-{index}-chunk-0",
            doc_id=f"doc-{index}",
            text=f"Section {index}.1 clause text for document {index}.",
            score=0.9 - index * 0.1,
            source="fused",
        )
        for index in range(count)
    ]


def test_generate_returns_answer_text() -> None:
    """Asserts the answer passes through from the client."""
    generator = CitationGenerator(StubLLM())
    answer = generator.generate("what does section 1.1 say?", make_results())
    assert answer.answer == "The clause requires indemnification [1]."


def test_generate_parses_citation_markers() -> None:
    """Asserts [n] markers become grounded citations."""
    generator = CitationGenerator(StubLLM())
    answer = generator.generate("what does section 1.1 say?", make_results())
    assert [citation.chunk_id for citation in answer.citations] == ["doc-0-chunk-0"]


def test_generate_drops_out_of_range_citations() -> None:
    """Asserts citations pointing past the results are dropped."""
    generator = CitationGenerator(StubLLM("Grounded in thin air [9]."))
    answer = generator.generate("anything?", make_results(count=1))
    assert answer.citations == []


def test_generate_prompt_contains_sources() -> None:
    """Asserts the prompt carries the retrieved sources."""
    client = StubLLM()
    CitationGenerator(client).generate("clause?", make_results())
    assert "Section 0.1 clause text" in client.prompts[0]


def test_generate_forwards_max_tokens() -> None:
    """Asserts the token budget reaches the client."""
    client = StubLLM()
    CitationGenerator(client, max_tokens=256).generate("clause?", make_results())
    assert client.max_tokens_seen == [256]


def test_generate_empty_results() -> None:
    """Asserts generation over zero results does not crash."""
    generator = CitationGenerator(StubLLM("No sources available."))
    answer = generator.generate("clause?", [])
    assert answer.answer == "No sources available."
    assert answer.citations == []


def test_citation_quote_is_short() -> None:
    """Asserts citation quotes are trimmed to 80 characters."""
    generator = CitationGenerator(StubLLM())
    answer = generator.generate("clause?", make_results())
    assert all(len(citation.quote) <= 80 for citation in answer.citations)


def test_multiple_markers_parse_in_order() -> None:
    """Asserts multiple markers parse in order of appearance."""
    client = StubLLM("First [1], then [2].")
    generator = CitationGenerator(client)
    answer = generator.generate("clause?", make_results())
    assert [citation.chunk_id for citation in answer.citations] == [
        "doc-0-chunk-0",
        "doc-1-chunk-0",
    ]


def test_duplicate_markers_dedupe() -> None:
    """Asserts repeated markers for one source dedupe."""
    client = StubLLM("Again [1] and again [1].")
    generator = CitationGenerator(client)
    answer = generator.generate("clause?", make_results())
    assert len(answer.citations) == 1


def test_generate_retries_once_on_failure() -> None:
    """Asserts one transient failure is retried."""

    class FlakyLLM(StubLLM):
        """Fails once, then succeeds."""

        def __init__(self) -> None:
            """Creates the flaky client."""
            super().__init__()
            self.failures = 0

        def __call__(self, prompt: str, max_tokens: int = 512) -> str:
            """Fails the first call, succeeds after."""
            if self.failures == 0:
                self.failures += 1
                raise RuntimeError("transient llm error")
            return super().__call__(prompt, max_tokens)

    generator = CitationGenerator(FlakyLLM())
    answer = generator.generate("clause?", make_results())
    assert answer.answer


def test_generate_reraises_persistent_failure() -> None:
    """Asserts a persistent failure propagates after the retry."""

    class AlwaysFailsLLM(StubLLM):
        """Fails on every call."""

        def __call__(self, prompt: str, max_tokens: int = 512) -> str:
            """Always raises."""
            raise RuntimeError("llm down")

    generator = CitationGenerator(AlwaysFailsLLM())
    try:
        generator.generate("clause?", make_results())
    except RuntimeError:
        pass
    else:
        raise AssertionError("expected RuntimeError")


def test_marker_zero_drops() -> None:
    """Asserts a [0] marker drops (numbering starts at 1)."""
    client = StubLLM("Broken citation [0].")
    answer = CitationGenerator(client).generate("clause?", make_results())
    assert answer.citations == []


def test_faithfulness_full_when_grounded() -> None:
    """Asserts faithfulness is 1.0 when every citation is grounded."""
    generator = CitationGenerator(StubLLM())
    results = make_results()
    answer = generator.generate("clause?", results)
    assert generator.faithfulness(answer, results) == 1.0


def test_faithfulness_vacuous_without_citations() -> None:
    """Asserts an answer with no citations is vacuously faithful."""
    generator = CitationGenerator(StubLLM("No markers here."))
    results = make_results()
    answer = generator.generate("clause?", results)
    assert generator.faithfulness(answer, results) == 1.0


def test_prompt_includes_refusal_guidance() -> None:
    """Asserts the prompt tells the model to say when it does not know."""
    client = StubLLM()
    CitationGenerator(client).generate("clause?", make_results())
    assert "do not know" in client.prompts[0].lower()
