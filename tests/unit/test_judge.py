#!/usr/bin/env python3
"""
test_judge.py --- golden-set regression tests for the LLM-as-judge harness

Contains:
    CannedJudgeClient: returns a fixed judge reply per prompt shape
    make_eval_query(): builds a golden-set query
    make_answer(): builds a generated answer
    test_judge_parses_scores(): asserts scores parse from the judge reply
    test_judge_unparseable_scores_zero(): asserts garbage replies score zero
"""

from src.eval.judge import EvalQuery, LLMJudge
from src.generation.generate import GeneratedAnswer

CANNED_REPLY = """relevance: 0.9
faithfulness: 1.0
rationale: The answer addresses the query and cites the sources correctly."""


class CannedJudgeClient:
    """Returns a fixed judge reply and records prompts."""

    def __init__(self, reply: str = CANNED_REPLY) -> None:
        """Stores the canned reply.

        Args:
            reply: Text returned for every prompt.
        """
        self.reply = reply
        self.prompts: list[str] = []

    def __call__(self, prompt: str) -> str:
        """Records the prompt and returns the canned reply.

        Args:
            prompt: Prompt text to record.

        Returns:
            reply: Canned judge reply.
        """
        self.prompts.append(prompt)
        return self.reply


def make_eval_query() -> EvalQuery:
    """Builds a golden-set query.

    Returns:
        query: Evaluation query for judge tests.
    """
    return EvalQuery(
        query_id="q-test",
        query="What does Section 3.1 say?",
        relevant_doc_ids={"license-agreement"},
        reference_answer="It requires indemnification.",
    )


def make_answer() -> GeneratedAnswer:
    """Builds a generated answer.

    Returns:
        answer: Generated answer with one citation marker.
    """
    return GeneratedAnswer(answer="Section 3.1 requires indemnification [1].", citations=[])


def test_judge_parses_scores() -> None:
    """Asserts scores parse from the judge reply."""
    judge = LLMJudge(CannedJudgeClient())
    verdict = judge.judge_answer(make_eval_query(), make_answer(), [])
    assert verdict.relevance == 0.9
    assert verdict.faithfulness == 1.0
    assert "cites" in verdict.rationale


def test_judge_unparseable_scores_zero() -> None:
    """Asserts garbage replies score zero rather than crash."""
    judge = LLMJudge(CannedJudgeClient(reply="I cannot score this."))
    verdict = judge.judge_answer(make_eval_query(), make_answer(), [])
    assert verdict.relevance == 0.0
    assert verdict.faithfulness == 0.0
