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


def test_client_receives_one_prompt() -> None:
    """Asserts one judged answer costs exactly one judge call."""
    client = CannedJudgeClient()
    judge = LLMJudge(client)
    judge.judge_answer(make_eval_query(), make_answer(), [])
    assert len(client.prompts) == 1


def test_verdict_is_frozen() -> None:
    """Asserts verdicts are immutable."""
    judge = LLMJudge(CannedJudgeClient())
    verdict = judge.judge_answer(make_eval_query(), make_answer(), [])
    try:
        verdict.relevance = 0.1  # noqa: frozen dataclass guard
    except Exception:
        pass
    else:
        raise AssertionError("expected frozen dataclass")


def test_judge_batch_one_verdict_per_query() -> None:
    """Asserts the batch returns verdicts in input order."""
    from src.eval.judge import EvalQuery

    judge = LLMJudge(CannedJudgeClient())
    queries = [
        EvalQuery(query_id=f"q-{index}", query="q?", relevant_doc_ids=set(), reference_answer="")
        for index in range(3)
    ]
    verdicts = judge.judge_batch(queries, lambda _q: make_answer(), lambda _q: [])
    assert [verdict.query_id for verdict in verdicts] == ["q-0", "q-1", "q-2"]


def test_summarize_means() -> None:
    """Asserts summarize averages the scores."""
    from src.eval.judge import JudgeVerdict, summarize

    verdicts = [
        JudgeVerdict("q-1", 0.8, 1.0, "ok"),
        JudgeVerdict("q-2", 0.4, 0.5, "ok"),
    ]
    summary = summarize(verdicts)
    assert summary["relevance"] == 0.6
    assert summary["faithfulness"] == 0.75
