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


def test_judge_prompt_has_rubric_and_contexts() -> None:
    """Asserts the judge prompt carries the rubric, query, answer, contexts."""
    from src.retrieval.fusion import RankedResult

    client = CannedJudgeClient()
    judge = LLMJudge(client)
    contexts = [
        RankedResult("c-1", "doc-1", "Section 3.1 clause text.", 0.9, "fused")
    ]
    judge.judge_answer(make_eval_query(), make_answer(), contexts)
    prompt = client.prompts[0]
    assert "relevance" in prompt and "faithfulness" in prompt
    assert "Section 3.1 clause text." in prompt


def test_load_eval_dataset_parses_jsonl(tmp_path) -> None:
    """Asserts the dataset loader parses JSONL into EvalQuery objects."""
    from src.eval.judge import load_eval_dataset

    target = tmp_path / "queries.jsonl"
    target.write_text(
        '{"query_id": "q-1", "query": "text?", "relevant_doc_ids": ["doc-a"], "reference_answer": "ans"}\n',
        encoding="utf-8",
    )
    (query,) = load_eval_dataset(target)
    assert query.query_id == "q-1"
    assert query.relevant_doc_ids == {"doc-a"}


def test_load_eval_dataset_skips_blank_lines(tmp_path) -> None:
    """Asserts blank JSONL lines are skipped."""
    from src.eval.judge import load_eval_dataset

    target = tmp_path / "queries.jsonl"
    target.write_text(
        '{"query_id": "q-1", "query": "text?", "relevant_doc_ids": [], "reference_answer": ""}\n\n',
        encoding="utf-8",
    )
    assert len(load_eval_dataset(target)) == 1


def test_verdict_keeps_query_id() -> None:
    """Asserts the verdict carries the query id through."""
    judge = LLMJudge(CannedJudgeClient())
    verdict = judge.judge_answer(make_eval_query(), make_answer(), [])
    assert verdict.query_id == "q-test"


def test_golden_indemnity_case_scores_high() -> None:
    """Asserts the golden indemnity case parses to the canned high scores."""
    judge = LLMJudge(CannedJudgeClient())
    verdict = judge.judge_answer(make_eval_query(), make_answer(), [])
    assert verdict.relevance >= 0.8
    assert verdict.faithfulness >= 0.8


def test_batch_empty_queries_returns_empty() -> None:
    """Asserts an empty batch yields an empty verdict list."""
    judge = LLMJudge(CannedJudgeClient())
    assert judge.judge_batch([], lambda _q: make_answer(), lambda _q: []) == []


def test_summarize_empty_returns_zeros() -> None:
    """Asserts summarizing no verdicts yields zeroed means."""
    from src.eval.judge import summarize

    assert summarize([]) == {"relevance": 0.0, "faithfulness": 0.0}


def test_judge_default_model() -> None:
    """Asserts the judge defaults to the cheap deterministic model."""
    judge = LLMJudge(CannedJudgeClient())
    assert judge.model == "gpt-4o-mini"
