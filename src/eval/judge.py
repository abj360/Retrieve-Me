#!/usr/bin/env python3
"""
judge.py --- LLM-as-judge evaluation harness

Contains:
    JudgeVerdict: one scored judgement for an answer
    EvalQuery: one golden-set evaluation query
    load_eval_dataset(): loads the golden set from JSONL
    LLMJudge: scores answers with an LLM judge
"""

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

from src.generation.generate import GeneratedAnswer
from src.retrieval.fusion import RankedResult

logger = logging.getLogger(__name__)

DEFAULT_JUDGE_MODEL = "gpt-4o-mini"  # cheap, deterministic at temperature 0
FAITHFULNESS_THRESHOLD = 0.8
SCORE_PATTERN = re.compile(
    r"relevance:\s*([0-9]*\.?[0-9]+).*?faithfulness:\s*([0-9]*\.?[0-9]+)",
    re.DOTALL | re.IGNORECASE,
)

JUDGE_RUBRIC = """Score the answer on two axes from 0.0 to 1.0:
relevance: does the answer address the query?
faithfulness: is every claim grounded in the provided contexts?
Reply exactly in the form:
relevance: <score>
faithfulness: <score>
rationale: <one paragraph>"""


@dataclass(frozen=True)
class JudgeVerdict:
    """Carries one scored judgement for an answer.

    Attributes:
        query_id: Identifier of the evaluated query.
        relevance: Relevance score in [0, 1].
        faithfulness: Grounding score in [0, 1].
        rationale: Judge's explanation of the scores.
    """

    query_id: str
    relevance: float
    faithfulness: float
    rationale: str

    @property
    def is_faithful(self) -> bool:
        """Returns whether the verdict clears the faithfulness threshold.

        Returns:
            faithful: True when faithfulness meets the threshold.
        """
        return self.faithfulness >= FAITHFULNESS_THRESHOLD


@dataclass(frozen=True)
class EvalQuery:
    """Carries one golden-set evaluation query.

    Attributes:
        query_id: Stable identifier of the query.
        query: Query text.
        relevant_doc_ids: Document ids considered relevant.
        reference_answer: Golden reference answer.
    """

    query_id: str
    query: str
    relevant_doc_ids: set[str]
    reference_answer: str


def load_eval_dataset(path: str | Path) -> list[EvalQuery]:
    """Loads the golden set from JSONL.

    Args:
        path: JSONL file with one query object per line.

    Returns:
        queries: Parsed evaluation queries in file order.
    """
    queries = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        queries.append(
            EvalQuery(
                query_id=record["query_id"],
                query=record["query"],
                relevant_doc_ids=set(record["relevant_doc_ids"]),
                reference_answer=record["reference_answer"],
            )
        )
    logger.info("loaded %d eval queries from %s", len(queries), path)
    return queries


class LLMJudge:
    """Scores answers with an LLM judge.

    Attributes:
        model: Judge model identifier.
    """

    def __init__(self, llm_client, model: str = DEFAULT_JUDGE_MODEL) -> None:
        """Stores the judge client and model.

        Args:
            llm_client: Callable returning completion text for a prompt.
            model: Judge model identifier.
        """
        self.llm_client = llm_client
        self.model = model

    def judge_answer(
        self, query: EvalQuery, answer: GeneratedAnswer, contexts: list[RankedResult]
    ) -> JudgeVerdict:
        """Scores one answer with the LLM judge.

        Args:
            query: Golden-set query being evaluated.
            answer: Generated answer to score.
            contexts: Retrieved chunks the answer should be grounded in.

        Returns:
            verdict: Scored judgement for the answer.
        """
        prompt = self._build_judge_prompt(query, answer, contexts)
        raw = self.llm_client(prompt)
        return self._parse_verdict(query.query_id, raw)

    def _build_judge_prompt(
        self, query: EvalQuery, answer: GeneratedAnswer, contexts: list[RankedResult]
    ) -> str:
        """Builds the judge prompt from query, answer, and contexts.

        Args:
            query: Golden-set query being evaluated.
            answer: Generated answer to score.
            contexts: Retrieved chunks for grounding.

        Returns:
            prompt: Rubric plus query, answer, and numbered contexts.
        """
        context_block = "\n\n".join(
            f"[{index}] {context.text}" for index, context in enumerate(contexts, 1)
        )
        return (
            f"{JUDGE_RUBRIC}\n\nQuery: {query.query}\n\n"
            f"Answer: {answer.answer}\n\nContexts:\n{context_block}"
        )

    def _parse_verdict(self, query_id: str, raw: str) -> JudgeVerdict:
        """Parses the judge's reply into a verdict.

        Args:
            query_id: Identifier of the evaluated query.
            raw: Raw judge reply text.

        Returns:
            verdict: Parsed scores and rationale.
        """
        match = SCORE_PATTERN.search(raw)
        if match is None:
            logger.info("judge reply did not parse, retrying once: %r", raw[:120])
            match = SCORE_PATTERN.search(self.llm_client(self._repair_prompt(raw)))
        if match is None:
            logger.warning("judge reply still unparseable; scoring zero")
            return JudgeVerdict(query_id=query_id, relevance=0.0, faithfulness=0.0, rationale=raw)
        rationale = raw.split("rationale:", 1)[-1].strip()
        return JudgeVerdict(
            query_id=query_id,
            relevance=float(match.group(1)),
            faithfulness=float(match.group(2)),
            rationale=rationale,
        )


    def judge_batch(
        self,
        queries: list[EvalQuery],
        answer_fn,
        contexts_fn,
    ) -> list[JudgeVerdict]:
        """Scores a batch of golden-set queries.

        Args:
            queries: Golden-set queries to evaluate.
            answer_fn: Callable producing a GeneratedAnswer per query.
            contexts_fn: Callable producing retrieved chunks per query.

        Returns:
            verdicts: One verdict per query, in input order.
        """
        verdicts = []
        for query in queries:
            logger.info("judging %s", query.query_id)
            verdicts.append(
                self.judge_answer(query, answer_fn(query), contexts_fn(query))
            )
        return verdicts


def summarize(verdicts: list[JudgeVerdict]) -> dict[str, float]:
    """Aggregates verdicts into mean scores.

    Args:
        verdicts: Verdicts to aggregate.

    Returns:
        summary: Mean relevance and faithfulness across verdicts.
    """
    if not verdicts:
        return {"relevance": 0.0, "faithfulness": 0.0}
    return {
        "relevance": sum(verdict.relevance for verdict in verdicts) / len(verdicts),
        "faithfulness": sum(verdict.faithfulness for verdict in verdicts) / len(verdicts),
        "grounded_share": sum(
            1.0 for verdict in verdicts if verdict.faithfulness >= FAITHFULNESS_THRESHOLD
        )
        / len(verdicts),
    }


def main() -> None:
    """Runs the judge over a golden set and prints the summary."""
    import argparse

    parser = argparse.ArgumentParser(description="Run the LLM-as-judge eval harness")
    parser.add_argument("--dataset", required=True, type=Path, help="golden set JSONL")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    queries = load_eval_dataset(args.dataset)
    logger.info("judge harness ready: %d queries (wire llm + pipeline to run)", len(queries))


if __name__ == "__main__":
    main()


    def _repair_prompt(self, raw: str) -> str:
        """Builds a repair prompt asking the judge to reformat its reply.

        Args:
            raw: Unparseable judge reply.

        Returns:
            prompt: Reformatting instruction with the offending reply.
        """
        return (
            "Reformat the following judgement exactly as:\n"
            "relevance: <score>\nfaithfulness: <score>\nrationale: <one paragraph>\n\n"
            f"Judgement:\n{raw}"
        )
