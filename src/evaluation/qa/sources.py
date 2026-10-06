"""Deterministic expected-source checks over persisted tested-agent responses."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional, Sequence, Tuple

from .dataset import validate_expected_sources, validate_nonempty_string
from .tool_traces import ToolCallStatus, load_tool_call_records


class SourceOutcome(str, Enum):
    MATCHED = "matched"
    MISSING = "missing"
    UNAVAILABLE = "unavailable"


class SourceEvaluationStatus(str, Enum):
    SCORED = "scored"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class MatchingCall:
    ordinal: int
    name: str

    def to_dict(self) -> Dict[str, Any]:
        return {"ordinal": self.ordinal, "name": self.name}


@dataclass(frozen=True)
class SourceMatch:
    source: str
    outcome: SourceOutcome
    matching_calls: Tuple[MatchingCall, ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "outcome": self.outcome.value,
            "matching_calls": [call.to_dict() for call in self.matching_calls],
        }


@dataclass(frozen=True)
class SourceEvaluation:
    matches: Tuple[SourceMatch, ...]

    @classmethod
    def from_dict(cls, raw: Any) -> "SourceEvaluation":
        if not isinstance(raw, dict) or set(raw) != {"status", "recall", "matches"}:
            raise ValueError("source evaluation has invalid fields")
        if not isinstance(raw["matches"], list) or not raw["matches"]:
            raise ValueError("source evaluation requires source matches")
        matches = []
        for row in raw["matches"]:
            if not isinstance(row, dict) or set(row) != {
                "source",
                "outcome",
                "matching_calls",
            }:
                raise ValueError("source match has invalid fields")
            outcome = SourceOutcome(row["outcome"])
            if not isinstance(row["matching_calls"], list):
                raise ValueError("source matching_calls must be an array")
            calls = []
            for call in row["matching_calls"]:
                if not isinstance(call, dict) or set(call) != {"ordinal", "name"}:
                    raise ValueError("source matching call has invalid fields")
                ordinal = call["ordinal"]
                if (
                    isinstance(ordinal, bool)
                    or not isinstance(ordinal, int)
                    or ordinal <= 0
                ):
                    raise ValueError("source matching call ordinal must be positive")
                name = validate_nonempty_string(
                    call["name"], "source matching call name"
                )
                calls.append(MatchingCall(ordinal, name))
            if bool(calls) != (outcome is SourceOutcome.MATCHED):
                raise ValueError("source outcome does not match its supporting calls")
            ordinals = [call.ordinal for call in calls]
            if ordinals != sorted(set(ordinals)):
                raise ValueError("source matching calls must be unique and ordered")
            matches.append(SourceMatch(row["source"], outcome, tuple(calls)))
        validate_expected_sources(
            [match.source for match in matches], context="source evaluation matches"
        )
        evaluation = cls(tuple(matches))
        if (
            SourceEvaluationStatus(raw["status"]) is not evaluation.status
            or isinstance(raw["recall"], bool)
            or raw["recall"] != evaluation.recall
        ):
            raise ValueError("source evaluation metrics disagree with matches")
        return evaluation

    @property
    def status(self) -> SourceEvaluationStatus:
        return (
            SourceEvaluationStatus.UNAVAILABLE
            if any(match.outcome is SourceOutcome.UNAVAILABLE for match in self.matches)
            else SourceEvaluationStatus.SCORED
        )

    @property
    def recall(self) -> Optional[float]:
        if self.status is SourceEvaluationStatus.UNAVAILABLE:
            return None
        return sum(
            match.outcome is SourceOutcome.MATCHED for match in self.matches
        ) / len(self.matches)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "recall": self.recall,
            "matches": [match.to_dict() for match in self.matches],
        }


def evaluate_sources(
    expected_sources: Sequence[str], tool_calls: Any
) -> SourceEvaluation:
    """Match literal strings, never inputs, final answers, or oracle evidence.

    Missing historical responses leave unmatched expectations unknown. A match
    in an available response remains proven even when other evidence is absent.
    """
    if not expected_sources:
        raise ValueError("source evaluation requires at least one expected source")
    calls = (
        load_tool_call_records(tool_calls, context="source evaluation tool_calls")
        if tool_calls is not None
        else []
    )
    successful = [call for call in calls if call.status is ToolCallStatus.SUCCESS]
    incomplete_evidence = tool_calls is None or any(
        call.response is None for call in successful
    )
    responses = [
        (call, call.response.casefold())
        for call in successful
        if call.response is not None
    ]
    matches = []
    for source in expected_sources:
        needle = source.casefold()
        matching_calls = tuple(
            MatchingCall(call.ordinal, call.name)
            for call, response in responses
            if needle in response
        )
        outcome = (
            SourceOutcome.MATCHED
            if matching_calls
            else (
                SourceOutcome.UNAVAILABLE
                if incomplete_evidence
                else SourceOutcome.MISSING
            )
        )
        matches.append(SourceMatch(source, outcome, matching_calls))
    return SourceEvaluation(tuple(matches))
