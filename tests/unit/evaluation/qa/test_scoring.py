from src.evaluation.qa.preparation import PreparationRecord
from src.evaluation.qa.scoring import build_summary, score_attempt
from src.evaluation.qa.validation import Atom, Judgment


def _judgment(atom_id, outcome):
    return Judgment(
        atom_id=atom_id,
        outcome=outcome,
        rationale="reason",
    )


class TestAttemptScoring:
    def test_required_gate_and_negative_floor(self):
        gold = [
            Atom(id="required", text="required", required=True),
            Atom(id="optional", text="optional", required=False),
        ]

        result = score_attempt(
            gold,
            [
                _judgment("required", "not_mentioned"),
                _judgment("optional", "contradicted"),
            ],
        )

        assert result == {
            "atom_score": 0.0,
            "required_atom_recall": 0.0,
            "passed": False,
        }

    def test_optional_omission_can_pass(self):
        gold = [
            Atom(id="required", text="required", required=True),
            Atom(id="optional", text="optional", required=False),
        ]

        result = score_attempt(
            gold,
            [_judgment("required", "entailed"), _judgment("optional", "not_mentioned")],
        )

        assert result["atom_score"] == 0.5
        assert result["required_atom_recall"] == 1.0
        assert result["passed"] is True


class TestSummaryScoring:
    def test_execution_failures_count_and_evaluation_failures_are_excluded(self):
        gold = Atom(id="g1", text="gold", required=True)
        preparation = [
            PreparationRecord(
                item_id="item",
                status="prepared",
                question="question",
                answer="answer",
                time_sensitive=False,
                gold_atoms=(gold,),
                atom_source="inferred",
            )
        ]
        results = [
            {
                "item_id": "item",
                "status": "scored",
                "passed": True,
                "atom_score": 1.0,
                "required_atom_recall": 1.0,
                "judgments": [{"atom_id": "g1", "outcome": "entailed"}],
            },
            {"item_id": "item", "status": "execution_failed"},
            {"item_id": "item", "status": "evaluation_failed"},
        ]

        summary = build_summary(preparation, results)

        assert summary["item_lifecycle_counts"] == {
            "skipped_time_sensitive": 0,
            "preparation_failed": 0,
            "prepared": 1,
            "skipped_live": 0,
        }
        assert summary["attempt_lifecycle_counts"] == {
            "execution_failed": 1,
            "evaluation_failed": 1,
            "scored": 1,
            "live_validation_failed": 0,
        }
        assert summary["quality_accounted_attempts"] == 2
        assert summary["overall_attempt_pass_rate"] == 0.5
        assert summary["items"][0]["k"] == 2
        assert summary["items"][0]["item_pass_rate"] == 0.5
        assert summary["items"][0]["gold_atom_pass_rates"] == [
            {"atom_id": "g1", "atom_pass_count": 1, "k": 2, "atom_pass_rate": 0.5}
        ]
