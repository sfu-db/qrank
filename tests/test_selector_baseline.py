import json
from pathlib import Path
import tempfile
import unittest

from qrank.baselines.selector.config import load_selector_config
from qrank.baselines.selector.pipeline import _compute_metrics, run_selection
from qrank.llms import LLMBackend


class FixtureBackend(LLMBackend):
    def __init__(self, responses):
        super().__init__("fixture")
        self.responses = list(responses)

    def generate_batch(self, requests):
        out = self.responses[: len(requests)]
        self.responses = self.responses[len(requests):]
        return out

    def generate(self, **request):
        raise AssertionError("generate() should not be called")


class SelectorBaselineTests(unittest.TestCase):
    def test_load_selector_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cfg.json"
            path.write_text(json.dumps({
                "output_dir": "dataset/out",
                "dataset": {"name": "bird", "path": "dataset/processed/bird", "split": "test"},
                "oracle_confidence_threshold": 0.5,
                "oracle_budget_ratio": 0.25,
            }))
            cfg = load_selector_config(path)
            self.assertEqual(cfg.output_dir, "dataset/out")
            self.assertEqual(cfg.dataset.name, "bird")
            self.assertEqual(cfg.dataset.split, "test")
            self.assertEqual(cfg.oracle_confidence_threshold, 0.5)
            self.assertEqual(cfg.oracle_budget_ratio, 0.25)

    def test_selector_uses_model_index_when_valid(self):
        rows = [{
            "question_id": "q1",
            "question": "Q",
            "candidates": [
                {"text": "SELECT 1", "invalid_mask": False},
                {"text": "SELECT 2", "invalid_mask": False},
            ],
        }]
        backend = FixtureBackend(['{"selected_index": 1, "confidence": 0.8, "reason": "better"}'])
        result = run_selection(
            rows,
            backend,
            batch_size=1,
            oracle_confidence_threshold=0.0,
            oracle_budget_ratio=0.0,
            oracle_budget_count=None,
            show_progress=False,
        )
        self.assertEqual(result[0].status, "ok")
        self.assertEqual(result[0].selected_index, 1)
        self.assertAlmostEqual(result[0].confidence, 0.8)
        self.assertEqual(result[0].final_source, "llm")

    def test_selector_falls_back_on_invalid_model_output(self):
        rows = [{
            "question_id": "q2",
            "question": "Q",
            "candidates": [
                {"text": "", "invalid_mask": False},
                {"text": "SELECT 2", "invalid_mask": False},
                {"text": "SELECT 3", "invalid_mask": True},
            ],
        }]
        backend = FixtureBackend(["not json"])
        result = run_selection(
            rows,
            backend,
            batch_size=1,
            oracle_confidence_threshold=0.0,
            oracle_budget_ratio=0.0,
            oracle_budget_count=None,
            show_progress=False,
        )
        self.assertEqual(result[0].status, "fallback")
        self.assertEqual(result[0].selected_index, 1)

    def test_selector_can_defer_to_oracle_and_budget_is_enforced(self):
        rows = [
            {
                "question_id": "q1",
                "question": "Q1",
                "candidates": [{"text": "A", "invalid_mask": False}],
                "labels": [False],
            },
            {
                "question_id": "q2",
                "question": "Q2",
                "candidates": [{"text": "B", "invalid_mask": False}],
                "labels": [True],
            },
        ]
        backend = FixtureBackend([
            '{"selected_index": null, "confidence": 0.2, "reason": "unsure"}',
            '{"selected_index": null, "confidence": 0.1, "reason": "unsure"}',
        ])
        result = run_selection(
            rows,
            backend,
            batch_size=2,
            oracle_confidence_threshold=0.9,
            oracle_budget_ratio=1.0,
            oracle_budget_count=1,
            show_progress=False,
        )
        self.assertEqual(result[0].status, "oracle")
        self.assertTrue(result[0].used_oracle)
        self.assertIsNone(result[0].selected_index)
        self.assertTrue(result[0].is_correct)
        self.assertEqual(result[1].status, "fallback")
        self.assertFalse(result[1].used_oracle)
        self.assertEqual(result[1].selected_index, 0)

        metrics = _compute_metrics(result)
        self.assertEqual(metrics["oracle_calls"], 1)
        self.assertEqual(metrics["n_rows_with_labels"], 2)
        self.assertIsNotNone(metrics["end_to_end_accuracy"])


if __name__ == "__main__":
    unittest.main()
