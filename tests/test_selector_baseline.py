import json
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from datasets import Dataset, DatasetDict

from qrank.baselines.selector.cli import main
from qrank.baselines.selector.config import load_selector_config
from qrank.baselines.selector.pipeline import _compute_metrics, run_selection, run_target
from qrank.llms import LLMBackend


class FixtureBackend(LLMBackend):
    def __init__(self, responses):
        super().__init__("fixture")
        self.responses = list(responses)
        self.requests = []

    def generate_batch(self, requests):
        self.requests.extend(requests)
        out = self.responses[: len(requests)]
        self.responses = self.responses[len(requests):]
        return out

    def generate(self, **request):
        raise AssertionError("generate() should not be called")


class SelectorBaselineTests(unittest.TestCase):
    def select(self, rows, responses, **kwargs):
        return run_selection(
            rows, FixtureBackend(responses), batch_size=2,
            show_progress=False,
            oracle_budget_ratio=kwargs.pop("oracle_budget_ratio", 0.0), **kwargs,
        )

    def write_config(self, root, rows):
        dataset_path = root / "dataset"
        DatasetDict({"test": Dataset.from_list(rows)}).save_to_disk(str(dataset_path))
        path = root / "config.json"
        path.write_text(json.dumps({
            "backend": "transformers", "model": "fixture", "device": "cpu",
            "dtype": "auto", "temperature": 0.0, "max_tokens": 128, "revision": "main",
            "output_dir": str(root / "output"),
            "dataset": {"name": "fixture", "path": str(dataset_path), "split": "test"},
            "show_progress": False, "oracle_budget_ratio": 0.0,
        }))
        return path

    def test_load_selector_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cfg.json"
            path.write_text(json.dumps({
                "backend": "transformers", "model": "fixture", "device": "cpu",
                "dtype": "auto", "temperature": 0.0, "max_tokens": 128, "revision": "main",
                "output_dir": "dataset/out",
                "dataset": {"name": "bird", "path": "dataset/processed/bird", "split": "test"},
                "oracle_budget_ratio": 0.25,
            }))
            cfg = load_selector_config(path)
            self.assertEqual(cfg.output_dir, "dataset/out")
            self.assertEqual(cfg.dataset.name, "bird")
            self.assertEqual(cfg.dataset.split, "test")
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

    def test_budget_selects_lowest_confidence_with_stable_ties_without_using_labels(self):
        responses = [json.dumps({"selected_index": 0, "confidence": confidence})
                     for confidence in (0.99, 0.8, 0.9, 0.8)]
        for labels in ([False, True, False, False], [True, False, True, True]):
            rows = [{"question_id": str(i), "candidates": [{"text": "A"}], "labels": [label]}
                    for i, label in enumerate(labels)]
            results = self.select(rows, responses, oracle_budget_ratio=0.5)
            self.assertEqual([r.question_id for r in results], ["0", "1", "2", "3"])
            self.assertEqual([r.used_oracle for r in results], [False, True, False, True])
            self.assertFalse(any(r.requested_oracle for r in results))
            self.assertEqual(results[0].selected_index, 0)
            self.assertEqual(results[2].selected_index, 0)

    def test_explicit_deferral_precedes_low_confidence_and_parse_failure(self):
        rows = [{"candidates": [{"text": "A"}]} for _ in range(3)]
        results = self.select(rows, [
            '{"selected_index": 0, "confidence": 0}',
            'not json',
            '{"selected_index": null, "confidence": 1}',
        ], oracle_budget_ratio=1.0, oracle_budget_count=1)
        self.assertEqual([r.used_oracle for r in results], [False, False, True])
        self.assertEqual([r.requested_oracle for r in results], [False, False, True])
        self.assertEqual(results[1].status, "fallback")

    def test_no_valid_candidates_prioritized_over_selected_answers(self):
        results = self.select([
            {"candidates": [{"text": "A"}]},
            {"candidates": [{"text": ""}]},
        ], ['{"selected_index": 0, "confidence": 0}'], oracle_budget_ratio=0.5)
        self.assertEqual([r.used_oracle for r in results], [False, True])
        self.assertTrue(results[1].requested_oracle)
        self.assertEqual(results[1].status, "oracle")

    def test_filtered_prompts_and_skipped_rows_preserve_indices_across_batches(self):
        candidates = [
            {"text": "INVALID SQL", "invalid_mask": True},
            {"text": "   "},
            {"text": "SELECT 2"},
            {"text": None},
            {"text": "SELECT 4"},
        ]
        rows = [{"question_id": str(i), "candidates": candidates if i % 2 else
                 [{"text": "ALL INVALID", "invalid_mask": True}]}
                for i in range(7)]
        backend = FixtureBackend(['{"selected_index": 4}', '{"selected_index": 2}',
                                  '{"selected_index": 4}'])
        with patch.object(backend, "generate_batch", wraps=backend.generate_batch) as generate:
            results = run_selection(rows, backend, batch_size=2, oracle_budget_ratio=0,
                                    show_progress=False)
        self.assertEqual(generate.call_count, 2)
        self.assertEqual([r["question_id"] for r in backend.requests], ["1", "3", "5"])
        self.assertEqual([r.question_id for r in results], list(map(str, range(7))))
        self.assertEqual([r.selected_index for r in results], [0, 4, 0, 2, 0, 4, 0])
        for request in backend.requests:
            self.assertIn("[2] (invalid=false)\nSELECT 2", request["user"])
            self.assertIn("[4] (invalid=false)\nSELECT 4", request["user"])
            for hidden in ("INVALID SQL", "[0]", "[1]", "[3]", "None"):
                self.assertNotIn(hidden, request["user"])
        for result in results[::2]:
            self.assertEqual(result.raw_response, "")
            self.assertTrue(result.requested_oracle)
            self.assertEqual(result.valid_indices, ())

    def test_failed_batch_maps_back_to_rows_after_skipping_invalid_candidates(self):
        rows = [{"candidates": [{"text": "bad", "invalid_mask": True}]},
                {"candidates": [{"text": "A"}]},
                {"candidates": []},
                {"candidates": [{"text": "B"}]}]
        backend = FixtureBackend([])
        with patch.object(backend, "generate_batch", side_effect=OSError("unavailable")):
            results = run_selection(rows, backend, batch_size=2, oracle_budget_ratio=0,
                                    show_progress=False)
        for i in (1, 3):
            self.assertEqual(results[i].status, "fallback")
            self.assertIn("backend error: OSError: unavailable", results[i].reason)
        for i in (0, 2):
            self.assertIn("model inference skipped", results[i].reason)
            self.assertNotIn("backend error", results[i].reason)

    def test_all_invalid_random_fallback_is_scored_and_stable_across_budgets(self):
        rows = [{"question_id": str(i),
                 "candidates": [{"text": "A", "invalid_mask": True},
                                {"text": "B", "invalid_mask": True}],
                 "labels": [True, False]} for i in range(20)]
        backend = FixtureBackend([])
        with patch.object(backend, "generate_batch") as generate:
            def select(ratio, cap=None):
                return run_selection(rows, backend, batch_size=2, oracle_budget_ratio=ratio,
                                     oracle_budget_count=cap, show_progress=False)
            baseline = select(0)
            repeated = select(0)
            partial = select(1, 3)
            full = select(1)
            generate.assert_not_called()
        self.assertEqual(baseline, repeated)
        self.assertEqual({r.selected_index for r in baseline}, {0, 1})
        self.assertEqual([r.used_oracle for r in partial], [True] * 3 + [False] * 17)
        self.assertEqual(partial[3:], baseline[3:])
        self.assertTrue(all(r.used_oracle and r.is_correct for r in full))
        for row, result in zip(rows, baseline, strict=True):
            self.assertEqual(result.status, "fallback")
            self.assertEqual(result.final_source, "fallback")
            self.assertTrue(result.requested_oracle)
            self.assertEqual(result.raw_response, "")
            self.assertEqual(result.is_correct, row["labels"][result.selected_index])
            self.assertIn("randomly selected", result.reason)

    def test_budget_fully_used_with_high_confidence_including_rounding_and_count_caps(self):
        rows = [{"candidates": [{"text": "A"}]} for _ in range(5)]
        responses = ['{"selected_index": 0, "confidence": 1}'] * 5
        for ratio, cap, expected in ((0.19, None, 0), (0.2, None, 1), (1, None, 5),
                                     (1, 2, 2), (1, 0, 0)):
            with self.subTest(ratio=ratio, cap=cap):
                results = self.select(rows, responses, oracle_budget_ratio=ratio,
                                      oracle_budget_count=cap)
                self.assertEqual([i for i, r in enumerate(results) if r.used_oracle],
                                 list(range(expected)))
        self.assertEqual(self.select([], [], oracle_budget_ratio=1.0), [])

    def test_end_to_end_counts_unanswered_rows_as_incorrect(self):
        rows = [
            {"candidates": [{"text": "A"}], "labels": [True]},
            {"candidates": [], "labels": []},
        ]
        results = self.select(rows, ['{"selected_index": 0}'])
        self.assertEqual(results[1].status, "no_valid_candidates")
        self.assertFalse(results[1].is_correct)
        metrics = _compute_metrics(results)
        self.assertEqual(metrics["end_to_end_accuracy"], 0.5)
        self.assertEqual(metrics["end_to_end_denominator"], 2)
        self.assertEqual(metrics["evaluation_status"], "complete")

    def test_oracle_successes_do_not_mask_missing_labels(self):
        rows = [{"candidates": [{"text": "A"}]} for _ in range(2)]
        results = self.select(rows, ['{"selected_index": null}', '{"selected_index": 0}'],
                              oracle_budget_ratio=0.5)
        metrics = _compute_metrics(results)
        self.assertEqual(metrics["n_rows_with_labels"], 0)
        self.assertEqual(metrics["oracle_calls"], 1)
        self.assertEqual(metrics["n_rows_missing_correctness"], 1)
        self.assertEqual(metrics["evaluation_coverage"], 0.5)
        self.assertEqual(metrics["evaluation_status"], "partial")
        self.assertIsNone(metrics["end_to_end_accuracy"])

    def test_oracle_status_matches_final_source_after_parse_failure(self):
        rows = [{"candidates": [{"text": "A"}], "labels": [False]}]
        results = run_selection(rows, FixtureBackend(["not json"]), batch_size=1,
                                oracle_budget_ratio=1.0, show_progress=False)
        self.assertEqual(results[0].status, "oracle")
        self.assertEqual(results[0].final_source, "oracle")
        self.assertTrue(results[0].used_oracle)

    def test_null_label_is_unknown_not_incorrect(self):
        results = self.select([{"candidates": [{"text": "A"}], "labels": [None]}],
                              ['{"selected_index": 0}'])
        self.assertIsNone(results[0].is_correct)
        self.assertIsNone(_compute_metrics(results)["end_to_end_accuracy"])

    def test_bad_labels_fail_before_inference(self):
        for labels in (["False"], [True, False], "True"):
            with self.subTest(labels=labels):
                backend = FixtureBackend([])
                with patch.object(backend, "generate_batch") as generate:
                    with self.assertRaisesRegex(ValueError, "labels must"):
                        run_selection([{"candidates": [{"text": "A"}], "labels": labels}],
                                      backend, batch_size=1, show_progress=False)
                    generate.assert_not_called()

    def test_malformed_responses_fall_back_without_aborting(self):
        for response in (None, '{"selected_index": true}', '{"selected_index": 0.5}',
                         '{"selected_index": 1, "confidence": NaN}'):
            with self.subTest(response=response):
                results = self.select([{"candidates": [{"text": "A"}, {"text": "B"}],
                                        "labels": [True, False]}], [response])
                self.assertEqual(results[0].status, "fallback")
                self.assertEqual(results[0].selected_index, 0)
                self.assertEqual(_compute_metrics(results)["end_to_end_accuracy"], 1.0)

    def test_null_candidate_is_not_selectable(self):
        results = self.select([{"candidates": [{"text": None}, {"text": "B"}]}],
                              ['{"selected_index": 0}'])
        self.assertEqual(results[0].selected_index, 1)

    def test_response_count_mismatch_falls_back_for_entire_batch(self):
        rows = [{"candidates": [{"text": "A"}, {"text": "B"}]}]
        for responses in ([], ['{"selected_index": 1}'] * 2):
            with self.subTest(responses=responses):
                backend = FixtureBackend([])
                with patch.object(backend, "generate_batch", return_value=responses):
                    results = run_selection(rows, backend, batch_size=1,
                                            oracle_budget_ratio=0.0, show_progress=False)
                self.assertEqual(results[0].status, "fallback")
                self.assertIn("Expected 1 responses", results[0].reason)

    def test_cli_saves_and_reports_end_to_end_results(self):
        rows = [
            {"question_id": "q1", "candidates": [{"text": "A"}], "labels": [True]},
            {"question_id": "q2", "candidates": [{"text": "B"}], "labels": [False]},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = self.write_config(root, rows)
            stdout, stderr = io.StringIO(), io.StringIO()
            with patch("qrank.baselines.selector.cli.load_backend", return_value=FixtureBackend(
                    ['{"selected_index": 0, "confidence": 1}'] * 2)), \
                    contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                self.assertEqual(main(["--config", str(path)]), 0)
            manifest = json.loads((root / "output/fixture/selector.manifest.json").read_text())
            selections = [json.loads(line) for line in
                          (root / "output/fixture/selections.jsonl").read_text().splitlines()]
            self.assertEqual(manifest, json.loads(stdout.getvalue())["run"])
            self.assertEqual(manifest["metrics"]["end_to_end_accuracy"], 0.5)
            self.assertEqual([r["is_correct"] for r in selections], [True, False])
            self.assertIn("raw_response", selections[0])
            self.assertIn("End-to-end accuracy: 50.00% over 2 rows", stderr.getvalue())

    def test_existing_output_fails_before_loading_model_or_inference(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = self.write_config(root, [{"candidates": [{"text": "A"}]}])
            output = root / "output/fixture"
            output.mkdir(parents=True)
            saved = output / "selections.jsonl"
            saved.write_text("existing results\n")
            with patch("qrank.baselines.selector.cli.load_backend") as load_backend, \
                    contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main(["--config", str(path)]), 2)
                load_backend.assert_not_called()
            config = load_selector_config(path)
            backend = FixtureBackend([])
            with patch.object(backend, "generate_batch") as generate:
                with self.assertRaisesRegex(ValueError, "not empty"):
                    run_target(config, config.dataset, backend)
                generate.assert_not_called()
            self.assertEqual(saved.read_text(), "existing results\n")

    def test_budget_sweep_reuses_generation_and_replays_original_deferrals(self):
        rows = [
            {"question_id": "q0", "candidates": [{"text": "A"}], "labels": [False]},
            {"question_id": "q1", "candidates": [{"text": "A"}, {"text": "B"}],
             "labels": [True, False]},
            {"question_id": "q2", "candidates": [{"text": "A"}], "labels": [True]},
            {"question_id": "q3", "candidates": [{"text": "A"}], "labels": [False]},
        ]
        responses = [
            '{"selected_index": null, "confidence": 0.1}',
            '{"selected_index": 1, "confidence": 0.1}',
            '{"selected_index": 0, "confidence": 0.9}',
            '{"selected_index": null, "confidence": 0.1}',
        ]
        for cap in (None, 1):
            with self.subTest(cap=cap), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                path = self.write_config(root, rows)
                data = json.loads(path.read_text())
                data.update(oracle_budget_ratios=[1.0, 0.0, 0.25, 0.5],
                            oracle_budget_ratio=0.5, oracle_budget_count=cap, batch_size=2)
                path.write_text(json.dumps(data))
                config = load_selector_config(path)
                backend = FixtureBackend(responses)
                with patch.object(backend, "generate_batch", wraps=backend.generate_batch) as generate:
                    manifest = run_target(config, config.dataset, backend)
                self.assertEqual(generate.call_count, 2)
                sweep = manifest["budget_sweep"]
                self.assertEqual([entry["metrics"]["end_to_end_accuracy"] for entry in sweep],
                                 [1.0, 0.25, 0.5, 0.75] if cap is None else [0.5, 0.25, 0.5, 0.5])
                self.assertEqual([entry["metrics"]["oracle_calls"] for entry in sweep],
                                 [4, 0, 1, 2] if cap is None else [1, 0, 1, 1])
                self.assertEqual([entry["effective_budget"] for entry in sweep],
                                 [4, 0, 1, 2] if cap is None else [1, 0, 1, 1])
                self.assertEqual(manifest["metrics"], sweep[-1]["metrics"])
                self.assertEqual(manifest["oracle_policy"]["allocation"],
                                 "deferrals_first_then_lowest_confidence")
                expected_indices = [[0, 1, 2, 3], [], [0], [0, 3]] if cap is None else [[0], [], [0], [0]]
                for entry, indices in zip(sweep, expected_indices, strict=True):
                    saved = [json.loads(line) for line in
                             (root / "output/fixture" / entry["selection_file"]).read_text().splitlines()]
                    self.assertEqual(len(saved), 4)
                    self.assertEqual([i for i, r in enumerate(saved) if r["used_oracle"]], indices)
                    self.assertEqual(sum(r["used_oracle"] for r in saved),
                                     entry["metrics"]["oracle_calls"])
                    self.assertEqual(sum(r["is_correct"] for r in saved) / 4,
                                     entry["metrics"]["end_to_end_accuracy"])

    def test_budget_sweep_config_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.write_config(Path(tmp), [{"candidates": [{"text": "A"}]}])
            data = json.loads(path.read_text())
            for sweep in (None, "0.5", [True], [-0.1], [1.1], [float("nan")], [None]):
                with self.subTest(sweep=sweep):
                    data["oracle_budget_ratios"] = sweep
                    path.write_text(json.dumps(data))
                    with self.assertRaisesRegex(ValueError, "oracle_budget_ratios"):
                        load_selector_config(path)

    def test_old_threshold_config_reports_migration(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.write_config(Path(tmp), [{"candidates": [{"text": "A"}]}])
            data = json.loads(path.read_text())
            data["oracle_confidence_threshold"] = 0.6
            path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "oracle_confidence_threshold has been removed"):
                load_selector_config(path)


if __name__ == "__main__":
    unittest.main()
