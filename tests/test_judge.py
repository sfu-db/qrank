import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from datasets import Dataset, DatasetDict, load_from_disk

from qrank.judge import cli
from qrank.judge.cli import build_parser
from qrank.judge.pipeline import build_clusters, judge_rows, parse_choice, select_models
from qrank.llms import LLMBackend
from qrank.verifier.context import detect_task_type


class FixtureBackend(LLMBackend):
    def __init__(self, responses):
        super().__init__("fixture")
        self.responses = list(responses)
        self.requests = []

    def generate_batch(self, requests):
        self.requests.extend(requests)
        return [self.responses.pop(0) for _ in requests]


class JudgeTests(unittest.TestCase):
    def test_cli_requires_models_and_defaults_to_m7_input(self):
        args = build_parser().parse_args([
            "--config", "config.json", "--models", "model-a", "model-b",
            "--output", "output",
        ])
        self.assertEqual(args.models, ["model-a", "model-b"])
        self.assertEqual(args.input, "dataset/raw/bird_dev_agr_q_sql_af_AQXNRSO_ss_m7")

    def test_selects_input_models_and_slices_known_agreement(self):
        row = {
            "question_id": 4,
            "question": "Q",
            "candidates": [
                {"source": "unused", "text": "OLD", "invalid_mask": False, "features": [9.0]},
                {"source": "peer-a", "text": "SELECT 1", "invalid_mask": False, "features": [1.0]},
                {"source": "peer-b", "text": "SELECT one", "invalid_mask": False, "features": [2.0]},
                {"source": "bad", "text": "BAD", "invalid_mask": True, "features": [3.0]},
            ],
            "agreement": [
                [1, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0],
            ],
            "labels": [False, True, True, False],
        }
        rebuilt = select_models(row, ["peer-b", "bad", "peer-a"])
        self.assertEqual(
            [candidate["source"] for candidate in rebuilt["candidates"]],
            ["peer-b", "peer-a"],
        )
        self.assertEqual(rebuilt["agreement"], [[1, 1], [1, 1]])
        self.assertEqual(rebuilt["labels"], [True, True])
        backend = FixtureBackend(["A"])
        result = judge_rows([rebuilt], ["schema"], backend)[0]
        self.assertNotIn("OLD", backend.requests[0]["user"])
        self.assertNotIn("BAD", backend.requests[0]["user"])
        self.assertEqual(result.selected_candidate_indices, (0, 1))

    def test_clusters_only_valid_candidates_and_uses_one_representative(self):
        row = {
            "candidates": [
                {"text": "INVALID SECRET", "invalid_mask": True},
                {"text": "SELECT 1", "invalid_mask": False},
                {"text": "SELECT one", "invalid_mask": False},
                {"text": "SELECT 2", "invalid_mask": False},
            ],
            "agreement": [
                [1, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 1],
            ],
        }
        clusters = build_clusters(row)
        self.assertEqual([cluster["identifier"] for cluster in clusters], ["A", "B"])
        self.assertEqual(clusters[0]["representative_index"], 1)
        self.assertEqual(clusters[0]["candidate_indices"], [1, 2])
        self.assertEqual(clusters[0]["representative_sql"], "SELECT 1")

    def test_prompt_hides_invalid_candidates_sources_labels_and_nonrepresentatives(self):
        row = {
            "question_id": 7,
            "question": "Which value?",
            "evidence": "Use x.",
            "candidates": [
                {"text": "BAD SQL", "invalid_mask": True, "source": "SECRET_SOURCE"},
                {"text": "SELECT x FROM t", "source": "SECRET_SOURCE"},
                {"text": "SELECT t.x FROM t", "source": "SECRET_SOURCE"},
            ],
            "agreement": [[1, 0, 0], [0, 1, 1], [0, 1, 1]],
            "labels": [False, True, True],
        }
        backend = FixtureBackend(["A"])
        result = judge_rows([row], ["CREATE TABLE t (x INT);"], backend, batch_size=1)[0]
        prompt = backend.requests[0]["user"]
        system = backend.requests[0]["system"]
        self.assertEqual(backend.requests[0]["response_choices"], ["A", "NONE"])
        self.assertIn("primary task is to choose the best available candidate", system)
        self.assertIn("Return NONE only when every displayed candidate", system)
        self.assertIn("Candidate A:\nSELECT x FROM t", prompt)
        self.assertIn("Only one candidate is displayed. Return A unless", prompt)
        self.assertIn("Uncertainty by itself is not an error", prompt)
        self.assertNotIn("BAD SQL", prompt)
        self.assertNotIn("SELECT t.x", prompt)
        self.assertNotIn("SECRET_SOURCE", prompt)
        self.assertNotIn("True", prompt)
        self.assertEqual(result.selected_candidate_index, 1)
        self.assertEqual(result.selected_candidate_indices, (1, 2))
        self.assertTrue(result.is_correct)

    def test_none_and_all_invalid_rows(self):
        valid = {
            "question": "Q", "candidates": [{"text": "SELECT 1"}],
            "agreement": [[1]], "labels": [False],
        }
        invalid = {
            "question": "Q2", "candidates": [{"text": "bad", "invalid_mask": True}],
            "agreement": [[0]], "labels": [False],
        }
        backend = FixtureBackend(["NONE"])
        results = judge_rows([valid, invalid], ["schema", "schema"], backend)
        self.assertEqual(len(backend.requests), 1)
        self.assertEqual(results[0].status, "none")
        self.assertTrue(results[0].is_correct)
        self.assertEqual(results[1].status, "no_valid_candidates")
        self.assertTrue(results[1].is_correct)

    def test_response_contract_is_strict(self):
        self.assertEqual(parse_choice(" a\n", ["A", "B"]), "A")
        self.assertEqual(parse_choice("Candidate B", ["A", "B"]), "B")
        self.assertEqual(parse_choice("candidate   a", ["A", "B"]), "A")
        self.assertIsNone(parse_choice("NONE", ["A"]))
        for raw in ("Candidate A because it is correct", "A.", "```A```", "C"):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                parse_choice(raw, ["A", "B"])

    def test_result_records_can_be_added_as_one_dataset_column(self):
        row = {
            "question_id": 1, "question": "Q",
            "candidates": [{"text": "SELECT 1", "invalid_mask": False}],
            "agreement": [[1]], "labels": [True],
        }
        result = judge_rows([row], ["schema"], FixtureBackend(["A"]))[0]
        from dataclasses import asdict
        source = Dataset.from_list([row])
        annotated = source.add_column("qwen_judge", [asdict(result)])
        self.assertEqual(annotated.column_names, source.column_names + ["qwen_judge"])
        self.assertEqual(annotated[0]["qwen_judge"]["selected_identifier"], "A")
        self.assertEqual(annotated[0]["candidates"], source[0]["candidates"])

    def test_prepbench_matrix_detection_and_in_memory_enrichment(self):
        matrix_row = {
            "case_id": "case_001",
            "setting": "Prep-Code Generation",
            "models": ["model-a", "model-b"],
            "agreement": [[1, 0], [0, 1]],
            "labels": [1, 0],
            "valid_models": ["model-a", "model-b"],
            "invalid_models": [],
            "answer_features": [[0], [0]],
            "feature_names": ["invalid_models"],
            "question_id": 1,
            "question": "case_001",
        }
        self.assertEqual(detect_task_type(matrix_row), "prepbench")
        export_row = {
            "question_id": "case_001",
            "question": "Clean the input and write output_01.csv.",
            "candidates": [
                {"source": "model-a", "text": "def solve(tables):\n return {'output_01.csv': tables['input.csv']}"},
                {"source": "model-b", "text": "def solve(tables):\n return {}"},
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "matrix"
            output = root / "judged"
            export = root / "prepbench.jsonl"
            DatasetDict({"test": Dataset.from_list([matrix_row])}).save_to_disk(str(source))
            export.write_text(json.dumps(export_row) + "\n", encoding="utf-8")
            args = build_parser().parse_args([
                "--config", "training_configs/judge_qwen2p5_coder_7b_vllm.json",
                "--models", "model-a", "model-b",
                "--input", str(source), "--output", str(output),
                "--prepbench-jsonl", str(export),
            ])
            backend = FixtureBackend(["A"])
            with patch.object(cli, "load_database_schemas",
                              side_effect=AssertionError("database schema discovery")), \
                 patch.object(cli, "load_database_descriptions",
                              side_effect=AssertionError("database description discovery")), \
                 patch.object(cli, "load_backend", return_value=backend):
                manifest = cli.run(args)

            self.assertEqual(manifest["task_types"], ["prepbench"])
            prompt = backend.requests[0]
            self.assertIn("Python/pandas programs", prompt["system"])
            self.assertIn("Data-preparation task", prompt["user"])
            self.assertIn("Input table schemas, samples, and runtime dtypes were not provided", prompt["user"])
            self.assertNotIn("Database context/schema", prompt["user"])
            result = load_from_disk(str(output))["test"][0]
            self.assertEqual(result["task_type"], "prepbench")
            self.assertEqual([candidate["source"] for candidate in result["candidates"]],
                             ["model-a", "model-b"])
            self.assertEqual(result["qwen_judge"]["selected_identifier"], "A")

    def test_bare_prepbench_matrix_explains_required_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "matrix"
            row = {
                "case_id": "case_001", "setting": "Prep-Code Generation",
                "models": ["model-a"], "agreement": [[1]], "labels": [1],
                "answer_features": [[0]], "feature_names": ["invalid_models"],
                "question_id": 1, "question": "case_001",
            }
            DatasetDict({"test": Dataset.from_list([row])}).save_to_disk(str(source))
            args = build_parser().parse_args([
                "--config", "training_configs/judge_qwen2p5_coder_7b_vllm.json",
                "--models", "model-a", "--input", str(source),
                "--output", str(root / "judged"),
            ])
            with self.assertRaisesRegex(ValueError, "--prepbench-jsonl"):
                cli.run(args)


if __name__ == "__main__":
    unittest.main()
