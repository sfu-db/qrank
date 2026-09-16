import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from datasets import Dataset, DatasetDict, load_from_disk

from qrank.llms import LLMBackend
from qrank.verifier import SQLiteCache, Verifier
from qrank.verifier import cli
from qrank.verifier.models import VerificationError
from qrank.verifier.models import ClusterInput, QuestionInput
from qrank.verifier.overall import parse_overall, verify_overall_questions


class OverallBackend(LLMBackend):
    def __init__(self):
        super().__init__("fixture")
        self.requests = []

    def generate_batch(self, requests):
        self.requests.extend(requests)
        responses = []
        for request in requests:
            responses.append(json.dumps({identifier: {
                "label": "C" if f"Candidate {identifier} — CANDIDATE SQL:\nSELECT wrong" in request["user"] else "S",
                "reason": "specific wrong output" if f"Candidate {identifier} — CANDIDATE SQL:\nSELECT wrong" in request["user"]
                else "matches the requested output",
            } for identifier in request["response_schema"]["required"]}))
        return responses


class OverallVerifierTests(unittest.TestCase):
    def test_api_one_request_per_question_dedup_and_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            backend = OverallBackend()
            verifier = Verifier(backend, cache=SQLiteCache(str(Path(tmp) / "cache.sqlite")))
            sql = ["SELECT id FROM items", "SELECT wrong FROM items", "SELECT id FROM items"]
            first = verifier.verify_overall_candidates("List IDs", sql, context="CREATE TABLE items(id INT)")
            self.assertEqual([result.label for result in first], ["S", "C", "S"])
            self.assertEqual(len(backend.requests), 1)
            self.assertEqual(first[0], first[2])
            self.assertIn(sql[1], backend.requests[0]["user"])
            self.assertIn(sql[0], backend.requests[0]["user"])
            self.assertEqual(backend.requests[0]["response_schema"]["required"], ["A", "B"])
            self.assertEqual(set(backend.requests[0]["response_schema"]["properties"]["A"]["properties"]["label"]["enum"]),
                             {"S", "U", "C", "NA"})
            second = verifier.verify_overall_candidates("List IDs", sql, context="CREATE TABLE items(id INT)")
            self.assertEqual([result.status for result in second], ["cached"] * 3)
            self.assertEqual(len(backend.requests), 1)
            verifier.verify_overall("List IDs", sql[0], context="different schema")
            self.assertEqual(len(backend.requests), 2)
            verifier.verify_overall("List IDs", sql[0], context="different schema",
                                    prompt_variant="v3")
            self.assertEqual(len(backend.requests), 3)
            self.assertIn("CANDIDATES:", backend.requests[-1]["user"])

    def test_response_contract_requires_each_candidate_and_allows_na(self):
        self.assertEqual(parse_overall(json.dumps({"A": {"label": "NA", "reason": "not assessable"}}), ["A"])["A"]["label"], "NA")
        for value in ({"A": {"label": "S", "reason": " "}},
                      {"A": {"label": "C", "reason": "wrong", "perspectives": []}},
                      {"B": {"label": "S", "reason": "wrong identifier"}}):
            with self.subTest(value=value), self.assertRaises(VerificationError):
                parse_overall(json.dumps(value), ["A"])
        with self.assertRaises(VerificationError):
            parse_overall(json.dumps({"A": {"label": "S", "reason": "okay"}}), ["A", "B"])

    def test_questions_batch_without_mixing_candidate_sets(self):
        backend = OverallBackend()
        questions = [
            QuestionInput("q1", "List IDs", "", "", "", "",
                          (ClusterInput("0", "SELECT id FROM items", 1, {}),
                           ClusterInput("1", "SELECT wrong FROM items", 1, {})), {}),
            QuestionInput("empty", "List IDs", "", "", "", "", (), {}),
            QuestionInput("q2", "Count rows", "", "", "", "",
                          (ClusterInput("0", "SELECT COUNT(*) FROM items", 1, {}),), {}),
        ]
        results = verify_overall_questions(questions, backend, batch_size=2)
        self.assertEqual([[result.label for result in group] for group in results],
                         [["S", "C"], [], ["S"]])
        self.assertEqual(len(backend.requests), 2)
        self.assertEqual([request["response_schema"]["required"] for request in backend.requests],
                         [["A", "B"], ["A"]])
        self.assertNotIn("SELECT wrong", backend.requests[1]["user"])

    def test_v3_prompt_is_opt_in_flat_and_has_separate_cache_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            backend = OverallBackend()
            cache = SQLiteCache(str(Path(tmp) / "cache.sqlite"))
            question = QuestionInput(
                "q1", "List IDs", "", "CREATE TABLE items(id INT)", "", "",
                (ClusterInput("0", "SELECT id FROM items", 1,
                              {"candidate_indices": [0], "agreement_cluster_id": "0"}),
                 ClusterInput("1", "SELECT wrong FROM items", 1,
                              {"candidate_indices": [1], "agreement_cluster_id": "1"})), {},
            )
            old = verify_overall_questions([question], backend, cache)[0]
            new = verify_overall_questions([question], backend, cache,
                                           prompt_variant="v3")[0]
            self.assertEqual([result.label for result in new], ["S", "C"])
            self.assertNotEqual(old[0].cache_key, new[0].cache_key)
            self.assertIn("EXECUTION-AGREEMENT GROUP", backend.requests[0]["user"])
            self.assertNotIn("EXECUTION-AGREEMENT GROUP", backend.requests[1]["user"])
            self.assertIn("CANDIDATES:", backend.requests[1]["user"])
            self.assertIn("required operations that are absent", backend.requests[1]["system"])
            again = verify_overall_questions([question], backend, cache,
                                             prompt_variant="v3")[0]
            self.assertEqual([result.status for result in again], ["cached", "cached"])
            self.assertEqual(len(backend.requests), 2)

    def test_v4_keeps_question_request_and_requires_grounded_c_reason(self):
        backend = OverallBackend()
        question = QuestionInput(
            "q1", "List IDs", "", "CREATE TABLE items(id INT)", "", "",
            (ClusterInput("0", "SELECT id FROM items", 1, {"candidate_indices": [0]}),
             ClusterInput("1", "SELECT wrong FROM items", 1, {"candidate_indices": [1]})), {},
        )
        results = verify_overall_questions([question], backend, prompt_variant="v4")[0]
        self.assertEqual([result.label for result in results], ["S", "C"])
        self.assertEqual(len(backend.requests), 1)
        self.assertIn("EXECUTION-AGREEMENT GROUP", backend.requests[0]["user"])
        self.assertIn("reread the displayed candidate", backend.requests[0]["system"])

    def test_v5_shared_requirements_in_one_question_response(self):
        backend = OverallBackend()
        question = QuestionInput(
            "q1", "List IDs of active items", "", "CREATE TABLE items(id INT, active INT)", "", "",
            (ClusterInput("0", "SELECT id FROM items WHERE active = 1", 1,
                          {"candidate_indices": [0]}),
             ClusterInput("1", "SELECT id FROM items", 1,
                          {"candidate_indices": [1]})), {},
        )
        response = json.dumps({
            "task_requirements": [
                {"source_quote": "IDs", "requirement": "return item IDs"},
                {"source_quote": "active items", "requirement": "keep active items only"},
            ],
            "A": {"label": "S", "reason": "active = 1 keeps active items"},
            "B": {"label": "C", "reason": "active items required; no active predicate"},
        })
        with patch.object(backend, "generate_batch", return_value=[response]) as generate:
            results = verify_overall_questions([question], backend, prompt_variant="v5")[0]
            self.assertEqual(generate.call_count, 1)
            request = generate.call_args.args[0][0]
        self.assertEqual([result.label for result in results], ["S", "C"])
        self.assertIn("task_requirements", results[0].raw_response)
        self.assertEqual(request["response_schema"]["required"],
                         ["task_requirements", "A", "B"])
        self.assertIn("EXECUTION-AGREEMENT GROUP", request["user"])

    def test_malformed_response_is_traced_not_cached_or_relabelled(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            backend = OverallBackend()
            cache = SQLiteCache(str(root / "cache.sqlite"))
            verifier = Verifier(backend, cache=cache, audit_path=root / "trace.jsonl")
            with patch.object(backend, "generate_batch", return_value=["not JSON"]):
                with self.assertRaises(json.JSONDecodeError):
                    verifier.verify_overall("List IDs", "SELECT id FROM items")
            events = [json.loads(line) for line in (root / "trace.jsonl").read_text().splitlines()]
            self.assertEqual(events[-1]["raw_text"], "not JSON")
            self.assertIsNone(cache._con().execute("SELECT cache_key FROM verifier_cache").fetchone())

    def test_dataset_alignment_and_source_blindness(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path, output = root / "input", root / "output"
            row = {
                "question_id": "q1", "question": "List IDs", "task_type": "text-to-sql",
                "group_id": "items_db",
                "context": "CREATE TABLE items(id INT)",
                "candidates": [
                    {"text": "BAD SQL", "invalid_mask": True, "source": "invalid-model"},
                    {"text": "SELECT id FROM items", "invalid_mask": False, "source": "model-a"},
                    {"text": "SELECT wrong FROM items", "invalid_mask": False, "source": "model-b"},
                    {"text": "SELECT items.id FROM items", "invalid_mask": False, "source": "model-c"},
                ],
                "agreement": [[1, 0, 0, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 1, 0, 1]],
                "labels": [False, True, False, True],
            }
            DatasetDict(test=Dataset.from_list([row])).save_to_disk(input_path)
            schema_path = root / "schemas.json"
            schema_path.write_text(json.dumps({"items_db": "CREATE TABLE items(id INT)"}))
            backend = OverallBackend()
            args = cli.build_parser().parse_args([
                "transform-overall-dataset", "--config", "training_configs/verifier_qwen3_1p7b.json",
                "--input", str(input_path), "--output", str(output), "--progress-every", "0",
                "--schemas", str(schema_path),
                "--models", "model-c", "model-b", "model-a", "invalid-model",
            ])
            with patch.object(cli, "make_backend", return_value=backend):
                self.assertEqual(args.func(args), 0)
            annotated = load_from_disk(output)["test"][0]
            self.assertEqual([item["source"] for item in annotated["candidates"]],
                             ["model-c", "model-b", "model-a"])
            self.assertEqual(annotated["agreement"], [[1, 0, 1], [0, 1, 0], [1, 0, 1]])
            self.assertEqual(annotated["labels"], [True, False, True])
            labels = [item["label"] for item in annotated["semantic_verifier_overall"]]
            self.assertEqual(labels, ["S", "C", "S"])
            self.assertEqual([item["identifier"] for item in annotated["semantic_verifier_overall"]],
                             ["A", "B", "C"])
            self.assertEqual(annotated["semantic_verifier_overall"][0]["cache_key"],
                             annotated["semantic_verifier_overall"][2]["cache_key"])
            self.assertEqual(len(backend.requests), 1)
            self.assertIn("EXECUTION-AGREEMENT GROUP 1:", backend.requests[0]["user"])
            self.assertIn("Candidate C — CANDIDATE SQL:", backend.requests[0]["user"])
            self.assertTrue(all("model-" not in request["user"] for request in backend.requests))
            self.assertTrue(all("BAD SQL" not in request["user"] for request in backend.requests))
            manifest = json.loads((output / "semantic_verifier_overall.manifest.json").read_text())
            self.assertEqual(manifest["label_counts"], {"S": 2, "U": 0, "C": 1, "NA": 0})
            self.assertEqual(manifest["n_candidates_checked"], 3)
            self.assertEqual(manifest["n_question_requests"], 1)
            self.assertEqual(manifest["n_model_generations"], 1)
            self.assertEqual(manifest["input_models"],
                             ["model-c", "model-b", "model-a", "invalid-model"])

    def test_all_invalid_question_skips_generation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, output = root / "input", root / "output"
            DatasetDict(test=Dataset.from_list([{
                "question_id": "q1", "question": "List IDs", "task_type": "prepbench",
                "candidates": [{"text": "BAD SQL", "invalid_mask": True, "source": "model-a"}],
                "agreement": [[1]],
            }])).save_to_disk(source)
            backend = OverallBackend()
            args = cli.build_parser().parse_args([
                "transform-overall-dataset", "--config", "training_configs/verifier_qwen3_1p7b.json",
                "--input", str(source), "--output", str(output), "--models", "model-a",
                "--progress-every", "0",
            ])
            with patch.object(cli, "make_backend", return_value=backend):
                self.assertEqual(args.func(args), 0)
            self.assertEqual(backend.requests, [])
            self.assertEqual(load_from_disk(output)["test"][0]["semantic_verifier_overall"], [])
            manifest = json.loads((output / "semantic_verifier_overall.manifest.json").read_text())
            self.assertEqual(manifest["n_question_requests"], 0)
            self.assertEqual(manifest["n_model_generations"], 0)


if __name__ == "__main__":
    unittest.main()
