import unittest

from datasets import Dataset

from qrank.judge.pipeline import build_clusters, judge_rows, parse_choice
from qrank.llms import LLMBackend


class FixtureBackend(LLMBackend):
    def __init__(self, responses):
        super().__init__("fixture")
        self.responses = list(responses)
        self.requests = []

    def generate_batch(self, requests):
        self.requests.extend(requests)
        return [self.responses.pop(0) for _ in requests]


class JudgeTests(unittest.TestCase):
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
        self.assertEqual(backend.requests[0]["response_choices"], ["A", "NONE"])
        self.assertIn("Candidate A:\nSELECT x FROM t", prompt)
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


if __name__ == "__main__":
    unittest.main()
