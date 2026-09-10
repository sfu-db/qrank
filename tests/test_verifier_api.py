import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from qrank import config
from qrank.llms import LLMBackend
from qrank.verifier import SQLiteCache, Verifier
from qrank.verifier.prompts import PERSPECTIVES
from qrank.verifier import api, cli


class RecordingBackend(LLMBackend):
    def __init__(self):
        super().__init__('fixture')
        self.requests = []
        self.batches = []

    def generate_batch(self, requests):
        self.batches.append(len(requests))
        return super().generate_batch(requests)

    def generate(self, **request):
        self.requests.append(request)
        label = 'C' if 'SELECT wrong' in request['user'] else 'S'
        return json.dumps({p: {'label': label, 'reason': 'fixture'} for p in PERSPECTIVES})


class VerifierAPITests(unittest.TestCase):
    def test_configured_backend_loads_once_and_cli_shares_defaults(self):
        backend = RecordingBackend()
        options = dict(model='configured-model', device='cpu', dtype='float32',
                       temperature=0, max_tokens=128, revision='pinned')
        with patch.object(config, 'LLM_BACKEND', 'configured-backend'), \
             patch.object(config, 'LLM_OPTIONS', options), \
             patch.object(api, 'load_backend', return_value=backend) as loader:
            verifier = Verifier()
            verifier.verify('Question', 'SELECT 1')
            verifier.verify('Question', 'SELECT 2')
            loader.assert_called_once_with('configured-backend', **options)
            args = cli.build_parser().parse_args(['transform-dataset'])
            self.assertEqual(args.backend, 'configured-backend')
            for name, value in options.items():
                self.assertEqual(getattr(args, name), value)
            overridden = cli.build_parser().parse_args(['transform-dataset', '--model', 'override'])
            self.assertEqual(overridden.model, 'override')

    def test_injected_backend_bypasses_configured_loading(self):
        backend = RecordingBackend()
        with patch.object(api, 'load_backend') as loader:
            self.assertIs(Verifier(backend).backend, backend)
            loader.assert_not_called()

    def test_single_and_batch_are_equivalent_and_candidates_are_isolated(self):
        backend = RecordingBackend()
        verifier = Verifier(backend, batch_size=1)
        kwargs = dict(context='CREATE TABLE items (id INTEGER)', evidence='Include every item')
        candidates = ['SELECT id FROM items', 'SELECT wrong FROM items']
        results = verifier.verify_candidates('List IDs', candidates, **kwargs)
        self.assertEqual([r.labels['overall'] for r in results], ['S', 'C'])
        self.assertEqual(backend.batches, [1, 1])
        self.assertNotIn(candidates[1], backend.requests[0]['user'])
        self.assertNotIn(candidates[0], backend.requests[1]['user'])
        self.assertIn(kwargs['context'], backend.requests[0]['user'])
        self.assertIn(kwargs['evidence'], backend.requests[0]['user'])
        single = verifier.verify('List IDs', candidates[0], **kwargs)
        self.assertEqual(single, results[0])
        self.assertEqual(len(backend.requests), 3)  # Caching is opt-in.

    def test_prepbench_detection_and_explicit_task(self):
        backend = RecordingBackend()
        verifier = Verifier(backend)
        for kwargs in ({}, {'task_type': 'prepbench'}):
            result = verifier.verify('Return input tables', 'def solve(tables): return tables', **kwargs)
            self.assertEqual(result.status, 'ok')
            self.assertIn('CANDIDATE PYTHON/PANDAS CODE:', backend.requests[-1]['user'])
            self.assertIn('runtime dtypes were not provided', backend.requests[-1]['user'])

    def test_optional_cache_reuses_results_and_context_changes_invalidate(self):
        backend = RecordingBackend()
        with tempfile.TemporaryDirectory() as tmp:
            verifier = Verifier(backend, cache=SQLiteCache(str(Path(tmp) / 'cache.sqlite')))
            first = verifier.verify('List IDs', 'SELECT id FROM items', context='schema A')
            cached = verifier.verify('List IDs', 'SELECT id FROM items', context='schema A')
            changed = verifier.verify('List IDs', 'SELECT id FROM items', context='schema B')
        self.assertEqual((first.status, cached.status, changed.status), ('ok', 'cached', 'ok'))
        self.assertEqual(first.labels, cached.labels)
        self.assertEqual(len(backend.requests), 2)

    def test_retries_and_failed_result_without_cache(self):
        class Broken(RecordingBackend):
            def generate(self, **request):
                self.requests.append(request)
                return 'invalid JSON'

        backend = Broken()
        result = Verifier(backend, retries=1, retry_delay=0).verify('Question', 'SELECT 1')
        self.assertEqual((result.status, result.attempts), ('failed', 2))
        self.assertEqual(set(result.labels.values()), {'NA'})
        self.assertEqual(len(backend.requests), 2)

    def test_empty_candidates_do_not_generate(self):
        backend = RecordingBackend()
        verifier = Verifier(backend)
        self.assertEqual(verifier.verify_candidates('Question', []), [])
        self.assertEqual(backend.requests, [])

    def test_import_does_not_load_dataset_or_model_libraries(self):
        subprocess.run([sys.executable, '-c',
                        'import sys; from qrank.verifier import Verifier; '
                        'assert not {"datasets", "torch", "transformers"}.intersection(sys.modules)'],
                       check=True, capture_output=True, text=True)
