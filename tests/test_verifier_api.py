import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from qrank.llms import LLMBackend
from qrank.verifier import SQLiteCache, Verifier
from qrank.verifier.prompts import PERSPECTIVES
from qrank.verifier import api, cli
from verifier_fixtures import response_for


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
        return response_for(request)


class VerifierAPITests(unittest.TestCase):
    def test_batch_size_from_config_and_explicit_override(self):
        options = dict(model='fixture', device='cpu', dtype='auto',
                       temperature=0, max_tokens=128, revision='main')
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'config.json'
            path.write_text(json.dumps(dict(backend='fixture', batch_size=2, **options)))
            for override, expected in (({}, [1] + [2] * 12 + [1]), ({'batch_size': 3}, [1] + [3] * 8 + [1])):
                backend = RecordingBackend()
                with patch.object(api, 'load_backend', return_value=backend) as loader:
                    verifier = Verifier(config_path=path, **override)
                    verifier.verify_candidates('Question', [f'SELECT {i}' for i in range(5)])
                    self.assertEqual(backend.batches, expected)
                    loader.assert_called_once_with('fixture', **options)

    def test_batch_size_validation_precedes_model_loading(self):
        for value in (0, -1, True, 1.5, '8'):
            with self.subTest(value=value), patch.object(api, 'load_backend') as loader:
                with self.assertRaisesRegex(ValueError, 'batch_size'):
                    Verifier(config_path='unused.json', batch_size=value)
                loader.assert_not_called()
                with self.assertRaisesRegex(ValueError, 'batch_size'):
                    Verifier(RecordingBackend(), batch_size=value)

    def test_configured_backend_loads_once_and_cli_shares_settings(self):
        backend = RecordingBackend()
        options = dict(model='configured-model', device='cpu', dtype='float32',
                       temperature=0.4, max_tokens=128, revision='pinned')
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'config.json'
            path.write_text(json.dumps(dict(backend='configured-backend', **options)))
            with patch.object(api, 'load_backend', return_value=backend) as loader:
                verifier = Verifier(config_path=path)
                self.assertEqual(verifier.batch_size, 8)
                verifier.verify('Question', 'SELECT 1')
                verifier.verify('Question', 'SELECT 2')
                loader.assert_called_once_with('configured-backend', **options)
            args = cli.build_parser().parse_args(['transform-dataset', '--config', str(path)])
            with patch.object(cli, 'load_backend') as loader:
                cli.make_backend(args)
                loader.assert_called_once_with('configured-backend', **options)
            overridden = cli.build_parser().parse_args([
                'transform-dataset', '--config', str(path), '--model', 'override',
                '--temperature', '0',
            ])
            with patch.object(cli, 'load_backend') as loader:
                cli.make_backend(overridden)
                loader.assert_called_once_with(
                    'configured-backend', **{**options, 'model': 'override', 'temperature': 0},
                )

    def test_config_path_is_required_unless_backend_is_injected(self):
        with patch.object(api, 'load_backend') as loader:
            with self.assertRaisesRegex(ValueError, 'config_path is required'):
                Verifier()
            with self.assertRaisesRegex(ValueError, 'not both'):
                Verifier(RecordingBackend(), config_path='unused.json')
            loader.assert_not_called()

    def test_injected_backend_bypasses_configured_loading(self):
        backend = RecordingBackend()
        with patch.object(api, 'load_backend') as loader:
            verifier = Verifier(backend)
            self.assertIs(verifier.backend, backend)
            self.assertEqual(verifier.batch_size, 8)
            loader.assert_not_called()

    def test_single_and_batch_are_equivalent_and_candidates_are_isolated(self):
        backend = RecordingBackend()
        verifier = Verifier(backend, batch_size=1)
        kwargs = dict(context='CREATE TABLE items (id INTEGER)', evidence='Include every item')
        candidates = ['SELECT id FROM items', 'SELECT wrong FROM items']
        results = verifier.verify_candidates('List IDs', candidates, **kwargs)
        self.assertEqual([r.labels['output_semantics'] for r in results], ['S', 'C'])
        self.assertEqual(backend.batches, [1] * 11)
        self.assertNotIn(candidates[1], backend.requests[1]['user'])
        self.assertNotIn(candidates[0], backend.requests[6]['user'])
        self.assertIn(kwargs['context'], backend.requests[1]['user'])
        self.assertIn(kwargs['evidence'], backend.requests[1]['user'])
        single = verifier.verify('List IDs', candidates[0], **kwargs)
        self.assertEqual(single, results[0])
        self.assertEqual(len(backend.requests), 17)  # Extraction, four blocks and one coverage check.

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
        self.assertEqual(len(backend.requests), 12)

    def test_format_error_propagates_without_cache_or_retry(self):
        class Broken(RecordingBackend):
            def generate(self, **request):
                self.requests.append(request)
                return 'invalid JSON'

        backend = Broken()
        with self.assertRaises(json.JSONDecodeError):
            Verifier(backend).verify('Question', 'SELECT 1', perspective='output_semantics')
        self.assertEqual(len(backend.requests), 1)
        with self.assertRaises(TypeError):
            Verifier(backend, retries=1)

    def test_duplicate_candidates_share_checks(self):
        backend = RecordingBackend()
        results = Verifier(backend).verify_candidates('Question', ['SELECT 1', 'SELECT 1'])
        self.assertEqual(results[0], results[1])
        self.assertEqual(backend.batches, [1, 5])

    def test_one_perspective_and_cached_reuse_with_all_perspectives(self):
        backend = RecordingBackend()
        with tempfile.TemporaryDirectory() as tmp:
            verifier = Verifier(backend, cache=SQLiteCache(str(Path(tmp) / 'cache')))
            one = verifier.verify('List IDs', 'SELECT id FROM items', perspective='output_semantics')
            self.assertEqual(one.labels, {'output_semantics': 'S'})
            all_results = verifier.verify('List IDs', 'SELECT id FROM items')
            self.assertEqual(set(all_results.labels), set(PERSPECTIVES))
            self.assertNotIn('overall', all_results.labels)
            self.assertEqual(backend.batches, [1, 2, 3])
            for request in backend.requests[1:]:
                self.assertEqual(request['system'].count('PERSPECTIVE:'), 1)
                self.assertIn('RAW CANDIDATE SQL:', request['user'])
                self.assertIn('DETERMINISTIC AST SUMMARY:', request['user'])
            with self.assertRaises(ValueError):
                verifier.verify('List IDs', 'SELECT id FROM items', perspective='overall')

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
