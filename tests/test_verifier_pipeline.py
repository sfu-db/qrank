from contextlib import chdir
import io
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from qrank.llms import LLMBackend
from qrank.verifier import pipeline as sv
from qrank.verifier import cli, context
from datasets import Dataset, DatasetDict, load_from_disk


class FakeBackend(LLMBackend):
    backend_name = 'test'

    def __init__(self):
        super().__init__('test')
        self.calls = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        return json.dumps({p: {'label': 'S', 'reason': 'fixture'} for p in sv.PERSPECTIVES})


class DatasetTransformTests(unittest.TestCase):
    def test_batches_across_questions_preserve_order_cache_and_retry_only_failures(self):
        class Batched(FakeBackend):
            def __init__(self):
                super().__init__()
                self.batches = []

            def generate_batch(self, requests):
                self.batches.append([r['question_id'] for r in requests])
                return ['bad JSON' if r['question_id'] == '2' and len(self.batches) == 1
                        else self.generate(**r) for r in requests]

        with tempfile.TemporaryDirectory() as tmp:
            c = sv.ClusterInput('0', 'SELECT 1', 1, {})
            questions = [sv.QuestionInput(str(i), f'question {i}', '', '', '', '', (c,), {})
                         for i in range(4)]
            backend = Batched()
            cache = sv.SQLiteCache(str(Path(tmp) / 'cache'))
            sv.verify_one(questions[0], c, backend, cache, 0, 0)
            backend.batches.clear()
            empty = sv.QuestionInput('empty', 'empty', '', '', '', '', (), {})
            rows = sv.collect_questions([questions[0], empty, *questions[1:]], backend, cache, 1, 0, 8)
            self.assertEqual(backend.batches, [['1', '2', '3'], ['2']])
            self.assertEqual([r['question_id'] for r in rows], ['0', 'empty', '1', '2', '3'])
            self.assertEqual(rows[0]['clusters'][0]['verification']['status'], 'cached')
            self.assertEqual(rows[1]['clusters'], [])
            self.assertEqual(rows[3]['clusters'][0]['verification']['attempts'], 2)
            self.assertTrue(all(r['collection_status'] == 'ok' for r in rows if r['clusters']))

    def test_gpu_runtime_error_is_not_silently_saved_as_na(self):
        with tempfile.TemporaryDirectory() as tmp:
            q = sv.QuestionInput('1', 'question', '', '', '', '', (), {})
            c = sv.ClusterInput('0', 'SELECT 1', 1, {})
            backend = FakeBackend()
            with patch.object(backend, 'generate_batch', side_effect=RuntimeError('CUDA out of memory')) as generate:
                with self.assertRaisesRegex(RuntimeError, 'CUDA out of memory'):
                    sv.verify_one(q, c, backend, sv.SQLiteCache(str(Path(tmp) / 'cache')), 2, 0)
                generate.assert_called_once()

    def test_response_json_with_wrappers_and_escaped_braces(self):
        payload = {p: {'label': 'S', 'reason': 'nested {value}, quote " and slash \\'}
                   for p in sv.PERSPECTIVES}
        raw = json.dumps(payload)
        for text in (raw, f'```json\n{raw}\n```', f'Result:\n{raw}\nDone.'):
            with self.subTest(text=text):
                labels, reasons = sv.parse_verifier_response(text)
                self.assertEqual(labels, {p: 'S' for p in sv.PERSPECTIVES})
                self.assertEqual(reasons, {p: payload[p]['reason'] for p in sv.PERSPECTIVES})
        for text in ('not JSON', '{"overall":', '{"overall": S}', '{}'):
            with self.subTest(text=text), self.assertRaises(sv.VerificationError):
                sv.parse_verifier_response(text)

    def test_retry_recovers_from_transport_and_format_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            q = sv.QuestionInput('1', 'question', '', '', '', '', (), {})
            c = sv.ClusterInput('0', 'SELECT 1', 1, {})
            backend = FakeBackend()
            cache = sv.SQLiteCache(str(Path(tmp) / 'cache'))
            raw = backend.generate()
            with patch.object(backend, 'generate', side_effect=[TimeoutError(), 'bad JSON', raw]) as generate:
                result = sv.verify_one(q, c, backend, cache, 2, 0)
                self.assertEqual((result.status, result.attempts), ('ok', 3))
                cached = sv.verify_one(q, c, backend, cache, 2, 0)
                self.assertEqual(cached.status, 'cached')
                self.assertEqual(cached.labels, result.labels)
                self.assertEqual(cached.reasons, result.reasons)
                self.assertEqual(generate.call_count, 3)

    def test_cache_and_programming_errors_propagate(self):
        with tempfile.TemporaryDirectory() as tmp:
            q = sv.QuestionInput('1', 'question', '', '', '', '', (), {})
            c = sv.ClusterInput('0', 'SELECT 1', 1, {})
            backend = FakeBackend()
            cache = sv.SQLiteCache(str(Path(tmp) / 'cache'))
            with patch.object(cache, 'put', side_effect=sqlite3.OperationalError('disk full')):
                with self.assertRaises(sqlite3.OperationalError):
                    sv.verify_one(q, c, backend, cache, 2, 0)
            self.assertEqual(len(backend.calls), 1)
            with patch.object(backend, 'generate', side_effect=TypeError('bug')) as generate:
                with self.assertRaises(TypeError):
                    sv.verify_one(q, c, backend, cache, 2, 0)
                generate.assert_called_once()

    def test_schema_discovery_prefers_ddl_outside_project_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / 'dataset/base/bird/dev'
            base.mkdir(parents=True)
            ddl = ['CREATE TABLE items (id INTEGER PRIMARY KEY)',
                   'CREATE TABLE related (item_id INTEGER REFERENCES items(id))']
            (base / 'schema.json').write_text(json.dumps({'example': ddl}))
            (base / 'dev_tables.json').write_text(json.dumps([{'db_id': 'example'}]))
            args = cli.build_parser().parse_args(['transform-dataset'])
            with patch.object(context, '__file__', str(root / 'src/qrank/verifier/context.py')), chdir(base):
                self.assertEqual(sv.load_database_schemas(args), {'example': '\n\n'.join(ddl)})
                (base / 'schema.json').unlink()
                self.assertEqual(sv.load_database_schemas(args),
                                 {'example': sv.stable_json({'db_id': 'example'})})

    def test_explicit_schema_mapping_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            schema = Path(tmp) / 'schema.json'
            args = cli.build_parser().parse_args(['transform-dataset', '--schemas', str(schema)])
            schema.write_text(json.dumps({'example': 'CREATE TABLE items (id INTEGER)'}))
            self.assertEqual(sv.load_database_schemas(args),
                             {'example': 'CREATE TABLE items (id INTEGER)'})
            schema.write_text(json.dumps({'example': [42]}))
            with self.assertRaisesRegex(sv.VerificationError, 'Invalid schema definitions'):
                sv.load_database_schemas(args)

    def test_missing_dataset_reports_actionable_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = str(Path(tmp) / 'missing')
            stderr = io.StringIO()
            with patch('sys.stderr', stderr), patch.object(cli, 'make_backend') as backend:
                status = cli.main([
                    'transform-dataset', '--input', missing,
                    '--output', str(Path(tmp) / 'output'),
                ])
            self.assertEqual(status, 2)
            self.assertIn(missing, stderr.getvalue())
            self.assertIn(f'Working directory: {Path.cwd()}', stderr.getvalue())
            self.assertIn('absolute path', stderr.getvalue())
            backend.assert_not_called()
            self.assertFalse((Path(tmp) / 'output').exists())

    def test_alignment_invalid_filter_cache_and_preservation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dbs = root / 'dbs'
            dbs.mkdir()
            with sqlite3.connect(dbs / 'example.sqlite') as con:
                con.execute('CREATE TABLE items (id INTEGER PRIMARY KEY)')
            row = {
                'question_id': 1, 'question': 'List item IDs', 'evidence': '',
                'context': '', 'group_id': 'example', 'metadata': {'db_id': 'example'},
                'candidates': [
                    {'text': 'BAD SQL', 'invalid_mask': True, 'source': 'SECRET_SOURCE'},
                    {'text': 'SELECT id FROM items', 'invalid_mask': False, 'source': 'SECRET_SOURCE'},
                    {'text': 'SELECT items.id FROM items', 'invalid_mask': False, 'source': 'SECRET_SOURCE'},
                ],
                'agreement': [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
                'labels': [False, True, True],
            }
            invalid = {**row, 'question_id': 2, 'candidates': [row['candidates'][0]],
                       'agreement': [[1]], 'labels': [False]}
            DatasetDict(test=Dataset.from_list([row, invalid])).save_to_disk(root / 'input')
            backend = FakeBackend()
            for run in range(2):
                args = cli.build_parser().parse_args([
                    'transform-dataset', '--input', str(root / 'input'),
                    '--output', str(root / f'output{run}'), '--db-root', str(dbs),
                    '--cache', str(root / 'cache.sqlite'),
                ])
                with patch.object(cli, 'make_backend', return_value=backend):
                    self.assertEqual(args.func(args), 0)
                output = load_from_disk(args.output)['test']
                for key, value in row.items():
                    self.assertEqual(output[0][key], value)
                annotations = output[0]['semantic_verifier']
                self.assertEqual(len(annotations), 3)
                self.assertEqual(annotations[0]['status'], 'invalid')
                self.assertEqual(annotations[1], annotations[2])
                self.assertEqual(output[1]['semantic_verifier'][0]['status'], 'invalid')
                manifest = json.loads((Path(args.output) / 'semantic_verifier.manifest.json').read_text())
                self.assertEqual(manifest['n_clusters'], 1)
                self.assertEqual(manifest['n_all_invalid_or_no_valid_cluster'], 1)
                self.assertFalse(list(Path(args.output).glob('*.jsonl*')))
                _, _, questions = sv.dataset_questions(args)
                self.assertEqual(questions[0].clusters[0].vote_count, 2)
                self.assertEqual(questions[1].clusters, ())
            self.assertEqual(len(backend.calls), 1)
            prompt = backend.calls[0]['user']
            self.assertIn('CREATE TABLE items', prompt)
            self.assertNotIn('SECRET_SOURCE', prompt)
            self.assertNotIn('BAD SQL', prompt)
            self.assertNotIn('SELECT items.id', prompt)
            self.assertNotIn('vote_count', prompt)

    def test_failure_is_missing_evidence(self):
        class Broken(FakeBackend):
            def generate(self, **kwargs):
                return 'not JSON'
        with tempfile.TemporaryDirectory() as tmp:
            q = sv.QuestionInput('1', 'question', '', '', '', '', (), {})
            c = sv.ClusterInput('0', 'SELECT 1', 1, {})
            result = sv.verify_one(q, c, Broken(), sv.SQLiteCache(str(Path(tmp) / 'cache')), 0, 0)
            self.assertEqual(result.status, 'failed')
            self.assertEqual(set(result.labels.values()), {'NA'})


if __name__ == '__main__':
    unittest.main()
