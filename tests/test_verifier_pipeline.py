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


from verifier_fixtures import FakeBackend, response_for


class DatasetTransformTests(unittest.TestCase):
    def test_batches_across_questions_preserve_order_and_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            c = sv.ClusterInput('0', 'SELECT 1', 1, {})
            questions = [sv.QuestionInput(str(i), f'question {i}', '', '', '', '', (c,), {})
                         for i in range(4)]
            backend = FakeBackend()
            cache = sv.SQLiteCache(str(Path(tmp) / 'cache'))
            sv.verify_one(questions[0], c, backend, cache)
            backend.batches.clear()
            empty = sv.QuestionInput('empty', 'empty', '', '', '', '', (), {})
            rows = sv.collect_questions([questions[0], empty, *questions[1:]], backend, cache, 8)
            self.assertEqual([[r['question_id'] for r in b] for b in backend.batches],
                             [['1', '2', '3'], ['1'] * 5 + ['2'] * 3, ['2'] * 2 + ['3'] * 5])
            self.assertEqual([r['question_id'] for r in rows], ['0', 'empty', '1', '2', '3'])
            self.assertEqual(rows[0]['clusters'][0]['verification']['status'], 'cached')
            self.assertEqual(rows[1]['clusters'], [])
            self.assertEqual(rows[3]['clusters'][0]['verification']['attempts'], 1)

    def test_all_generation_errors_propagate_without_retry(self):
        from qrank.llms import BackendError
        q = sv.QuestionInput('1', 'question', '', '', '', '', (), {})
        c = sv.ClusterInput('0', 'SELECT 1', 1, {})
        for error in (TimeoutError('timeout'), BackendError('engine'),
                      RuntimeError('CUDA out of memory'), TypeError('bug'), ValueError('bad')):
            backend = FakeBackend()
            with self.subTest(error=error), patch.object(backend, 'generate_batch', side_effect=error) as generate:
                with self.assertRaises(type(error)) as raised:
                    sv.verify_one(q, c, backend)
                self.assertIs(raised.exception, error)
                generate.assert_called_once()

    def test_malformed_candidate_stops_and_preserves_raw_response(self):
        q = sv.QuestionInput('1', 'question', '', '', '', '', (), {})
        c = sv.ClusterInput('0', 'SELECT 1', 1, {})
        backend = FakeBackend()
        with tempfile.TemporaryDirectory() as tmp:
            cache = sv.SQLiteCache(str(Path(tmp) / 'cache'))
            trace = Path(tmp) / 'trace.jsonl'
            from qrank.verifier.prompts import requirements_request
            good = response_for(requirements_request(q))
            with patch.object(backend, 'generate_batch', side_effect=[[good], ['bad JSON']]) as generate:
                with self.assertRaises(json.JSONDecodeError):
                    sv.verify_one(q, c, backend, cache, audit_path=trace)
                self.assertEqual(generate.call_count, 2)
            events = [json.loads(line) for line in trace.read_text().splitlines()]
            self.assertTrue(any(e.get('raw_text') == 'bad JSON' for e in events))
            self.assertEqual(cache._con().execute('SELECT COUNT(*) FROM verifier_cache').fetchone()[0], 1)
            backend.batches.clear()
            result = sv.verify_one(q, c, backend, cache)
            self.assertEqual(result.status, 'ok')
            self.assertEqual(len(backend.batches), 5)

    def test_cache_errors_propagate(self):
        with tempfile.TemporaryDirectory() as tmp:
            q = sv.QuestionInput('1', 'question', '', '', '', '', (), {})
            c = sv.ClusterInput('0', 'SELECT 1', 1, {})
            backend = FakeBackend()
            cache = sv.SQLiteCache(str(Path(tmp) / 'cache'))
            with patch.object(cache, 'put', side_effect=sqlite3.OperationalError('disk full')):
                with self.assertRaises(sqlite3.OperationalError):
                    sv.verify_one(q, c, backend, cache)
            self.assertEqual(len(backend.calls), 1)

    def test_schema_discovery_prefers_ddl_outside_project_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / 'dataset/base/bird/dev'
            base.mkdir(parents=True)
            ddl = ['CREATE TABLE items (id INTEGER PRIMARY KEY)',
                   'CREATE TABLE related (item_id INTEGER REFERENCES items(id))']
            (base / 'schema.json').write_text(json.dumps({'example': ddl}))
            (base / 'dev_tables.json').write_text(json.dumps([{'db_id': 'example'}]))
            args = cli.build_parser().parse_args([
                'transform-dataset', '--config', 'training_configs/verifier_qwen3_1p7b.json',
            ])
            with patch.object(context, '__file__', str(root / 'src/qrank/verifier/context.py')), chdir(base):
                self.assertEqual(sv.load_database_schemas(args), {'example': '\n\n'.join(ddl)})
                (base / 'schema.json').unlink()
                self.assertEqual(sv.load_database_schemas(args),
                                 {'example': sv.stable_json({'db_id': 'example'})})

    def test_explicit_schema_mapping_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            schema = Path(tmp) / 'schema.json'
            args = cli.build_parser().parse_args(['transform-dataset', '--config', 'training_configs/verifier_qwen3_1p7b.json', '--schemas', str(schema)])
            schema.write_text(json.dumps({'example': 'CREATE TABLE items (id INTEGER)'}))
            self.assertEqual(sv.load_database_schemas(args),
                             {'example': 'CREATE TABLE items (id INTEGER)'})
            schema.write_text(json.dumps({'example': [42]}))
            with self.assertRaisesRegex(sv.VerificationError, 'Invalid schema definitions'):
                sv.load_database_schemas(args)

    def test_missing_dataset_propagates_before_model_loading(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = str(Path(tmp) / 'missing')
            with patch.object(cli, 'make_backend') as backend:
                with self.assertRaises(FileNotFoundError) as error:
                    cli.main([
                        'transform-dataset', '--config', 'training_configs/verifier_qwen3_1p7b.json',
                        '--input', missing, '--output', str(Path(tmp) / 'output'),
                    ])
            self.assertIn(missing, str(error.exception))
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
            config_path = root / 'verifier.json'
            config = json.loads(Path('training_configs/verifier_qwen3_1p7b.json').read_text())
            config_path.write_text(json.dumps({**config, 'batch_size': 3}))
            for run in range(2):
                args = cli.build_parser().parse_args([
                    'transform-dataset', '--config', str(config_path), '--input', str(root / 'input'),
                    '--output', str(root / f'output{run}'), '--db-root', str(dbs),
                    '--cache', str(root / 'cache.sqlite'),
                    *(['--batch-size', '2'] if run else []),
                ])
                with patch.object(cli, 'make_backend', return_value=backend), \
                        patch.object(cli, 'collect_questions', wraps=cli.collect_questions) as collect:
                    self.assertEqual(args.func(args), 0)
                    self.assertEqual(collect.call_args.args[-1], 2 if run else 3)
                output = load_from_disk(args.output)['test']
                for key, value in row.items():
                    self.assertEqual(output[0][key], value)
                annotations = output[0]['semantic_verifier']
                self.assertEqual(len(annotations), 3)
                self.assertEqual(annotations[0]['status'], 'invalid')
                self.assertEqual(annotations[0]['block_results'], [])
                self.assertEqual(len(annotations[1]['block_results']), 4)
                self.assertEqual(len(annotations[1]['requirement_results']), 1)
                self.assertEqual(annotations[0]['requirement_results'], [])
                contract = json.loads(annotations[1]['requirements'])
                self.assertEqual(contract['requirements'][0]['source_refs'][0]['quote'], row['question'])
                self.assertTrue(annotations[1]['block_results'][0]['sql_evidence'])
                self.assertEqual(annotations[1]['block_results'][0]['block_id'], 'q0:output_semantics')
                self.assertIn('SELECT', json.loads(annotations[1]['block_results'][0]['sections']))
                self.assertNotEqual(annotations[1]['cache_key'], annotations[2]['cache_key'])
                self.assertEqual(annotations[1]['agreement_cluster_id'], annotations[2]['agreement_cluster_id'])
                self.assertEqual(set(annotations[0]['labels'].values()), {None})
                self.assertEqual(output[1]['semantic_verifier'][0]['status'], 'invalid')
                manifest = json.loads((Path(args.output) / 'semantic_verifier.manifest.json').read_text())
                self.assertEqual(manifest['n_clusters'], 2)
                self.assertEqual(manifest['n_requirement_contracts'], 1)
                self.assertEqual(manifest['requirement_label_counts']['output_semantics']['S'], 2)
                self.assertEqual(manifest['batch_size'], 2 if run else 3)
                self.assertEqual(manifest['n_all_invalid_or_no_valid_cluster'], 1)
                self.assertTrue((Path(args.output) / 'verifier_trace.jsonl').is_file())
                _, _, questions = sv.dataset_questions(args)
                self.assertEqual(questions[0].clusters[0].vote_count, 1)
                self.assertEqual(questions[1].clusters, ())
            self.assertEqual(len(backend.calls), 11)
            prompt = backend.calls[1]['user']
            self.assertIn('CREATE TABLE items', prompt)
            self.assertNotIn('SECRET_SOURCE', prompt)
            self.assertNotIn('BAD SQL', prompt)
            self.assertNotIn('SELECT items.id', prompt)
            self.assertNotIn('vote_count', prompt)

    def test_single_perspective_dataset_preserves_skipped_candidate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'input.jsonl'
            source.write_text(json.dumps({'question_id': 'q', 'question': 'Return tables',
                'task_type': 'prepbench', 'candidates': [{'text': 'def solve(tables): return tables'},
                                                       {'text': '', 'invalid_mask': True}]}) + '\n')
            backend = FakeBackend()
            with patch.object(cli, 'make_backend', return_value=backend):
                cli.main(['transform-dataset', '--config', 'training_configs/verifier_qwen3_1p7b.json',
                          '--input', str(source), '--output', str(root / 'out'),
                          '--perspective', 'relation_semantics'])
            row = load_from_disk(root / 'out')['test'][0]
            self.assertEqual(row['semantic_verifier'][0]['labels'], {'relation_semantics': 'S'})
            self.assertEqual(row['semantic_verifier'][1]['labels'], {'relation_semantics': None})
            manifest = json.loads((root / 'out/semantic_verifier.manifest.json').read_text())
            self.assertEqual(manifest['perspectives'], ['relation_semantics'])
            self.assertEqual(len(backend.calls), 1)

    def test_failed_run_has_diagnostics_but_no_saved_dataset(self):
        class Broken(FakeBackend):
            def generate(self, **kwargs):
                return 'not JSON'
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'input.jsonl'
            source.write_text(json.dumps({'question_id': 'q', 'question': 'Return tables',
                'task_type': 'prepbench', 'candidates': [{'text': 'def solve(tables): return tables'}]}) + '\n')
            with patch.object(cli, 'make_backend', return_value=Broken()):
                with self.assertRaises(json.JSONDecodeError):
                    cli.main(['transform-dataset', '--config', 'training_configs/verifier_qwen3_1p7b.json',
                              '--input', str(source), '--output', str(root / 'out')])
            self.assertFalse((root / 'out/dataset_dict.json').exists())
            self.assertFalse((root / 'out/semantic_verifier.manifest.json').exists())
            events = [json.loads(line) for line in (root / 'out/verifier_trace.jsonl').read_text().splitlines()]
            self.assertEqual(events[-1]['raw_text'], 'not JSON')


if __name__ == '__main__':
    unittest.main()
