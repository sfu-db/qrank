import dataclasses
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from datasets import load_from_disk
from qrank.verifier import pipeline as sv
from qrank.verifier import cli, context
from qrank.verifier import build_context
from qrank.verifier.prompts import SYSTEM_PROMPT
from test_verifier_pipeline import FakeBackend


class ContextTests(unittest.TestCase):
    def test_sql_schema_keeps_row_context(self):
        context = build_context({'question': 'Q', 'group_id': 'db', 'context': 'Extra hints'},
                                schemas={'db': 'CREATE TABLE t (id INT)'})
        self.assertEqual(context.task_type, 'text-to-sql')
        self.assertIn('CREATE TABLE', context.context)
        self.assertIn('Extra hints', context.context)
        self.assertIn('Task: text-to-SQL', context.context)

    def test_prepbench_only_reads_input_samples(self):
        with tempfile.TemporaryDirectory() as tmp:
            case = Path(tmp) / 'case_001'
            (case / 'inputs').mkdir(parents=True)
            (case / 'inputs/input_01.csv').write_text('id,value\n1,blue\n')
            (case / 'output_01.csv').write_text('SECRET_GOLD')
            row = {'question_id': 'case_001', 'question': 'Clean data',
                   'task_type': 'prepbench', 'metadata': {'gold': 'SECRET_GOLD'}}
            context = build_context(row, prepbench_root=tmp, schemas={'unrelated': 'SECRET_SQL'})
            self.assertIn('input_01.csv', context.context)
            self.assertIn('blue', context.context)
            self.assertNotIn('SECRET', context.context)
            with self.assertRaisesRegex(ValueError, 'No input CSV'):
                build_context({**row, 'question_id': 'missing'}, prepbench_root=tmp)
            with self.assertRaisesRegex(ValueError, 'within'):
                build_context({**row, 'question_id': '../../escape'}, prepbench_root=tmp)

    def test_prepbench_jsonl_alignment_prompt_and_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            row = {'question_id': 'case_001', 'question': 'Clean data', 'candidates': [
                {'text': 'def solve(tables): return tables', 'cluster_id': 'a', 'source': 'SECRET'},
                {'text': 'OTHER_CANDIDATE', 'cluster_id': 'a', 'source': 'SECRET'},
                {'text': '', 'cluster_id': 'b', 'source': 'SECRET'},
            ]}
            source = root / 'prepbench.jsonl'
            source.write_text(json.dumps(row) + '\n')
            args = cli.build_parser().parse_args(['transform-dataset', '--input', str(source),
                                                '--output', str(root / 'output')])
            backend = FakeBackend()
            with patch.object(sv, 'load_database_schemas', side_effect=AssertionError('SQL discovery')), \
                 patch.object(cli, 'make_backend', return_value=backend):
                self.assertEqual(args.func(args), 0)
            result = load_from_disk(args.output)['test'][0]
            self.assertEqual(result['candidates'], row['candidates'])
            annotations = result['semantic_verifier']
            self.assertEqual(annotations[0], annotations[1])
            self.assertEqual(annotations[2]['status'], 'invalid')
            self.assertEqual(len(backend.calls), 1)
            prompt = backend.calls[0]
            self.assertEqual(SYSTEM_PROMPT, prompt['system'])
            self.assertIn('Task: PrepBench', prompt['user'])
            self.assertIn('solve(tables: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]', prompt['user'])
            self.assertIn('requested output filenames', prompt['user'])
            self.assertIn('runtime globals are unspecified', prompt['user'])
            self.assertIn('CANDIDATE PYTHON/PANDAS CODE', prompt['user'])
            self.assertIn('not provided', prompt['user'])
            self.assertNotIn('SECRET', prompt['user'])
            self.assertNotIn('OTHER_CANDIDATE', prompt['user'])
            q = sv.QuestionInput('q', 'Q', '', '', '', '', (), {}, 'prepbench')
            c = sv.ClusterInput('0', 'code', 1, {})
            key = sv.make_cache_key(q, c, backend)
            self.assertNotEqual(key, sv.make_cache_key(dataclasses.replace(q, task_type='text-to-sql'), c, backend))
            self.assertEqual(key, sv.make_cache_key(q, dataclasses.replace(c, vote_count=9, metadata={'gold': True}), backend))


if __name__ == '__main__':
    unittest.main()
