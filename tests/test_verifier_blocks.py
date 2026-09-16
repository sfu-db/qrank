import json
import tempfile
import unittest
from pathlib import Path

from qrank.verifier import SQLiteCache, Verifier
from qrank.verifier.sql import decompose_sql
from verifier_fixtures import FakeBackend, response_for


class BlockTests(unittest.TestCase):
    def test_scoped_signals_are_not_aggregated_and_are_cached(self):
        class ScopedBackend(FakeBackend):
            def generate(self, **request):
                self.calls.append(request)
                if 'TARGET CHECK:\n' not in request['user']:
                    return response_for(request)
                target = json.loads(request['user'].split('TARGET CHECK:\n')[1])
                return response_for(request, label='C' if target['scope_id'] == 'q1' else 'S')

        backend = ScopedBackend()
        with tempfile.TemporaryDirectory() as tmp:
            verifier = Verifier(backend, cache=SQLiteCache(str(Path(tmp) / 'cache')))
            sql = 'WITH c AS (SELECT id FROM t WHERE x = 2) SELECT id FROM c'
            first = verifier.verify('List IDs with x = 1', sql)
            second = verifier.verify('List IDs with x = 1', sql)
        self.assertEqual(len(first.block_results), 8)
        self.assertEqual(first.labels['output_semantics'], 'S')
        self.assertEqual(next(b['label'] for b in first.block_results if b['block_id'] == 'q1:output_semantics'), 'C')
        self.assertEqual(len(backend.calls), 10)
        self.assertEqual(second.status, 'cached')
        self.assertEqual(len({b['cache_key'] for b in first.block_results}), 8)
        target = json.loads(backend.calls[2]['user'].split('TARGET CHECK:\n')[1])
        self.assertEqual(target['sections']['FILTERS (WHERE)'], [])


    def test_boolean_and_set_dependencies_survive_decomposition(self):
        blocks = decompose_sql('SELECT id FROM a WHERE x = 1 AND (y = 2 OR y = 3) UNION ALL SELECT id FROM b')
        self.assertEqual(blocks[0]['sections']['SET OPERATION'], ['UNION ALL', 'inputs: q2, q1'])
        filtered = next(b for b in blocks if b['sections']['FILTERS (WHERE)'])
        self.assertEqual(filtered['sections']['FILTERS (WHERE)'], ['WHERE x = 1 AND (y = 2 OR y = 3)'])
        self.assertEqual(filtered['parent_scope_id'], 'q0')
