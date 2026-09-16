import json
import unittest
from sqlglot.errors import ParseError
from qrank.verifier.sql import summarize_sql
from qrank.verifier.responses import parse_response


class SQLSummaryTests(unittest.TestCase):
    def test_clauses_and_boolean_structure(self):
        sql = 'SELECT customer_id, AVG(amount) FROM sales WHERE year = 2024 AND (region = 1 OR region = 2) GROUP BY customer_id HAVING COUNT(*) > 1 ORDER BY AVG(amount) DESC LIMIT 5'
        summary = summarize_sql(sql, 'sqlite')
        for text in ('SELECT:\n    customer_id\n    AVG(amount)', 'AGGREGATION:\n    AVG(amount)\n    COUNT(*)',
                     'year = 2024 AND (region = 1 OR region = 2)', 'GROUP BY customer_id',
                     'HAVING COUNT(*) > 1', 'ORDER BY AVG(amount) DESC', 'LIMIT 5'):
            self.assertIn(text, summary)
        self.assertEqual(summary, summarize_sql(sql, 'sqlite'))

    def test_nested_aggregates_stay_in_their_query_blocks(self):
        summary = summarize_sql('WITH c AS (SELECT id, SUM(x) s FROM t GROUP BY id) SELECT id FROM c WHERE s > (SELECT AVG(x) FROM t) ORDER BY id LIMIT 5', 'sqlite')
        root, scalar, cte = summary.split('QUERY ')[1:]
        self.assertIn('c: q2', root)
        self.assertIn('AGGREGATION:\n    <none>', root)
        self.assertIn('AGGREGATION:\n    AVG(x)', scalar)
        self.assertIn('AGGREGATION:\n    SUM(x)', cte)
        self.assertNotIn('CORRELATED REFERENCES', summary)

    def test_window_aggregate_is_not_a_group_aggregate(self):
        summary = summarize_sql('SELECT id, SUM(x) OVER (PARTITION BY id ORDER BY day ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) FROM t', 'sqlite')
        self.assertIn('AGGREGATION:\n    <none>', summary)
        self.assertIn('WINDOW EXPRESSIONS:\n    SUM(x) OVER', summary)
        self.assertIn('ROWS BETWEEN 2 PRECEDING AND CURRENT ROW', summary)

    def test_union_limit_and_correlated_reference(self):
        union = summarize_sql('SELECT id FROM a UNION ALL SELECT id FROM b ORDER BY id LIMIT 5', 'sqlite')
        self.assertIn('SET OPERATION:\n    UNION ALL', union)
        root = union.split('QUERY ')[1]
        self.assertIn('LIMIT 5', root)
        correlated = summarize_sql('SELECT t.id FROM t WHERE EXISTS (SELECT 1 FROM u WHERE u.id = t.id)', 'sqlite')
        self.assertIn('EXTERNAL REFERENCES (binding requires schema):\n    t.id', correlated)

    def test_parse_failures_are_not_repaired(self):
        with self.assertRaises(ParseError):
            summarize_sql('SELECT x FROM (SELECT x FROM t', 'sqlite')
        for sql in ('SELECT 1; SELECT 2', 'DELETE FROM t', ''):
            with self.subTest(sql=sql), self.assertRaises(ValueError):
                summarize_sql(sql, 'sqlite')

    def test_perspective_response_has_no_overall_or_label_repair(self):
        self.assertEqual(parse_response('{"label":"C","reason":"Wrong population"}', 'constraint_semantics'), ('C', 'Wrong population'))
        for raw in ('bad JSON', '```json\n{}\n```'):
            with self.assertRaises(json.JSONDecodeError):
                parse_response(raw, 'output_semantics')
        for label in ('SATISFIED', 'NA'):
            with self.assertRaises(RuntimeError):
                parse_response(json.dumps({'label': label, 'reason': 'fixture'}), 'output_semantics')
