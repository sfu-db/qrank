"""Grounding and coverage contracts; fixtures do not measure LLM accuracy."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from qrank.verifier import SQLiteCache, Verifier, VerificationError
from qrank.verifier.models import QuestionInput
from qrank.verifier.prompts import requirements_request
from qrank.verifier.responses import grounded_response_schema, parse_grounded, parse_requirements, validate_schema
from qrank.verifier.sql import decompose_sql
from verifier_fixtures import FakeBackend, response_for


class GroundingTests(unittest.TestCase):
    def setUp(self):
        self.backend = FakeBackend()
        self.sql = 'SELECT id FROM items'
        self.result = Verifier(self.backend).verify('List IDs', self.sql)
        source = self.result.requirements['source_inputs']
        self.q = QuestionInput('', source['question'], source['evidence'], source['schema'], '', '', (), {})
        self.blocks = decompose_sql(self.sql)
        self.target = json.loads(self.backend.calls[1]['user'].split('TARGET CHECK:\n')[1])
        self.good = json.loads(response_for(self.backend.calls[1]))

    def parse(self, result, target=None, blocks=None, sql=None):
        return parse_grounded(json.dumps(result), q=self.q, sql=sql or self.sql,
                              requirements=self.result.requirements,
                              blocks=blocks or self.blocks, target=target or self.target)

    def test_s_and_c_both_require_grounding_and_valid_references(self):
        mutations = [
            {'source_refs': []}, {'sql_evidence': []}, {'requirement_ids': []},
            {'requirement_ids': ['r1', 'r1']}, {'inspected_scope_ids': ['q0', 'q0']},
            {'requirement_ids': ['unknown']}, {'inspected_scope_ids': ['q99']},
            {'source_refs': [{'field': 'question', 'quote': 'invented source'}]},
            {'reason': ' '}, {'overall': 'S'},
        ]
        for label in ['S', 'C']:
            for changes in mutations:
                response = {**self.good, 'label': label,
                            'conflict_type': 'wrong_output' if label == 'C' else None, **changes}
                with self.subTest(label=label, changes=changes), self.assertRaises(VerificationError):
                    self.parse(response)
        for changes in [{'scope_id': 'q99'}, {'section': 'NO SECTION'},
                        {'operation_index': 99}, {'operation_index': True}, {'fragment': 'not in raw SQL'}]:
            response = copy.deepcopy(self.good)
            response['sql_evidence'][0].update(changes)
            with self.subTest(changes=changes), self.assertRaises(VerificationError):
                self.parse(response)

    def test_u_and_na_are_distinct_and_root_output_cannot_be_na(self):
        uncertain = {**self.good, 'label': 'U', 'sql_evidence': [], 'unresolved_reason': 'Missing column meaning'}
        self.assertEqual(self.parse(uncertain)['label'], 'U')
        with self.assertRaises(VerificationError):
            self.parse({**uncertain, 'unresolved_reason': None})
        with self.assertRaisesRegex(VerificationError, 'root output'):
            self.parse({**self.good, 'label': 'NA', 'applicability_reason': 'No output'})
        relation = json.loads(self.backend.calls[4]['user'].split('TARGET CHECK:\n')[1])
        na = {**self.good, 'label': 'NA', 'requirement_ids': [], 'sql_evidence': [],
              'source_refs': [], 'applicability_reason': 'No relationship responsibility'}
        self.assertEqual(self.parse(na, target=relation)['label'], 'NA')

    def test_missing_operation_has_no_required_offending_fragment(self):
        target = json.loads(self.backend.calls[-1]['user'].split('TARGET CHECK:\n')[1])
        omitted = {**self.good, 'label': 'C', 'sql_evidence': [],
                   'missing_behavior': 'Required behavior absent from inspected scopes',
                   'conflict_type': 'missing_required_behavior'}
        self.assertEqual(self.parse(omitted, target=target)['label'], 'C')
        with self.assertRaisesRegex(VerificationError, 'coverage'):
            self.parse(omitted)
        with self.assertRaises(VerificationError):
            self.parse({**omitted, 'missing_behavior': None}, target=target)
        with self.assertRaisesRegex(VerificationError, 'assigned requirement'):
            self.parse({**self.good, 'label': 'NA', 'applicability_reason': 'No filter'}, target=target)

    def test_omission_requires_all_reachable_scopes_and_local_evidence_stays_local(self):
        sql = 'WITH c AS (SELECT id FROM items) SELECT id FROM c'
        blocks = decompose_sql(sql)
        coverage = {**self.target, 'observation_kind': 'requirement_coverage', 'requirement_id': 'r1'}
        omitted = {**self.good, 'label': 'C', 'sql_evidence': [],
                   'missing_behavior': 'Missing behavior', 'conflict_type': 'missing_required_behavior'}
        with self.assertRaisesRegex(VerificationError, 'all reachable'):
            self.parse(omitted, target=coverage, blocks=blocks, sql=sql)
        omitted['inspected_scope_ids'] = ['q0', 'q1']
        self.assertEqual(self.parse(omitted, target=coverage, blocks=blocks, sql=sql)['label'], 'C')
        inner = {**self.target, 'scope_id': 'q1', 'scope_role': 'intermediate'}
        response = {**self.good, 'inspected_scope_ids': ['q0', 'q1']}
        with self.assertRaisesRegex(VerificationError, 'target block'):
            self.parse(response, target=inner, blocks=blocks, sql=sql)

    def test_requirements_preserve_ambiguity_and_reject_invented_sources(self):
        request = requirements_request(self.q)
        contract = json.loads(response_for(request))
        contract['requirements'][0]['ambiguity'] = 'Meaning of IDs is unspecified'
        contract['unresolved'] = ['Entity identity needs clarification']
        parsed = parse_requirements(json.dumps(contract), self.q)
        self.assertEqual(parsed, contract)
        for update in [{'requirement_id': 'r2'}, {'dimension': 'unknown'},
                       {'source_refs': []}, {'source_refs': [{'field': 'question', 'quote': 'absent'}]}]:
            bad = copy.deepcopy(contract)
            bad['requirements'][0].update(update)
            with self.subTest(update=update), self.assertRaises(VerificationError):
                parse_requirements(json.dumps(bad), self.q)
        with self.assertRaises(VerificationError):
            parse_requirements(json.dumps({'requirements': [], 'unresolved': []}), self.q)

    def test_decoder_branches_enforce_label_fields_and_local_assignments(self):
        schema = grounded_response_schema(self.target, self.blocks)
        validate_schema(self.good, schema)
        with self.assertRaises(VerificationError):
            validate_schema({**self.good, 'applicability_reason': 'Applies'}, schema)
        with self.assertRaises(VerificationError):
            validate_schema({**self.good, 'label': 'U'}, schema)
        # Empty ranking blocks cannot borrow SELECT evidence to produce S.
        ranking = json.loads(self.backend.calls[3]['user'].split('TARGET CHECK:\n')[1])
        self.assertEqual([b['properties']['label']['enum'] for b in grounded_response_schema(ranking, self.blocks)['anyOf']], [['NA']])
        with self.assertRaisesRegex(VerificationError, 'assigned responsibility'):
            self.parse(self.good, target=ranking)

    def test_extra_limit_is_assigned_output_coverage_without_ranking_requirement(self):
        backend = FakeBackend()
        Verifier(backend).verify('List IDs', self.sql + ' LIMIT 1')
        target = json.loads(backend.calls[3]['user'].split('TARGET CHECK:\n')[1])
        self.assertEqual(target['allowed_requirement_ids'], ['r1'])
        self.assertEqual(target['sections']['LIMIT'], ['LIMIT 1'])

    def test_contract_is_candidate_independent_and_cached_separately(self):
        backend = FakeBackend()
        with tempfile.TemporaryDirectory() as tmp:
            cache = SQLiteCache(str(Path(tmp) / 'cache.sqlite'))
            verifier = Verifier(backend, cache=cache)
            results = verifier.verify_candidates('List IDs', ['SELECT id FROM items', 'SELECT wrong FROM items'])
            self.assertEqual(results[0].requirements, results[1].requirements)
            self.assertEqual(sum('REQUIREMENT EXTRACTION' in r['system'] for r in backend.calls), 1)
            self.assertNotIn('SELECT', backend.calls[0]['user'])
            verifier.verify('List IDs', 'SELECT items.id FROM items')
            self.assertEqual(sum('REQUIREMENT EXTRACTION' in r['system'] for r in backend.calls), 1)
            changed = verifier.verify('List IDs', self.sql, evidence='IDs must be distinct')
            self.assertNotEqual(changed.requirements['cache_key'], results[0].requirements['cache_key'])
            self.assertEqual(sum('REQUIREMENT EXTRACTION' in r['system'] for r in backend.calls), 2)
            # Corrupted raw cache entries are revalidated instead of trusting parsed_json.
            key = changed.requirements['cache_key']
            cache._con().execute('UPDATE verifier_cache SET raw_text = ? WHERE cache_key = ?', ('{}', key))
            cache._con().commit()
            with self.assertRaises(VerificationError):
                verifier.verify('List IDs', self.sql, evidence='IDs must be distinct')

    def test_unused_ctes_do_not_add_observations(self):
        backend = FakeBackend()
        result = Verifier(backend).verify('List IDs', 'WITH unused AS (SELECT 42) SELECT id FROM items')
        self.assertEqual(len(result.block_results), 4)
        self.assertEqual(len(result.requirement_results), 1)
        blocks = decompose_sql('WITH unused AS (SELECT 42) SELECT id FROM items')
        self.assertFalse(blocks[1]['reachable'])
        self.assertEqual({b['scope_id'] for b in result.block_results}, {'q0'})

    def test_invalid_sql_stops_before_extraction(self):
        backend = FakeBackend()
        with self.assertRaises(ValueError):
            Verifier(backend).verify('List IDs', 'DELETE FROM items')
        self.assertEqual(backend.calls, [])

    def test_real_requirement_omission_remains_separate_from_local_s(self):
        class OmissionBackend(FakeBackend):
            def generate(self, **request):
                self.calls.append(request)
                response = json.loads(response_for(request))
                if 'REQUIREMENT EXTRACTION' in request['system']:
                    response['requirements'].append({
                        'requirement_id': 'r2', 'dimension': 'constraint_semantics',
                        'text': 'Require direct funding', 'ambiguity': None,
                        'source_refs': [{'field': 'question', 'quote': 'directly funded'}],
                    })
                else:
                    target = json.loads(request['user'].split('TARGET CHECK:\n')[1])
                    if target['requirement_id'] == 'r2':
                        response.update(label='C', requirement_ids=['r2'], sql_evidence=[],
                                        source_refs=[{'field': 'question', 'quote': 'directly funded'}],
                                        missing_behavior='Direct funding restriction is absent',
                                        conflict_type='missing_required_behavior')
                return json.dumps(response)
        result = Verifier(OmissionBackend()).verify('List IDs of directly funded charter schools',
                                                   'SELECT id FROM schools WHERE Charter = 1')
        self.assertEqual(result.labels['constraint_semantics'], 'S')
        missing = next(r for r in result.requirement_results if r['requirement_id'] == 'r2')
        self.assertEqual(missing['label'], 'C')
        self.assertEqual(missing['sql_evidence'], [])
        self.assertNotIn('overall', result.labels)
        local = result.block_results[0]
        coverage = result.requirement_results[0]
        self.assertEqual(local['dependency_group_ids'], coverage['dependency_group_ids'])
        self.assertNotEqual(local['source_id'], coverage['source_id'])


if __name__ == '__main__':
    unittest.main()
