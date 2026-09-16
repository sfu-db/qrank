"""Deterministic responses for verifier plumbing tests, not semantic evaluation."""
import json
from qrank.llms import LLMBackend


def response_for(request, label=None):
    if 'REQUIREMENT EXTRACTION' in request.get('system', ''):
        source = json.loads(request['user'].split('SOURCE INPUTS:\n')[1])
        return json.dumps({'requirements': [{
            'requirement_id': 'r1', 'dimension': 'output_semantics',
            'text': 'Return the requested output',
            'source_refs': [{'field': 'question', 'quote': source['question']}],
            'ambiguity': None,
        }], 'unresolved': []})
    if '\n\nTARGET CHECK:\n' in request['user']:
        target = json.loads(request['user'].split('\n\nTARGET CHECK:\n')[1])
        contract = json.loads(request['user'].split('\n\nFIXED REQUIREMENTS:\n')[1].split('\n\nTARGET CHECK:')[0])
        sql = request['user'].split('RAW CANDIDATE SQL:\n')[1].split('\n\nDETERMINISTIC AST SUMMARY:')[0]
        section = next((name for name, values in target['sections'].items() if values), None)
        label = label or ('C' if 'SELECT wrong' in sql else 'S')
        if section is None or not target['allowed_requirement_ids']:
            label = 'NA'
        return json.dumps({
            'label': label, 'reason': 'Fixture judgment',
            'requirement_ids': [target['requirement_id'] or target['allowed_requirement_ids'][0]] if target['allowed_requirement_ids'] else [],
            'source_refs': contract['requirements'][0]['source_refs'],
            'sql_evidence': [{'scope_id': target['scope_id'], 'section': section,
                              'operation_index': 0, 'fragment': sql}] if section else [],
            'inspected_scope_ids': [target['scope_id']],
            'missing_behavior': None, 'conflict_type': 'wrong_operation' if label == 'C' else None,
            'unresolved_reason': 'Fixture unknown' if label == 'U' else None,
            'applicability_reason': 'No local operation' if label == 'NA' else None,
        })
    return json.dumps({'label': label or ('C' if 'SELECT wrong' in request['user'] else 'S'),
                       'reason': 'Fixture judgment'})


class FakeBackend(LLMBackend):
    backend_name = 'test'

    def __init__(self):
        super().__init__('test')
        self.calls = []
        self.batches = []

    def generate_batch(self, requests):
        self.batches.append(list(requests))
        return super().generate_batch(requests)

    def generate(self, **request):
        self.calls.append(request)
        return response_for(request)
