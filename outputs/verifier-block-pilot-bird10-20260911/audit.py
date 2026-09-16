"""Audit the frozen ten-question pilot; benchmark labels are evaluation-only."""
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
import json

from datasets import load_from_disk

ROOT = Path(__file__).resolve().parent
PERSPECTIVES = ('output_semantics', 'constraint_semantics', 'ranking_semantics', 'relation_semantics')
data = load_from_disk(str(ROOT / 'dataset'))['test']
manifest = json.loads((ROOT / 'dataset/semantic_verifier.manifest.json').read_text())
counts = {p: Counter() for p in PERSPECTIVES}
root_counts = {p: Counter() for p in PERSPECTIVES}
cross = {p: defaultdict(Counter) for p in PERSPECTIVES}
patterns = defaultdict(Counter)
records = []
invalid = 0
valid = 0
for row in data:
    seen = set()
    for index, (candidate, result, benchmark) in enumerate(zip(
        row['candidates'], row['semantic_verifier'], row['labels'], strict=True
    )):
        if result['status'] == 'invalid':
            invalid += 1
            continue
        valid += 1
        if result['cluster_id'] in seen:
            continue
        seen.add(result['cluster_id'])
        blocks = [{**b, 'sections': json.loads(b['sections'])} for b in result['block_results']]
        record = {
            'question_id': row['question_id'], 'candidate_index': index,
            'question': row['question'], 'evidence': row['evidence'],
            'sql': candidate['text'], 'benchmark_correct': benchmark,
            'root_labels': result['labels'], 'block_results': blocks,
        }
        records.append(record)
        for p, label in result['labels'].items():
            root_counts[p][label] += 1
            cross[p][label][str(benchmark)] += 1
        for b in blocks:
            counts[b['perspective']][b['label']] += 1
        pattern = 'any_C' if any(b['label'] == 'C' for b in blocks) else (
            'U_without_C' if any(b['label'] == 'U' for b in blocks) else 'S_or_NA_only')
        patterns[pattern][str(benchmark)] += 1

for p in PERSPECTIVES:
    assert dict(counts[p]) == {k: v for k, v in manifest['block_label_counts'][p].items() if v}
    assert dict(root_counts[p]) == {k: v for k, v in manifest['label_counts'][p].items() if v}
assert len(records) == manifest['n_clusters']
events = [json.loads(line) for line in (ROOT / 'dataset/verifier_trace.jsonl').read_text().splitlines()]
generated = [e for e in events if e.get('status') == 'generated']
assert len(generated) == sum(sum(c.values()) for c in counts.values())
for e in generated:
    response = json.loads(e['raw_text'])
    assert set(response) == {'reason', 'label'}
    assert isinstance(response['reason'], str) and response['reason'].strip()
    assert response['label'] in e['request']['response_schema']['properties']['label']['enum']
requested = [e for e in events if e.get('status') == 'requested']
inference_seconds = (datetime.fromisoformat(generated[-1]['created_utc']) -
                     datetime.fromisoformat(requested[0]['created_utc'])).total_seconds()
summary = {
    'questions': len(data), 'candidate_slots': valid + invalid,
    'valid_candidate_slots': valid, 'invalid_candidate_slots': invalid,
    'distinct_valid_candidates': len(records), 'model_requests_completed': len(generated),
    'unique_query_scopes': len(generated) // len(PERSPECTIVES),
    'candidates_with_nested_scopes': sum(len(r['block_results']) > 4 for r in records),
    'inference_seconds_excluding_model_startup': inference_seconds,
    'distinct_candidate_benchmark_counts': Counter(str(r['benchmark_correct']) for r in records),
    'block_counts': counts, 'root_counts': root_counts,
    'root_labels_vs_candidate_benchmark': cross,
    'block_patterns_vs_candidate_benchmark': patterns,
    'limitations': [
        'First ten questions, all California schools; not a representative BIRD sample.',
        'Benchmark execution labels are not block-level semantic ground truth.',
        'Any-C is a diagnostic grouping, not an implemented candidate verdict.',
    ],
}
checks = json.loads((ROOT / 'spot_checks.json').read_text())
lookup = {(str(r['question_id']), r['candidate_index'], b['block_id']): b
          for r in records for b in r['block_results']}
check_results = []
for check in checks['checks']:
    observed = lookup[(str(check['question_id']), check['candidate_index'], check['block_id'])]
    check_results.append({**check, 'observed_label': observed['label'],
                          'observed_reason': observed['reason'],
                          'matches_expectation': check['expected_label'] == observed['label']})
summary['targeted_spot_checks'] = {
    'method': checks['method'], 'count': len(check_results),
    'matched': sum(c['matches_expectation'] for c in check_results),
    'results': check_results,
}
(ROOT / 'audit.json').write_text(json.dumps(summary, indent=2) + '\n')
(ROOT / 'cases.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in records))
lines = ['# BIRD ten-question verifier pilot: complete model judgments', '']
for r in records:
    lines += [f"## Question {r['question_id']}, candidate {r['candidate_index']}", '',
              r['question'], '', f"Evidence: {r['evidence'] or '(none)'}", '',
              f"Benchmark execution label: {r['benchmark_correct']}", '',
              '```sql', r['sql'], '```', '']
    for b in r['block_results']:
        lines += [f"- **{b['block_id']}: {b['label']}** — {b['reason']}"]
    lines.append('')
(ROOT / 'judgments.md').write_text('\n'.join(lines) + '\n')
print(json.dumps(summary, indent=2))
