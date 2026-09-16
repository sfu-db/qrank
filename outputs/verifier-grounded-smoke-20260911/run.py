"""Synthetic integration smoke; not an estimate of semantic accuracy."""
import dataclasses
import json
from pathlib import Path
import shutil
import time

from qrank.verifier import SQLiteCache, Verifier


def main():
    root = Path(__file__).resolve().parent
    config_path = root / 'config.json'
    if not config_path.exists():
        shutil.copyfile('outputs/verifier-block-pilot-bird10-20260911/pilot_config.json', config_path)
    verifier = Verifier(config_path=config_path, cache=SQLiteCache(str(root / 'cache.sqlite')),
                        audit_path=root / 'trace.jsonl')
    task = {
        'question': 'List IDs of all directly funded charter schools.',
        'evidence': "Directly funded means Funding = 'D'. Charter schools means Charter = 1.",
        'context': 'CREATE TABLE schools (id INTEGER PRIMARY KEY, Funding TEXT, Charter INTEGER)',
        'sql_dialect': 'sqlite',
        'candidates': [
            "SELECT id FROM schools WHERE Charter = 1 AND Funding = 'D'",
            'SELECT id FROM schools WHERE Charter = 1',
            "WITH funded AS (SELECT id, Charter FROM schools WHERE Funding = 'D') SELECT id FROM funded WHERE Charter = 1",
        ],
    }
    (root / 'input.json').write_text(json.dumps(task, indent=2))
    started = time.monotonic()
    results = verifier.verify_candidates(**task)
    (root / 'results.json').write_text(json.dumps([dataclasses.asdict(r) for r in results], indent=2))
    print(json.dumps({'seconds_after_initialization': time.monotonic() - started,
                      'contract': results[0].requirements,
                      'signals': [{'root': r.labels,
                                   'coverage': [(x['requirement_id'], x['label'], x['missing_behavior'])
                                                for x in r.requirement_results]} for r in results]}, indent=2), flush=True)


if __name__ == '__main__':
    main()
