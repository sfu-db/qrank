from datasets import load_from_disk
from qrank.verifier.sql import summarize_sql
from pathlib import Path
import json
seen=set()
for row in load_from_disk('dataset/bird_dev_agr_q_sql_af_AQXRO_ss_m5')['test']:
 for c in row['candidates']:
  sql=c['text'].strip()
  if c['invalid_mask'] or not sql or sql in seen: continue
  summarize_sql(sql,'sqlite')
  seen.add(sql)
Path('outputs/verifier-audit/ast-coverage.json').write_text(json.dumps({'distinct_valid_sql':len(seen),'parsed':len(seen)},indent=2))
print('Complete:',len(seen))
