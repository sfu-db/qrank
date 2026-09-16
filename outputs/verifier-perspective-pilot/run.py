from qrank.verifier import Verifier
from pathlib import Path
from dataclasses import asdict
import json

root=Path('outputs/verifier-perspective-pilot')
root.mkdir(exist_ok=True)
verifier=Verifier(config_path='training_configs/verifier_qwen3_8b.json', audit_path=root/'trace.jsonl')
question='Return the sum of salaries for employees in department Sales.'
context='CREATE TABLE employees (salary REAL, department TEXT);'
records=[]
for name,sql,expected in [
 ('correct',"SELECT SUM(salary) FROM employees WHERE department = 'Sales'",['S','S','NA','NA']),
 ('wrong_filter',"SELECT SUM(salary) FROM employees WHERE department = 'HR'",['S','C','NA','NA']),
 ('wrong_output',"SELECT AVG(salary) FROM employees WHERE department = 'Sales'",['C','S','NA','NA']),
]:
 result=verifier.verify(question,sql,context=context,sql_dialect='sqlite')
 records.append({'case':name,'sql':sql,'expected':expected,'result':asdict(result)})
 print(name,result.labels,flush=True)
(root/'results.json').write_text(json.dumps(records,indent=2))
