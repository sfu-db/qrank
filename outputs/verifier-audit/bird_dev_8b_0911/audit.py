from datasets import load_from_disk
from collections import Counter, defaultdict
from pathlib import Path
import json, re
p=Path('dataset/processed/bird_dev_agr_q_sql_af_AQXRO_ss_m5_8b_0911')
d=load_from_disk(str(p))['test']
status=Counter(); errors=Counter(); error_positions=Counter(); counts=defaultdict(Counter); mismatch=Counter(); cross=defaultdict(Counter); examples=defaultdict(list); db=defaultdict(Counter); applicability=Counter(); affected=0; multiple_distinct=0; unique_sql=0; valid_candidates=0; mixed_truth=0; all_cases=[]
for row in d:
 groups=defaultdict(list)
 for i,(c,a) in enumerate(zip(row['candidates'],row['semantic_verifier'])):
  if a['status']!='invalid': groups[a['cluster_id']].append(i); valid_candidates+=1
 unique_sql+=len({row['candidates'][i]['text'].strip() for ids in groups.values() for i in ids})
 successful=[]; row_failed=False
 for cid,ids in groups.items():
  i=ids[0]; c=row['candidates'][i]; a=row['semantic_verifier'][i]; lab=a['labels']; truth=row['labels'][i]
  multiple_distinct+=len({row['candidates'][j]['text'].strip() for j in ids})>1
  mixed_truth+=len({row['labels'][j] for j in ids})>1
  status[a['status']]+=1; db[row['group_id']][a['status']]+=1
  case={'question_id':row['question_id'],'db_id':row['group_id'],'candidate_index':i,'members':ids,'question':row['question'],'evidence':row['evidence'],'sql':c['text'],'benchmark_correct':truth,'annotation':a}
  all_cases.append(case)
  if a['status']=='failed':
   row_failed=True
   reason=a['reasons']['overall']; errors[re.sub(r': line \d+ column \d+ \(char \d+\)', '', reason)]+=1
   pos=re.search(r'line \d+ column \d+',reason)
   if pos: error_positions[pos.group()]+=1
   if len(examples['failed'])<3: examples['failed'].append(case)
   continue
  successful.append(a)
  for k,v in lab.items(): counts[k][v]+=1; cross[k][f'{v}/benchmark_{truth}']+=1
  fs=[v for k,v in lab.items() if k!='overall']; o=lab['overall']
  issue='feature_C_overall_'+o if 'C' in fs and o!='C' else 'feature_U_overall_S' if 'U' in fs and o=='S' else 'overall_C_without_feature_C' if 'C' not in fs and o=='C' else None
  if issue:
   mismatch[issue]+=1
   if len(examples[issue])<3: examples[issue].append(case)
  if o=='C' and truth and len(examples['C_benchmark_correct'])<6: examples['C_benchmark_correct'].append(case)
  if o=='S' and not truth and len(examples['S_benchmark_incorrect'])<6: examples['S_benchmark_incorrect'].append(case)
 if row_failed: affected+=1
 if len(successful)>1:
  for k in counts:
   if k=='overall':continue
   if len({a['labels'][k]=='NA' for a in successful})>1: applicability[k]+=1
report={'dataset':str(p),'unit':'one representative per noninvalid agreement cluster','n_questions':len(d),'statuses':status,'questions_with_failures':affected,'failure_categories':errors,'top_failure_positions':error_positions.most_common(8),'successful_label_counts':counts,'feature_overall_inconsistencies':mismatch,'benchmark_cross_tabs':cross,'db_status':db,'questions_with_inconsistent_applicability':applicability,'n_valid_candidates':valid_candidates,'n_distinct_sql_per_question_summed':unique_sql,'clusters_with_multiple_distinct_sql':multiple_distinct,'clusters_with_mixed_benchmark_labels':mixed_truth,'examples':examples}
out=Path('outputs/verifier-audit/bird_dev_8b_0911');out.mkdir(parents=True,exist_ok=True)
(out/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
(out/'cases.jsonl').write_text(''.join(json.dumps(c)+'\n' for c in all_cases))
print(json.dumps({k:v for k,v in report.items() if k not in ('examples','db_status')},indent=2))
manifest = json.loads((p / "semantic_verifier.manifest.json").read_text())
assert sum(status.values()) == manifest["n_clusters"]
assert status["failed"] == manifest["n_failed_verifications"]
assert sum(errors.values()) == manifest["n_failed_verifications"]
for perspective, observed in counts.items():
    for label, expected in manifest["label_counts"][perspective].items():
        assert observed.get(label, 0) + (status["failed"] if label == "NA" else 0) == expected
print("Audit counts reconcile with the manifest.")
