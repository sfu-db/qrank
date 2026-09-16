# AXRO v3 prompt pilot: first 100 BIRD-dev questions

This compares `dataset/processed/bird_dev_overall_verifier_AXRO_1` (v3) with
the first 100 rows of `dataset/processed/bird_dev_overall_verifier_AXRO` (v2).
The saved manifests specify the same source, four candidate models, Qwen2.5-
Coder-7B revision, and decoding settings. The task/evidence/context portions
of all 99 generated question prompts also match between trace files. Candidate source,
SQL text, and benchmark label match at every one of the 345 retained slots.
There are 116 benchmark-positive and 229 benchmark-negative candidates. One
question has no valid candidate. The comparison is paired by candidate; the
100 adjacent questions are a diagnostic subset, not a held-out sample.

| Benchmark label | v2 S | v2 U | v2 C | v3 S | v3 U | v3 C |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Positive (116) | 96 | 1 | 19 | 69 | 0 | 47 |
| Negative (229) | 146 | 6 | 77 | 110 | 1 | 118 |

V3 assigns S to 179/345 (51.9%), versus 242/345 (70.1%) for v2. Its
benchmark-negative S rate falls from 146/229 (63.8%) to 110/229 (48.0%).
However, its S recall on benchmark-positive candidates falls from 96/116
(82.8%) to 69/116 (59.5%). S precision is 96/242 (39.7%) for v2 and 69/179
(38.5%) for v3. C precision against benchmark-negative labels falls from
77/96 (80.2%) to 118/165 (71.5%). These changes are consistent with a more
aggressive C threshold, not better separation of the two benchmark classes.

Of the 89 candidate verdicts that changed, 42 benchmark-negative candidates
move from S to C, but so do 30 benchmark-positive candidates. Six negative
candidates move from C to S. At the question level, all candidates receive S
in 42/99 nonempty v2 questions and 33/99 v3 questions. A benchmark-positive
S remains available in 46/50 questions that have a positive candidate under
v2, but only 35/50 under v3.

## Explanation audit

The changed verdicts reveal failures that aggregate counts cannot explain:

- **Question 4:** All four retained queries omit the direct-funding condition.
  V3 still assigns all four S and explains only charter status and opening date.
- **Question 0:** V3 assigns C to two benchmark-positive queries using the
  stored eligible-free-rate column, then S to the benchmark-negative query
  that multiplies the specified count/enrollment ratio by 100. The model has
  not resolved the unit/scale question reliably.
- **Question 35:** V3 says the charter filter is absent even though the SQL
  contains `WHERE f."Charter School (Y/N)" = 1`.
- **Question 40:** V3 says `LIMIT 1` is missing even though the SQL has
  `ORDER BY ... ASC NULLS LAST LIMIT 1`.
- **Question 19:** Some C explanations discuss other candidate identifiers
  and NULL handling instead of establishing a candidate-specific task error.

Across the 318 distinct question-level judgments, 22 v3 reasons refer to a
different candidate identifier from their own. None of the v2 reasons do so
under the same simple `Candidate [A-Z]` text check. This is a direct sign that
the flat multi-candidate response did not make every explanation independent.

## Decision

Do not scale v3 on the basis of this pilot. It lowers false S against execution
labels by also losing many benchmark-positive S decisions, and its explanations
include demonstrably incorrect claims about the displayed SQL. Benchmark
labels are not reviewed semantic labels, so this pilot still cannot estimate
recall on manually established C cases.

The next controlled test should avoid a single response for several candidates
and require a candidate-specific grounding check before a C verdict. For
example, evaluate one candidate per call and have the response cite the exact
task requirement and SQL operation it claims is wrong or missing. Check those
citations before aggregation. Compare on a frozen reviewed contradiction set,
including equivalent rewrites, and keep C precision, S precision, and U rate
visible. Further wording changes on these same 100 questions alone risk
overfitting this diagnostic subset.

Reproduce the candidate tables with:

```bash
uv run --no-sync python scripts/analyze_overall_verifier.py \
  dataset/processed/bird_dev_overall_verifier_AXRO --limit 100
uv run --no-sync python scripts/analyze_overall_verifier.py \
  dataset/processed/bird_dev_overall_verifier_AXRO_1
```
