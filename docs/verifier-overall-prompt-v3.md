# Overall verifier prompt v3 pilot

The first 100-question AXRO pilot has now been run. Its paired results are in
[the v3 pilot analysis](verifier-overall-AXRO-v3-pilot-analysis.md). V3 lowers
benchmark-negative S but also loses many benchmark-positive S judgments and
produces some demonstrably incorrect C explanations. Do not scale it based on
this pilot.

`--overall-prompt v3` is an opt-in experiment for the SQL overall verifier.
The default remains the saved AXRO v2 prompt. V3 changes only the SQL request:

1. It tells the model to form a checklist from the question and evidence, then
   compare **every** applicable requirement with each candidate's actual SQL.
   The comparison explicitly includes missing operations.
2. It makes `S` conditional on satisfying the whole checklist. A definite
   result-changing mismatch gets `C`; a genuinely missing fact gets `U` only
   when no definite contradiction is already known.
3. It asks `C` reasons to identify the requirement and exact wrong or missing
   SQL behavior, and tells the model to check its explanation against the SQL.
4. It presents candidates as a flat list, without execution-agreement groups.
   Agreement is useful downstream but can distract from candidate-specific
   semantic checking.

The prompt does not contain BIRD-dev cases, benchmark labels, or source names.
It keeps the same JSON verdict schema and candidate set as v2. V3 has a
different prompt version and cache key, so it can share a cache safely with v2.
The exact prompt is `SQL_SYSTEM_V3` in `src/qrank/verifier/overall.py`.
API callers can pass `prompt_variant="v3"` to `Verifier.verify_overall(...)` or
`Verifier.verify_overall_candidates(...)`.

Run a matched pilot when the GPU is available:

```bash
uv run --no-sync python -m qrank.verifier transform-overall-dataset \
  --config training_configs/verifier_qwen3_1p7b.json \
  --input dataset/raw/bird_dev_agr_q_sql_af_AQXNRSO_ss_m7 \
  --models Arctic-R1-7B XiYanSQL-7B rnj-1-8B OmniSQL-7B \
  --overall-prompt v3 \
  --limit 100 \
  --output dataset/processed/bird_dev_overall_verifier_AXRO_v3_pilot

uv run --no-sync python scripts/analyze_overall_verifier.py \
  dataset/processed/bird_dev_overall_verifier_AXRO --limit 100
uv run --no-sync python scripts/analyze_overall_verifier.py \
  dataset/processed/bird_dev_overall_verifier_AXRO_v3_pilot
```

Compare the same questions and candidate SQL. The first 100 questions are a
development diagnostic, not a general accuracy estimate. Inspect false `S`,
false `C`, and explanation errors directly. For a credible improvement claim,
use a question-disjoint human-reviewed set with known semantic contradictions;
report reviewed `C` recall and false-`S` rate alongside precision and `U` rate.
Benchmark-negative candidates are a useful auxiliary signal, but are not all
reviewed `C` cases. A lower overall `S` rate alone is insufficient evidence.
