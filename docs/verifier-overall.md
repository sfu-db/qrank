# Overall semantic verifier

Use `transform-overall-dataset` to obtain one source-blind S/U/C/NA conclusion
per distinct valid candidate. The verifier sends **one prompt per question**
containing the task context and all distinct candidate texts, grouped by their
rebuilt execution-agreement clusters. The model returns one structured verdict
per candidate identifier. Separate questions can be batched for throughput with
`--batch-size`; this does not create extra candidate-level prompts. This is a
separate experimental path; the existing
`transform-dataset` block/requirement signals and saved datasets are unchanged.

```bash
uv run --no-sync python -m qrank.verifier transform-overall-dataset \
  --config training_configs/verifier_qwen3_1p7b.json \
  --input dataset/raw/bird_dev_agr_q_sql_af_AQXNRSO_ss_m7 \
  --models Arctic-R1-7B XiYanSQL-7B rnj-1-8B OmniSQL-7B \
  --output dataset/processed/bird_overall_verifier_pilot \
  --max-tokens 2048 \
  --limit 100
```

`--models` is required. Use the exact `candidates[].source` names from the raw
dataset; the example matches the AXRO judge pool. For each question, the
command locates exactly one candidate per requested model, removes candidates
already marked invalid or containing empty text, and slices the input agreement
matrix to the survivors in the given model order. It then forms clusters from
that rebuilt matrix. Missing or duplicate source names fail before inference.
The output dataset contains only the retained candidates, the matching
agreement submatrix, and aligned labels/annotations; it does not modify the raw
input.

`S` means no specific material error was identified and the candidate is likely
to satisfy the task. `C` requires a specific result-changing contradiction.
`U` means a missing or ambiguous fact prevents a defensible decision. `NA` is
reserved for a displayed item that cannot meaningfully be assessed as a
candidate for the task; it is not uncertainty or an error label. Invalid/empty
candidates were already removed, so NA should be rare. These are
model judgments, not correctness certificates. Several candidates for one
question can be S, or all can be C. The verifier does not select a winner.

The new `semantic_verifier_overall` list is aligned with the rebuilt
`candidates` list. Each entry has `identifier`, `label`, `reason`, `status`,
`cache_key`, and `agreement_cluster_id`. Invalid/empty requested candidates are removed before
verification. Distinct texts within an execution-agreement cluster are assessed
in the same prompt; identical stripped texts in a question share a judgment.
The prompt includes all distinct candidates but never their source names, vote
counts, or benchmark labels. Context/schema and optional documented database descriptions
use the same loaders as the existing verifier. No candidate is executed.

The command writes `semantic_verifier_overall.manifest.json`, a response cache,
and `verifier_trace.jsonl`. Malformed JSON, backend errors, and cache failures
stop the run; they are never turned into U. Use a new output directory for a
rerun. API callers can use `Verifier.verify_overall(...)` or
`Verifier.verify_overall_candidates(...)`.
The manifest reports `n_question_requests` (nonempty questions) and
`n_model_generations` (uncached generated question responses). Changing one
candidate changes that question-level cache key. Fewer completions do not
guarantee lower total token cost: each prompt is longer and the model must
produce all verdicts. Check prompt length and output-token limits on a pilot.

## Evaluation before scaling

The current AXRO judge run reports 55.2% task accuracy (847/1,534). On those
same rows, 1,024 have at least one benchmark-positive candidate, so its
benchmark oracle ceiling is 66.8%. The judge selects a benchmark-negative
candidate on 618 rows: 195 have a positive alternative, while 423 have no
positive candidate and should return `NONE` under this benchmark. It returns
`NONE` despite an available positive on another 69 rows.
These are execution-benchmark comparisons, not a measure of semantic truth;
the saved run uses one representative per agreement cluster.

Run the overall verifier on the *same candidate pool*, then evaluate per-candidate
S/C precision, U frequency, all-C rows, and ranking/selection accuracy under a
predeclared S-first policy. Compare with the judge at matched coverage and model
cost. Stratify a manual review by false S, false C, U, one-candidate questions,
and disagreement-cluster members. Include controlled semantic mutations and
equivalent rewrites. Split train/development and held-out evaluation by question,
not by candidate; never put benchmark labels in prompts. The overall mode is a
testable simplification, not an established performance improvement.
