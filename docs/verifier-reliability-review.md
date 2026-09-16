# Verifier reliability review: BIRD 8B, 2026-09-11

## Implementation update

The current implementation follows the clarified objective: each request judges
one perspective independently, using the original task/context plus raw SQL and
a deterministic AST summary. It produces S/U/C/NA and a reason. There is no task
extraction, obligation contract, review/fallback pass, or overall verdict.
SQL parsing, JSON and backend errors propagate directly. vLLM uses xgrammar
structured responses; the CLI retains raw prompts/completions for debugging.

The remainder preserves the historical v8 audit and original proposal. Its
multi-stage workflow recommendations are historical and were not adopted.

The current run has both response-format failures and semantic judgment errors.
Changing the target S/U/C/NA proportions would not establish reliability. The
recommended next experiment separates task interpretation from candidate
evaluation, after fixing response handling and measurement.

This review inspects the current working-tree code and
`dataset/processed/bird_dev_agr_q_sql_af_AQXRO_ss_m5_8b_0911`. It does not rerun
inference or change verifier behavior. The manifest and current prompt both
identify version `qrank-svem-semantic-v8`; the manifest does not pin the source
commit or context snapshot, so exact historical context is not recoverable from
the manifest alone.

## Findings from the saved output

Counts below use one representative per valid agreement cluster, matching the
manifest. All counts were reconciled against it.

- 2,821 clusters across 1,534 questions; 2,092 successful judgments and 729
  failures (25.84%). Failures affect 538 questions.
- All 729 final recorded failures are JSON parsing errors: 724 missing-comma
  errors, four invalid escapes, and one unquoted-property error. Of these, 722
  report `Expecting ',' delimiter: line 21 column 2`. That repeated position
  suggests a systematic formatting problem, but the malformed responses were
  discarded, so the precise cause cannot be established retrospectively.
- The successful-response cache contains 2,090 entries, all v8; two additional
  cluster judgments were marked cached.

Successful judgments alone have this distribution:

| Perspective | S | U | C | NA |
| --- | ---: | ---: | ---: | ---: |
| Output | 981 | 41 | 1,069 | 1 |
| Constraint | 1,243 | 342 | 504 | 3 |
| Ranking | 455 | 18 | 177 | 1,442 |
| Relation | 1,522 | 371 | 197 | 2 |
| Overall | 998 | 48 | 1,046 | 0 |

Every perspective's original NA count includes 729 operational failures. The
remaining 1,442 ranking NA judgments may be appropriate for questions without
ranking requirements; a large NA proportion alone does not demonstrate error.
Conversely, near-zero constraint/relation NA does not establish that these
perspectives were applicable: the reasons sometimes describe a filter or a
single-table reference as satisfying a different dimension.

There are 25 successful judgments with a C feature and overall S, three with a
C feature and overall U, and three with a U feature and overall S. Ranking
applicability changes between successful clusters for the same question in 26
questions, despite the prompt defining applicability from the task.

Against the dataset's existing candidate correctness labels, using each
representative's label:

| Overall verdict | Benchmark true | Benchmark false |
| --- | ---: | ---: |
| S | 505 | 493 |
| U | 7 | 41 |
| C | 220 | 826 |

Only 50.6% of overall-S representatives are benchmark-positive; 21.0% of
overall-C representatives are benchmark-positive. These are useful disagreement
signals, conditional on successful verification. They are not per-feature
accuracy estimates: whole-query benchmark labels do not identify which feature
is wrong, and agreement on a database instance need not establish semantic
equivalence. There are no mixed benchmark labels within the saved clusters.

## Concrete semantic failures

Question IDs are the dataset IDs, with zero-based candidate indices.

1. **Inventing a condition despite explicit evidence: q5, candidate 0.** Evidence
   defines exclusively virtual as `Virtual = 'F'`. The SQL uses that exact filter
   and `AvgScrMath < 400`. The verifier assigns constraint/output C because it
   says the filter does not ensure exclusivity. This directly disregards the
   supplied definition.
2. **Inventing output requirements: q6, candidate 0.** The question asks to list
   qualifying schools; the SQL selects school names. Output C demands additional
   attributes such as `CDSCode`, without grounding that demand in the question.
3. **Missing a formula mismatch: q0, candidate 4.** Evidence defines a rate as
   free-meal count divided by enrollment. The SQL multiplies that expression by
   100; the verifier marks all dimensions S and says the formula matches.
4. **Label/reason disagreement: q40, candidate 0.** Output is labeled C while its
   reason explicitly says the selected phone number satisfies the requirement;
   overall is S. Mechanically changing overall to C would preserve the wrong
   local judgment rather than resolve this conflict.
5. **Incorrectly inherited explanation: q0, candidate 3.** This candidate selects
   from `frpm` alone, but its inherited relation reason says it joins `frpm` to
   `schools`. That explanation belongs to the cluster representative.

These examples establish problems beyond the distribution. They are illustrative
cases, not a randomly sampled estimate of each error category's frequency.

## Changes to the workflow

### 1. Repair observability and the response contract first

The failure path in `src/qrank/verifier/pipeline.py` assigns NA to every
perspective and replaces raw output with an empty string. Retries repeat the
same system/user messages. The run uses temperature zero, so these retries do
not provide an intentional alternative to a malformed generation.

- Keep verification availability in `status`; use absent/null judgments for
  failures in a versioned output contract. Preserve NA exclusively for a judged
  absence of task obligations. During migration, exclude failed/invalid entries
  from semantic counts even if their legacy labels remain NA.
- Record raw completions, parser errors, finish reasons, token counts, and
  attempts in an append-only audit record. Do not cache failed attempts as valid
  judgments. Record a context hash/snapshot, source revision, and contract version.
- Generate schema-constrained JSON with required fields and label enums. The
  installed vLLM 0.11.0 exposes `StructuredOutputsParams`; its
  [versioned documentation](https://docs.vllm.ai/en/v0.11.0/features/structured_outputs.html#offline-inference)
  describes JSON constraints through `SamplingParams`. This addresses formatting,
  not semantic accuracy.
- Use failure-specific recovery: retry transient backend errors; address context
  overflow or truncation explicitly; use a bounded format-repair request when
  necessary. Do not assume a larger token budget fixes these observed errors.
- Align prompt and parser: v8 advertises overall NA, while the parser rejects it.
  Define output as always applicable for valid tasks. Require meaningful evidence
  fields instead of accepting labels with empty reasons.
- Report operational coverage, genuine NA, and S/U/C separately. Define a pilot
  completion gate before running the full dataset.

### 2. Establish task obligations once per question

Before presenting any candidate, derive a small task contract from the question,
evidence, and schema. Record each obligation's ID, primary perspective, source
text, interpretation, and unresolved ambiguity. Examples for q5:

- output: count qualifying schools;
- constraint: Math average below 400;
- constraint: exclusively virtual means `Virtual = 'F'`;
- relation: associate each school's SAT record with that school;
- ranking: no obligation.

Reuse this contract for all candidates. This makes applicability consistent and
lets reviewers inspect invented requirements directly. An extracted contract is
itself a model prediction: require source grounding, test its completeness, and
route ambiguous interpretations for review. Do not freeze unsupported guesses
as authoritative requirements.

Keep explicit dialect, keys, types, and relevant column/value descriptions in
context. The current SQL context builder defaults to an unspecified dialect;
the BIRD input adapter should explicitly supply SQLite. Local BIRD
`database_description/*.csv` files are available but are not loaded by the
current DDL loader. Use relevant descriptions with provenance; sample values
should not be treated as guarantees about all rows.

### 3. Judge atomic obligations against actual candidate behavior

For each obligation, produce a compact record containing its ID, the relevant
SQL expression/operation, the established behavior, S/U/C, and the supporting
fact or missing fact. Source and SQL snippets make judgments reviewable; the
presence of a snippet alone does not prove the model's interpretation.

Use clear primary responsibilities:

| Perspective | Primary checks |
| --- | --- |
| Output | Requested values, projection, aggregate targets, units/formulas |
| Constraint | Population, predicates, literals, boundaries |
| Ranking | Ranking quantity, direction, k, scope, specified tie policy |
| Relation | Entity relationships, join keys, row grain, grouping/cardinality |

Record dependencies where an error affects other results, rather than repeating
one filter mismatch as unsupported contradictions in every dimension. A
candidate can also introduce a harmful operation absent from the task, such as
an unwanted LIMIT; the check must catch that as a violation of required output
coverage even if ranking itself is NA.

Start with one structured candidate call over the explicit obligation list.
Evaluate separate perspective calls as an ablation if local errors persist;
four calls are a cost increase and are not inherently more reliable. Deterministic
SQL analysis can supply syntax facts, but AST similarity should not substitute
for semantic equivalence.

### 4. Validate evidence, then aggregate and review

After resolving contradictory label/reason records, aggregate applicable atomic
checks in code: C if any established result-changing contradiction remains;
otherwise U if a material obligation remains unresolved; otherwise S. A
perspective is NA only when no task obligation applies. A valid assessed
candidate has no overall NA.

Do not apply this rule blindly to existing outputs: q40 shows why local C labels
must be checked first. Send unsupported contradictions, missing coverage, and
material unknowns to a targeted review pass. Audit a random subset of accepted S
judgments too, since false acceptance is substantial in the saved benchmark
comparison. A second model or repeated call is additional evidence, not ground
truth; measure the review's incremental benefit.

### 5. Verify distinct candidate SQL before pooling cluster evidence

The dataset pipeline verifies the first member of an agreement cluster and
copies its labels and reasons to every member. Of 2,821 clusters, 1,394 contain
multiple distinct stripped SQL strings. There are 6,684 valid candidates and
5,537 distinct stripped SQL strings when deduplicated within each question.

For candidate-specific semantic features, deduplicate identical SQL under the
same task/context and verify other candidates separately. That is approximately
1.96 times the current representative count before additional stages or broader
deduplication. Retain execution-agreement clusters as separate downstream
evidence. If representative-only verification remains an explicit budget choice,
attach provenance and label it as a representative assessment; do not present its
SQL-specific explanation as an assessment of every member.

## Pilot and acceptance criteria

Use a development sample of roughly 100–200 questions stratified by database,
difficulty, ranking applicability, failures, and benchmark disagreement. Include
all distinct candidates for sampled questions. Keep a separate held-out question
set; avoid splitting candidates from one question across development/evaluation.
Freeze the sample and rubric before comparing alternatives.

Have reviewers annotate applicability and atomic obligations, then per-feature
S/U/C with grounded reasons. Use benchmark correctness labels only for offline
evaluation, never in verifier or requirement-extraction prompts.

Compare incrementally:

1. Existing v8 results as the baseline.
2. Same judgment workflow with structured responses and operational metrics fixed.
3. Shared task contracts, explicit context, and evidence-backed obligation checks.
4. Targeted review and, if useful, separate perspective calls or model/reasoning
   configuration changes.

Measure parse success, coverage, applicability agreement, C precision, S
precision, per-feature confusion matrices, abstention rate with error among
decided cases, label/reason consistency, and inference cost. Include equivalent
SQL rewrites and controlled semantic mutations (wrong literal, wrong aggregate,
reversed ranking, wrong join key), with expected effects reviewed rather than
assumed. Measure downstream selection accuracy separately.

Proposed initial gates are at least 99% valid structured responses, no accepted
contract inconsistencies, and improved S/C precision on the held-out annotations
without obtaining that improvement solely by increasing abstentions. Numerical
semantic precision thresholds should follow the downstream error costs and the
pilot's uncertainty intervals. A more balanced label histogram is not an
acceptance criterion.

The first implementation slice should cover response constraints, failure
logging/counts, and a replayable pilot. Then introduce task contracts and distinct
candidate verification as measured changes. The SQL-specific v8 system prompt
also needs separation from the PrepBench prompt before applying this redesign to
Python/pandas tasks.

## Reproduce and inspect

From the repository root:

```bash
.venv/bin/python outputs/verifier-audit/bird_dev_8b_0911/audit.py
```

The audit reads the existing dataset and writes only derived artifacts:

- `outputs/verifier-audit/bird_dev_8b_0911/audit.json`: counts and example cases;
- `outputs/verifier-audit/bird_dev_8b_0911/cases.jsonl`: one representative record
  per valid cluster, for inspection and pilot selection.

The script verifies that the counts reconcile with the saved manifest. No model
inference, semantic accuracy benchmark, or production-code tests were run for
this review.
