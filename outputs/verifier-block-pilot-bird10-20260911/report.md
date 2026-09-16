# BIRD-dev ten-question verifier pilot, 2026-09-11

The AST-block collection pipeline works after a GPU-memory adjustment, but this
pilot does not support trusting the current model judgments as useful semantic
error detectors. Keep the block-based architecture; improve judgment grounding,
scope attribution, and semantic validation before collecting signals at scale.

## Run and verification

Used the first ten rows (question IDs 0–9) of
`dataset/bird_dev_agr_q_sql_af_AQXRO_ss_m5`, all from `california_schools`.
The input has 50 candidate slots: 41 valid slots, nine invalid slots, and 39
distinct valid candidates under the pipeline's deduplication policy. There are
46 query scopes, including six candidates with nested scopes, producing 184
requests across four perspectives.

The starting configuration was `training_configs/verifier_qwen3_8b.json`:
Qwen3-8B, vLLM 0.11.0, temperature zero, thinking disabled, batch size 16,
8,192 completion tokens, and 32,768 total context tokens. Prompt version was
`qrank-svem-semantic-v12-rubric-only`. The cached model revision was
`b968826d9c46dd6066d109eabc6255188de91218`.

The original configuration failed during sampler warmup on the RTX 4090:
`gpu_memory_utilization=0.95` left insufficient memory for a 446 MiB allocation.
The successful run used [pilot_config.json](pilot_config.json), changing only
that setting to `0.91`. Offline Hugging Face loading used the cached revision.
The original training config and verifier source were not edited.
The earlier startup and original-config failure logs are preserved separately.

All 184 requests completed with valid JSON, allowed labels, and nonempty string
reasons. Counts reconcile between raw completions, saved dataset annotations,
and the manifest. Invalid candidate annotations remained separate from semantic
NA. Verification took 54.2 seconds after backend initialization; the interval
from first requested batch to last generated completion was 52.3 seconds.
Model initialization and failed attempts are excluded from those timings.
The 34 existing verifier tests also passed; those tests do not measure model
semantic accuracy.

| Perspective | S | U | C | NA | Total |
| --- | ---: | ---: | ---: | ---: | ---: |
| Output | 43 | 2 | 1 | 0 | 46 |
| Constraints | 46 | 0 | 0 | 0 | 46 |
| Ranking | 25 | 0 | 0 | 21 | 46 |
| Relationships | 43 | 2 | 0 | 1 | 46 |
| Total | 157 | 4 | 1 | 22 | 184 |

## Semantic findings

Thirteen assistant-authored diagnostic expectations were recorded in
[spot_checks.json](spot_checks.json) before inspecting model completions.
Six matched: all six expected-S cases passed, while all seven expected-C cases
returned S. These are deliberately selected cases with repeated variants from
the same questions, not independent human annotations or an accuracy estimate.

All candidate indices below are zero-based. Full SQL and explanations are in
[judgments.md](judgments.md).

1. **Explicit formula mismatch missed — q0, candidate 4.** Evidence defines
   free rate as meal count divided by enrollment. SQL multiplies this by 100.
   Output returns S and explicitly says the formula matches the evidence,
   despite the rubric's specific warning about this exact difference.

2. **Missing requirement missed — q4, all five candidates.** The question asks
   for directly charter-funded schools. Candidates filter charter status and
   opening date but omit direct funding. Every constraint check returns S,
   repeatedly equating `Charter = 1` with direct funding. AST extraction can
   expose present predicates, but a separate check for missing obligations is
   still necessary within the perspective judgment.

3. **Wrong ranking quantity accepted — q7, candidate 0.** SQL ranks total
   `NumTstTakr`, not the count scoring over 1500. Ranking returns S. Output
   instead returns U because the SQL does not account for scores. This both
   misses the ranking error and assigns concern to the wrong perspective;
   the projected phone number itself is appropriate.

4. **Literal values assumed — q9, all five candidates.** Constraint checks
   accept funding strings such as `Direct Charter Funding`,
   `Directly Charter-Funded`, `Charter`, and `Direct Charter`. The local BIRD
   description lists `Directly funded`, `Locally funded`, and
   `Not in CS (California School) funding model`. That description was not in
   the prompts: the loader supplies DDL, not these CSV descriptions. Without
   the value mapping, the model should expose uncertainty rather than certify
   guessed string literals. Once supplied, the mapping supports checking the
   actual values. English similarity does not establish SQL string equality.

5. **Invented structural requirements — q0 candidate 3; q7 candidate 3.**
   Relationship explanations respectively demand a county GROUP BY for a
   scalar MAX already filtered to that county, and an explicit JOIN despite
   a subquery association. Both return U. These are unsupported requirements,
   not identified missing facts; the JOIN demand directly conflicts with the
   rubric's acceptance of equivalent query forms.

6. **Scope attribution leaks — q7 candidate 3; q8 candidate 1.** The output
   check for q7's intermediate `SELECT cds` describes projecting `Phone`,
   which belongs to the root. For q8, root output rejects the outer SUM, while
   inner output praises that same outer SUM. Different block labels may
   legitimately disagree; the problem is that the inner explanation evaluates
   an operation outside its target and contradicts the other assessment of
   that operation.

There are positive results: q5's four distinct candidates correctly receive S
for predicates matching the explicit `Virtual = 'F'` evidence and math-score
threshold. Q6 candidates 0 and 2 correctly receive output S for school names.
The 21 ranking checks on questions without ranking requirements all return NA.
The model also recognizes several intermediate MAX and ID-selection operations
as useful steps; this pilot does not show systematic rejection of helper outputs.

## What benchmark labels do and do not establish

Among 39 distinct valid candidates, 17 have positive execution benchmark labels
and 22 have negative ones. None of the 22 benchmark-negative candidates received
any C signal; 21 received only S/NA and one had U. The sole C is q8 candidate 1,
whose benchmark label is positive.

These are disagreement statistics, not verifier accuracy. A local S never means
whole-candidate correctness. The benchmark labels also reflect execution on one
database instance: q7 candidate 0 is benchmark-positive despite ordering the
wrong semantic quantity. Likewise, q8's SUM can match execution when there is
one maximal school, without establishing the intended behavior for multiple
maximal schools. Do not train block-level confusion matrices directly from these
whole-candidate labels.

## Design implications for the future EM model

Keep deterministic query-scope decomposition, intact Boolean expressions, and
grouped ordering/limit operations. Retaining full SQL as dependency context is
reasonable; it is not itself an overall model verdict. Separate calls, structured
responses, raw traces, and distinct-candidate verification are useful foundations.

Prioritize these changes:

1. Make each response ground its judgment in the relevant question/evidence
   phrase and exact target operation, or explicitly identify a missing operation
   or missing fact. Include absence of required predicates in constraint checks.
   A short structured reason can do this without adding an overall verdict.
   The ratio failure shows that rubric wording alone is not enough; verify any
   proposed change against fixed examples and held-out questions.
2. Define root versus intermediate responsibilities and record semantic scope
   roles/dependencies. Judge an intermediate ID lookup as an ID lookup. Record
   missing obligations once at an appropriate root perspective instead of
   blaming every empty nested clause group. Exclude unreachable unused CTEs
   from performance evidence, or explicitly mark their lack of contribution.
3. Supply relevant BIRD descriptions and value mappings with provenance. Use U
   for unknown mappings rather than assuming synonyms are stored values.
4. Revisit applicability gates. A query can introduce an unjustified LIMIT or
   a multiplicative JOIN even when the question requests no ranking or entity
   combination. The current immediate-NA rules can leave such operations
   without a clear checker. Harmless extra ordering and answer-truncating LIMIT
   need different treatment. These are static design risks, not additional
   failures measured by this ten-question pilot.
5. Preserve S/U/C/NA separately and operational failures outside the semantic
   label space. Treat S as local satisfaction, C as a local contradiction, U as
   insufficient grounding, and NA as absence of an applicable local check.
   Do not turn these into independent votes for overall correctness.

For EM, define the latent target first: local semantic correctness and candidate
correctness are different quantities. Checks share a model/context and often
discuss the same operations; a candidate with more scopes must not receive more
independent evidence merely because of its syntax. Scope IDs are candidate-local
and cannot identify shared annotators or aligned semantic requirements across
candidates. Pooling by perspective and semantic role, with explicit assumptions
about dependencies and the connection to candidate correctness, is more defensible
than treating every block as an exchangeable independent annotator.
Dependency structure affects weak-label inference; see
[Bach et al., ICML 2017](https://proceedings.mlr.press/v70/bach17a.html).

This sample's constant constraint output provides no observed discrimination
between candidates. EM should not be expected to recover useful error detection
from that signal alone. The next experiment should include reviewed semantic
mutations (wrong value, units, aggregate, ranking quantity, join key, and missing
predicate) plus equivalent SQL rewrites, split by question across development and
evaluation. Measure S/C precision, error detection, applicability, and attribution
alongside downstream candidate selection. Ten adjacent questions from one
database cannot establish general BIRD performance.

## Comparison with prompts_example.py

Reviewed the user's previous example after the pilot. It was not used in this
run; its content is preserved under `reference_review/prompts_example.py`.

The example has two stages: extract atomic requirements without candidate SQL,
then check one candidate against the fixed requirements. This complements AST
decomposition: requirements describe intended behavior, while AST blocks describe
implemented behavior. Candidate-independent requirements could keep direct
funding separate from charter status, make count/enrollment scaling explicit,
and hold ranking quantity fixed across candidate variants. This is a design
hypothesis to test, not a measured improvement from this pilot.

Useful pieces to reuse are requirement IDs, checks against explicit obligations,
concrete conflict types, SQL evidence, and an unresolved-reason field for U.
However, do not adopt it unchanged:

- Its C burden is strict while SQL evidence for S is optional. The pilot's main
  issue is unsupported S. Require grounding for S as well: a requirement/source
  reference, the target operation, and why the operation implements that
  requirement. A generated explanation still needs validation; the presence of
  fields is not proof of correctness.
- Extraction can omit or misstate a requirement. Freezing that mistake would
  affect every candidate for the question. Record source spans and uncertainty,
  retain the original question/evidence for auditing, and evaluate extraction
  coverage. The current check_user passes only extracted requirements and schema.
- A missing predicate has no exact fragment to quote. Permit a structured
  missing-behavior record with a scope anchor and a null offending fragment.
  Do not let the exact-fragment requirement turn every omission into U.
- The example checks all requirements in one candidate request and supplies no
  target AST block. Add block IDs, scope roles, and dependency references to use
  it for this project's local signals. Semantic requirement-to-block association
  needs interpretation; parsing alone does not establish that association.
- It defines NA only at question/dimension level. Local non-applicability,
  missing obligations, and extra harmful candidate operations still need
  distinct handling. An empty extracted list is not proof that extraction was
  complete or that an extra LIMIT/JOIN is harmless.
- Its final instruction anticipates deterministic overall aggregation. Omit
  that for the stated goal and retain individual observations for future EM.

A compatible target design is one record per applicable requirement/block
association, containing requirement_id, block_id, perspective, label, source
evidence, SQL evidence or missing-behavior anchor, and a reason. A requirement
may span several dependent blocks; avoid duplicating the same failure in every
scope. Shared requirements need not mean adding an overall model verdict.
If an extraction stage is undesirable, the grounding fields and explicit
missing-behavior checks can first be tested inside the existing perspective
calls. Compare these options empirically rather than assuming two stages will
fix the observed errors.

The example is currently a reference file, not a runnable alternative pipeline:
it imports `.schemas.RequirementSet`, but that module is absent, and its helpers
are not referenced by the current workflow. No example-prompt inference was run.

## Artifacts and reproduction

- [audit.json](audit.json): reconciled counts and all thirteen diagnostic results.
- [cases.jsonl](cases.jsonl): distinct candidates, benchmark labels, and blocks.
- [judgments.md](judgments.md): complete readable model judgments.
- [dataset/semantic_verifier.manifest.json](dataset/semantic_verifier.manifest.json): run identity.
- [dataset/verifier_trace.jsonl](dataset/verifier_trace.jsonl): supplied contexts, prompts, raw completions.
- [original-config-oom.log](original-config-oom.log): original config startup failure.
- [source_snapshot](source_snapshot): verifier source and original config used for review.

The original dataset and production source remain unchanged. Re-run the audit:

```bash
.venv/bin/python outputs/verifier-block-pilot-bird10-20260911/audit.py
```

For a fresh model run, choose an unused output directory:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -u -m qrank.verifier transform-dataset \
  --config outputs/verifier-block-pilot-bird10-20260911/pilot_config.json \
  --input dataset/bird_dev_agr_q_sql_af_AQXRO_ss_m5 \
  --output outputs/verifier-block-pilot-bird10-rerun \
  --limit 10
```
