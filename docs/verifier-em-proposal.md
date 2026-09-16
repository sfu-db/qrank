# Verifier signals for estimating semantic candidate correctness

Proposal, 2026-09-11. This document records the design and experiment plan;
implementation status is noted below. It does not establish a measured improvement.

Implementation update: the v13 verifier now extracts grounded requirements,
retains local block checks, adds individual requirement-coverage checks, validates
evidence references, and exports the structured observations. See
[grounded block signals](verifier-block-signals.md) for the implemented contract.
The EM estimator, independent contract audit/secondary checker, semantic
alignment across candidates, and stratified evaluation below remain proposed.

Estimate semantic correctness given the question and supplied evidence/context,
**conditional on the candidate already being valid**. Exclude candidates marked
invalid from training and inference; keep them only in collection accounting.
Validity is an existing upstream decision, not a semantic verifier observation.
All remaining candidates may be wrong, and several may be correct. Candidate
probabilities must not be normalized to sum to one within a question.

The recommendation is to retain local AST-block judgments, ground them in
explicit task requirements, add checks for missing behavior, and fit a small
candidate-level label model with EM. Every LLM call judges a part or a specific
semantic obligation, never overall candidate correctness. Keep the verifier's
individual S/U/C/NA records. S and C on the same candidate are expected weak
signals; neither is a hard candidate verdict. The downstream statistical model
estimates correctness. EM is the fitting algorithm; defining its latent target
and observation model is the substantive modeling decision.

**1. What the current evidence establishes**

The current [workflow](../src/qrank/verifier/workflow.py) makes one request per
SQL scope and perspective. It includes full candidate context, retains empty
clause groups, and stores individual observations. Requirements are inferred
inside each request; responses contain only a label and reason. The current
dataset adapter already verifies distinct SQL strings within execution-agreement
clusters. The `src/qrank/svem/__init__.py` file is empty; no EM implementation was
found there.

The [ten-question pilot](../outputs/verifier-block-pilot-bird10-20260911/report.md)
contains 39 distinct valid candidates, 46 SQL scopes, and 184 observations:

- All 46 constraint observations are S.
- All seven assistant-authored expected-C diagnostic cases returned S. These are
  selected, repeated cases, not independent human labels or an accuracy estimate.
- Six questions have identical root-label vectors across their valid candidates,
  including one question with only one distinct valid candidate. This count was
  recomputed from the saved `cases.jsonl`.
- Missing direct funding, incorrect formula scaling, wrong ranking quantity,
  unsupported literal mappings, and scope attribution need explicit evaluation.

An EM model using identical label vectors and the same prior cannot distinguish
those candidates. A constant constraint channel supplies no within-question
discrimination in this pilot. Better fitting cannot recover a distinction absent
from the observations. The saved benchmark labels concern execution and must
not be relabeled as semantic or requirement-level truth.

The older [reliability review](verifier-reliability-review.md) describes an earlier
implementation and recommends deterministic verdict aggregation. This proposal
uses the current block architecture and retains the user's no-verifier-verdict
constraint; that older aggregation recommendation does not apply here.

**2. Define the semantic item before defining the signal**

Use `(question, valid candidate, semantic check, target blocks)` as the
observation unit. A semantic check names a specific local responsibility or a
missing obligation. Requirements establish what that responsibility is. The
four perspectives categorize checks; they are not interchangeable judgments of
whole-candidate correctness. A correct projection and an incorrect filter can
legitimately produce S and C without disagreement. Their different statistical
relationships to candidate correctness are what EM must learn.

Keep these observation targets distinct:

| Target kind | Meaning | Initial EM use |
| --- | --- | --- |
| Local contribution | Whether a targeted block fulfills its specific semantic responsibility | Primary input, with type, role, and dependency metadata |
| Requirement coverage | Whether a specific obligation has an implementation, including omitted behavior | Additional weak signal with overlap recorded |
| Contract coverage | Whether extraction represents the question faithfully and completely | Audit and calibration metadata |
| Candidate-set comparison | A difference between candidates on a fixed requirement | Later ablation with explicit dependence |

A requirement can need several blocks jointly. A block can support several
requirements. Local S means the checked contribution is satisfied; it does not
claim complete requirement or candidate satisfaction. Neither SQL scope IDs nor question-specific requirement IDs are
reusable annotator identities. Pool reliability across recurring semantic types
such as literal comparison, population coverage, formula units, ranking quantity,
and entity association, initially using the four broad perspectives where data
is sparse.

**3. Collect observations in a grounded workflow**

1. **Build one versioned context and contract per question.** Extract requirements
   from the question, evidence, schema, and relevant documented column/value
   meanings, without candidate SQL, gold SQL, benchmark labels, or vote counts.
   Store source references, original text, interpretation, semantic type, and
   unresolved alternatives. Cache by content and extractor/model versions, not
   question ID alone. Preserve the original context in subsequent checking calls.
   The current automatic database loader supplies DDL; loading documented value
   descriptions with provenance is a proposed addition. Do not treat sampled
   values as exhaustive domains.
2. **Audit extraction independently of candidates.** Check that each meaningful
   question/evidence phrase is represented or explicitly unresolved. An audit
   from another LLM is noisy evidence, not certification. A material change
   creates a new contract version and requires rechecking all affected candidates.
   Ambiguity must not disappear merely because a candidate suggests an answer.
3. **Map requirements to the candidate's AST and dependencies.** Reuse scope,
   source, predicate, grouping, ordering, and set-operation extraction. Record
   contributing blocks and their roles. Mapping failure is uncertainty, not
   evidence that the SQL omitted behavior. Keep full SQL available for an
   unmapped requirement and for dependency resolution.
4. **Check a part against its assigned responsibility.** Retain block targets
   and narrow rubric responsibilities. Supply the relevant fixed requirement,
   target operation, and dependency context. An intermediate ID lookup is checked
   as an ID lookup, not as the final requested projection. Require grounded
   evidence for both S and C. Where an obligation inherently spans blocks, name
   that bounded group and its role explicitly. Request boundaries do not
   establish independence. No call emits a whole-candidate verdict.
5. **Check coverage in both directions.** Every extracted requirement needs an
   observation even when no implementing block is found. Every reachable,
   potentially result-changing operation needs a justified relationship to the
   contract. An unwanted LIMIT can violate the requirement to return all matching
   entities even when ranking is NA. Extra DISTINCT, joins, and grouping likewise
   need assessment of their actual effect. Harmless ordering is not automatically
   a contradiction. Unused CTEs must not add correctness evidence.
6. **Validate and retain records.** Check IDs, required evidence fields, source
   references, raw SQL offsets, and dependency references. SQLGlot renders
   normalized SQL, so an AST-rendered string is not necessarily an exact raw SQL
   substring: distinguish the two explicitly. Structural validation establishes
   a well-formed record, not a true semantic explanation. Preserve raw responses
   and operational failures separately; a parser failure is neither U nor NA.

For omitted behavior, store an expected operation, a scope anchor, the inspected
dependency scopes, and a null offending fragment. C requires an established
missing semantic effect, including consideration of equivalent implicit or
nested implementations. Absence of a WHERE clause alone is insufficient. Record
an omission once for its requirement, rather than once per empty scope.

| Label | Required interpretation and grounding |
| --- | --- |
| S | Evidence supports satisfaction of the targeted local responsibility; cite task evidence, SQL behavior, and relevant dependencies |
| C | Evidence establishes a conflict or omitted required behavior that can change the answer under the stated assumptions |
| U | The requirement matters, but a named missing fact, ambiguous interpretation, or unresolved dependency prevents S/C |
| NA | The particular check does not apply; state the applicability reason |

Do not invent a tie policy or a stored string value to avoid U. A documented
direct-funding restriction can be missing even when the correct stored literal
is unknown; conversely, a plausible but undocumented literal should not receive
S solely because its English wording looks right.

For an applicable assigned responsibility, the semantic checker normally emits
S/U/C. A local block can be NA for a responsibility that does not belong there;
a specialized arithmetic checker can emit NA for a relationship item.
Dimensions with no requirements retain explicit applicability records. An empty
extracted list is not evidence of complete extraction. Preserve U and NA in the
dataset even when the initial label model masks verified NA observations.

**4. Make evidence sources identifiable and auditable**

Start by measuring these source families on overlapping local check items:

| Source | Contribution | Limitation |
| --- | --- | --- |
| Grounded semantic checker | Explains whether the targeted part fulfills its assigned responsibility | May rationalize incorrect SQL |
| Blinded counterexample checker | Searches for an allowed case where the requirement fails, without seeing the first judgment | Failure to find a counterexample gives U, not S; same-model errors may remain correlated |
| Narrow deterministic checks | Check established literal/operator/units facts in explicitly supported SQL forms | Unsupported forms abstain; AST resemblance is not semantic equivalence |

An independently configured/modelled checker is an experimental source, not a
guarantee of independence. A deterministic check relying on an LLM-extracted
mapping shares that mapping's uncertainty. Do not count evidence field presence,
the label, and its explanation as three separate measurements.

Record source/model/prompt versions, shared call ID, context and contract
versions, target role, and dependency groups. Use a prespecified overlap sample
for secondary checking, including S cases. Reviewing only C/U cases would hide
false satisfaction and produce biased reliability estimates. Log review selection
and keep adaptively selected reviews outside the initial unsupervised likelihood
unless their selection mechanism is modeled.

Keep candidate checks independent of candidate ordering and popularity in the
first experiment. Use the candidate set to find informative disagreements for
offline annotation, and preserve generation multiplicity/agreement metadata for
later ablations. Identical SQL shares one semantic assessment. Equal results on
one database do not justify copying semantic labels across different SQL.

**5. Proposed EM model**

Let `Y_qc` be semantic correctness of valid candidate `c` for question `q`.
Let `L_qck` be an observed S/U/C/NA label for local check `k`. Its reusable type
includes semantic responsibility, structural role, and source version. These
labels are observations, not candidate truth labels.

Begin with a candidate-conditioned categorical label model:

```text
P(Y, L | context, contract, applicability, valid=1)
  = P_eta(Y | context, contract)
    product_over_evidence_groups P_theta(L_group | Y, group_type, applicability)
```

The first diagnostic baseline uses one emission per distinct local check and
assumes conditional independence. For type `t`, learn the two categorical rows
`P(L | Y=1,t)` and `P(L | Y=0,t)`. An incorrect candidate can have many locally
correct parts, so `P(S | Y=0,t)` can be high. A correct candidate can receive a
noisy C, so `P(C | Y=1,t)` is not forced to zero. These are predictive emission
probabilities, not local-checker confusion matrices: measuring a local checker's
accuracy separately requires local semantic annotations.

For independent retained emissions, posterior odds have an interpretable form:

```text
posterior odds(correct)
  = prior odds(correct)
    * product_k P(observed label_k | correct, type_k)
                / P(observed label_k | incorrect, type_k)
```

Thus S and C combine through learned likelihood ratios. A common projection S
may contribute little; a reliably diagnostic missing-filter C may contribute
much more. Neither label is a hard rule, and neither requires an LLM to judge
the whole candidate. All S need not imply a probability near one when the same
pattern frequently occurs on incorrect candidates.

Keep all four categorical outputs in storage. In the initial model, mask
established structural NA using applicability metadata and model S/U/C for
applicable checks; never turn NA into S. An unexpected NA from an assigned
applicable check remains a recorded applicability error and can be modeled as a
fourth emission category in an ablation. Operationally absent observations are
missing data, not any of these labels. U can be informative statistically, but
is not a half-correct numeric value or a third latent correctness class.

The E-step estimates each candidate's posterior correctness, fixing reviewed
semantic anchors. The M-step updates type-pooled emission probabilities and the
prior from expected counts, with smoothing and regularization. Start with a
constant prior and broad types, adding context-conditioned parameters only when
data supports them. Use several
initializations and check stability. EM-based crowd-label estimators optimize a
nonconvex likelihood; initialization is a substantive issue in the
[Zhang et al. study](https://www.jmlr.org/papers/v17/14-511.html).

**Use a small reviewed anchor set and unlabeled data together.** One unknown
weak channel cannot generally identify its relationship to correctness and
correctness prevalence from its marginal output alone. Multiple channels need
identifying assumptions and sufficient variation; many prompts from the same
model are not automatically sufficient. Candidate-level anchors should be
independently reviewed semantic labels, not whole-candidate LLM verdicts. Local
annotations assess signal quality separately. Anchors orient the latent classes
and constrain reliability estimates. Hold calibration and evaluation questions
separate from training anchors. The recommendation is semi-supervised EM; fully
unsupervised fitting is an ablation whose assumptions must be stated.

Conditional independence is an approximation, particularly because partial
errors are correlated within incorrect candidates. Before trusting probabilities,
group repeated judgments of the same fact or use pooled joint emissions/shared
factors for a small prespecified set of dependencies. Start by deduplicating
exact repeated observations and retaining one primary check for an equivalent
semantic role; keep other records for audits and ablations. Distinct useful
checks remain separate even if they disagree. Contract mistakes shared across
candidates and checks sharing a population, join, or formula also require
evaluation. Dependency structure affects weak-label inference, as studied by
[Bach et al.](https://proceedings.mlr.press/v70/bach17a.html); the proposed dependency
groups here are engineering hypotheses requiring held-out validation.

Emit checks for the same question-level obligations across candidates, including
omissions, so absence of implementing syntax cannot remove an inconvenient
signal. Do not reward extra helper scopes as extra independent successful work.
Evaluate invariance to equivalent rewrites. Ambiguous context and uncertain
contract coverage should be reported alongside probability estimates; calibrate
these cases separately if sufficient reviewed data exists.

If the direct label model fails because different error families remain strongly
dependent, extend it with latent satisfaction variables for individual semantic
responsibilities or latent error types. That is a second-stage hierarchical
model, not a requirement to replace local weak signals with hard judgments.
Export `P(Y_qc=1 | observations, context, valid=1)`, coverage metadata, and model
versions from the downstream estimator. Keep original verifier observations
unchanged and do not compute a deterministic overall S/U/C verdict.

**6. Experiment before large-scale collection**

Use a staged pilot of approximately 150 new questions across multiple databases:
50 development/training-anchor, 50 calibration, and 50 locked evaluation
questions. This is an initial budget proposal, not a power calculation. Include
all distinct valid candidates and assess whether there are enough reviewed
failures per semantic type; expand if uncertainty remains wide. The existing ten
questions are development regressions only. Keep all candidates, rewrites, and
mutations of a question in the same split. Add a separate database-held-out test
when coverage permits.

Review question contracts before seeing candidate judgments. Annotate local
satisfaction and candidate semantic correctness separately; do not derive one
ground-truth label mechanically from the other. Adjudicate disagreements and
record unresolved cases rather than forcing binary labels. Include a random
sample of natural candidates as well as targeted failures. Mutations enrich
diagnostics but have artificial error prevalence, so natural held-out candidates
must drive probability calibration and deployment-performance claims.

| Diagnostic family | Expected sensitivity or invariance |
| --- | --- |
| Omitted predicate; extra restrictive filter | Detect required population loss or expansion |
| Wrong literal, attribute, operator, date boundary | Detect a grounded constraint conflict; use U when mapping is unknown |
| Wrong aggregate, denominator, units/scaling | Attribute failure to the relevant formula/output requirement |
| Wrong ranking quantity/direction/k | Check the full quantity and selection behavior |
| Wrong join key, multiplicity, grouping grain | Detect changed entity association or result grain |
| Unrequested LIMIT/DISTINCT; unused CTE | Detect harmful effects; avoid rewards or penalties for irrelevant structure |
| Equivalent subquery/join/CTE rewrites | Preserve requirement judgments and candidate estimates |
| Missing versus supplied value descriptions | Resolve the specific unknown without inventing new obligations |
| Same omission in every candidate | Detect the error despite candidate agreement |

Review semantic mutations under the applicable keys, NULL behavior, duplicates,
and dialect; syntactic change alone does not establish changed semantics.
Optional counterexample databases should satisfy the supplied constraints and
have independently reviewed expected outcomes. A passing finite execution test
is supporting evidence, not proof of semantic equivalence. Distilled test-suite
evaluation is relevant methodology for separating execution on one instance from
semantic evaluation:
[Zhong et al.](https://aclanthology.org/2020.emnlp-main.29/). Gold SQL and expected
outputs used for offline evaluation must remain outside verifier prompts.

Compare incrementally on the same questions:

1. Current v12 block signals with a simple calibrated selection baseline.
2. Shared requirements and structured evidence using the same model/context.
3. Explicit omission and operation-coverage checks.
4. Added documented context, with provenance, as a separate ablation.
5. Additional sources at matched cost, with dependence and review selection logged.
6. Anchored EM versus simple count/rule baselines and a regularized supervised
   calibrator using the same reviewed training-label budget. Include a naive
   four-root-signal model as a diagnostic baseline and add dependency modeling
   incrementally.

Measure extraction coverage and invention rate; S/C precision and error
detection by type; U/NA correctness; attribution; decided-case error versus
coverage; within-question discrimination; rewrite stability; and latency/tokens.
For the candidate estimator, measure Brier score, log loss, calibration,
top-ranked semantic correctness, and accuracy versus abstention coverage.
Report performance where all candidates are wrong as well as where a correct
candidate exists. Bootstrap intervals by question, not by individual correlated
candidate. Report execution-benchmark agreement separately from semantic labels.

Proceed to scale only when grounded checking improves held-out semantic
discrimination without obtaining the gain solely through abstention, and the
candidate model improves selection/calibration over the simpler equal-budget
baselines. Fix decision thresholds on development/calibration data. Check that
adding a harmless CTE, duplicating a candidate, or duplicating a recorded
observation cannot manufacture confidence. A more balanced label histogram is
not a success criterion. If EM is unstable or adds no value, retain the useful
signals and use the simpler calibrated estimator.

**7. Concrete implementation sequence**

1. Add versioned requirement, alignment, observation, applicability, and source
   records in `models.py`; add source-grounded context loading and an extraction
   cache separate from candidate checks. Implement the pilot annotation format.
2. Add candidate-independent extraction and grounded block-targeted prompts,
   evidence validation in `responses.py`, and omission/dependency handling in
   `workflow.py`. Keep block-contribution records distinguishable from obligation
   coverage checks; extend dataset schemas without flattening them.
3. Replay the known regressions and run the stratified signal pilot. Freeze
   source identities and measure whether signals merit further collection.
4. Implement the anchored categorical EM baseline under `src/qrank/svem/`, with
   validity filtering, question-grouped splits, pooled emissions, multiple EM
   initializations, and separate calibration/evaluation artifacts.

For Q questions, the primary cost is Q extraction calls plus the assigned local
checks and omission checks over distinct valid candidates, before contract
audits and sampled secondary checks. Compare that measured cost with today's
four calls per AST scope. The immediate next
deliverable should be reviewed contracts and a replayable signal experiment;
fitting EM to the current near-constant signals would not validate this design.
