# Grounded AST block weak signals

SQL verification first extracts atomic requirements from the question, evidence,
and schema/context without candidate SQL. The contract retains exact source
quotes, requirement IDs, ambiguity, unresolved content, and original source
inputs. It is a model hypothesis, not ground truth. Extraction runs once per
question/context in a batch; a supplied SQLiteCache reuses it across calls.

Each candidate then produces two separate collections:

- `block_results`: one local contribution judgment per reachable SQL scope and
  selected perspective (output, constraints, ranking, relationships).
- `requirement_results`: one coverage judgment per requirement in the selected
  perspectives. Each checks that one obligation across the relevant scopes,
  including when the SQL has no implementing operation.

No request asks the model to judge overall candidate correctness. A projection
S and a filter C on the same candidate are legitimate weak signals. The legacy
`labels` and `reasons` dictionaries expose root block contributions only, never
an aggregation of block or coverage observations.

A block is a perspective's clause group in a scope. Boolean expressions and
ordering/limit combinations remain intact. Full SQL, the AST, original context,
and the fixed contract accompany each target. Intermediate outputs are checked
as contributions rather than as final projections. Static dependency traversal
excludes unreachable unused CTEs from observations; their AST remains visible.

```python
result = verifier.verify(question, sql, context=ddl, evidence=evidence)
print(result.requirements['requirements'])
for signal in result.block_results + result.requirement_results:
    print(signal['observation_kind'], signal['block_id'],
          signal['requirement_ids'], signal['label'], signal['reason'])
```

For SQL, each signal retains:

- The target scope, structural role, selected sections, and dependency scope IDs.
- Requirement IDs, original source quotes, inspected scopes, and SQL evidence.
- Label, reason, conflict type, missing behavior, unresolved reason, or
  applicability reason, as appropriate.
- Contract key, source ID, call/cache key, and dependency group IDs shared by
  observations referring to the same requirement in the same candidate.

S and C both require source grounding. S requires SQL evidence; a local S/C must
cite an operation in its target scope and selected sections. SQL evidence names
an AST section and zero-based operation index plus a fragment copied from raw
SQL. The parser validates source quotes, raw fragment membership, operation and
scope references, and label-specific fields. This validates structural grounding;
it does not prove that the quoted fragment implements the claimed semantics or
that its asserted association with the AST operation is correct.

A coverage C for `missing_required_behavior` permits empty SQL evidence. It must
name the omitted behavior and report inspection of every reachable scope. This
is an attestation by the checker, not a deterministic proof of semantic absence.
The prompt requires considering implicit and nested implementations. Mapping
uncertainty should yield U rather than an invented omission. Missing behavior
is recorded once per requirement, not once per empty block.

U requires an unresolved reason. NA requires an applicability reason and is
available for local checks; reachable root output always applies. Coverage checks
use S/U/C because their assigned requirement applies. Unrequested ranking does
not exempt a harmful LIMIT from checking against output coverage. Likewise,
extra relationships can affect requested grain even without a requested JOIN.

The API and JSON records retain structured contracts and evidence. Hugging Face
annotations store `requirements` and each signal's `sections` as JSON strings;
source references, SQL evidence, and ID lists remain structured. Invalid dataset
candidates have no observations or contract. Manifests count root, block, and
requirement labels separately, including the number of distinct contracts.

For EM, retain these individual categorical observations. Source IDs identify
checker configurations; scope IDs are candidate-local. Requirement dependency
groups expose known overlap but are not a complete dependence model. Separate
calls are not statistically independent. Candidate probabilities, cross-candidate
semantic alignment, and EM fitting are downstream work, not verifier outputs.

Default SQL cost is one extraction per question/context plus four calls per
reachable scope and one coverage call per requirement. Full context and structured
evidence increase prompt/completion size; measure cost before scaling. A
single-perspective run still extracts the full contract so later checks reuse it.

PrepBench retains its code-level perspective workflow (`code:*` IDs), with empty
requirements and coverage collections. SQL-specific grounding does not apply to
Python. Prompt version `qrank-svem-semantic-v13-grounded-blocks` isolates prior
judgment caches; requirement extraction has its own version and content key.
Errors propagate without retries or semantic fallback labels. The parser never
turns a malformed response into U or NA.
