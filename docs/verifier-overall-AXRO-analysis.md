# AXRO overall-verifier discrimination audit

The completed AXRO run covers 1,534 BIRD-dev questions, with 5,489 retained
candidate rows. All retained candidates have aligned Boolean execution-benchmark
labels and S/U/C/NA verdicts. Eleven questions have no retained candidate.
The saved manifest's verdict counts cover 4,776 distinct question-level judgments;
the table below counts all 5,489 candidate rows, including candidates that share
one judgment because their SQL text is identical within a question.

| Benchmark label | S | U | C | NA | Total |
| --- | ---: | ---: | ---: | ---: | ---: |
| Positive | 2,594 | 9 | 299 | 0 | 2,902 |
| Negative | 1,849 | 22 | 716 | 0 | 2,587 |
| Total | 4,443 | 31 | 1,015 | 0 | 5,489 |

Thus `S` covers 80.9% of candidates and 71.5% of benchmark-negative
candidates. An `S` candidate is benchmark-positive 58.4% of the time, compared
with the pool's 52.9% positive rate. `C` is benchmark-negative 70.5% of the
time, but catches only 27.7% of benchmark-negative candidates. `U` is 0.6% of
candidates; `NA` is absent. Benchmark-negative is **not** a synonym for a
reviewed semantic contradiction: execution labels can miss semantic errors and
can flag other kinds of mismatch. The table does not measure recall against
manually assigned `C` cases.

At the question level, 911 of 1,523 nonempty questions (59.8%) assign `S` to
every candidate. In 233 of these questions, no candidate is benchmark-positive.
Of the 1,024 questions with a benchmark-positive candidate, 932 retain at least
one positive candidate under an S-only filter, 54 assign `S` only to negative
candidates, and 38 assign no `S`. Only 132 questions have exactly one `S`, and
that candidate is benchmark-positive in 31. These counts show why `S` alone
offers little selection information even when it preserves many positives.

The smaller completed AXR run shows the same pattern: 3,285/4,054 (81.0%)
candidate rows are `S`, including 1,314/1,854 (70.9%) benchmark-negative
rows. These runs overlap heavily and are not independent replications.

The reviewed diagnostic cases stored in
`outputs/verifier-block-pilot-bird10-20260911/spot_checks.json` were authored
for a different, older block verifier. They are selected checks, not a manual
reference set for the current overall verifier. In the current AXRO output,
question 4's four candidates omit the direct-funding condition, yet all four
receive `S`. Question 0's percentage-scale error receives `C`, but the reason
incorrectly claims that its SQL lacks a `MAX` operation. This means verdict
counts alone also miss explanation quality.

## Next evaluation

Use a frozen, question-disjoint set of human-reviewed *distinct candidate SQL*
with explicit S/U/C/NA labels and a short cited contradiction for each `C`.
Report `C` recall on reviewed C cases, false-S rate on reviewed C cases, C
precision, S precision, abstention rate, and results by error type. Review cases
should include missing predicates, wrong literal or unit, wrong aggregation or
ranking target, join and row-grain errors, and equivalent rewrites. Count shared
SQL once for semantic recall; separately report candidate-weighted results for
downstream selection. Track whether a `C` reason identifies the actual error.

The next verifier experiment should make each decision check explicit task
requirements against the candidate's actual operations, including absent
requirements. Compare it with the current overall prompt on the frozen cases
before a full run. A useful change must improve reviewed contradiction recall
and reduce false `S` while keeping C precision, S precision, and abstention
visible; changing only the marginal S/C balance does not establish improvement.

Reproduce the saved-data counts with:

```bash
uv run --no-sync python scripts/analyze_overall_verifier.py \
  dataset/processed/bird_dev_overall_verifier_AXRO
```
