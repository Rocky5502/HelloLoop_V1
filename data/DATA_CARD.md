# HelloLoop Public Data Card

## Purpose And Unit

This release supports audit and reproduction of controlled LLM-agent policy comparisons. The unit
of observation in `data/processed/run_level.csv` is one registered terminal run: a benchmark task,
loop policy, repetition, seed, and frozen model configuration.

## Scope And Counts

The data cover a 36-run Lite development pilot, 160-run Lite main comparison, 40-run Verified
holdout, and 80-run same-task Lite replication: 316 registered rows in total. Of these, 291 reached
official evaluation, 289 submitted patches, 2 were evaluated abstentions, and 141 resolved. The
remaining terminal classes include 10 agent failures, 11 infrastructure failures, and 4 budget
stops. Registered-denominator metrics retain every row; evaluated-only metrics condition on
`evaluator_reached=1`.

## Included Data

- Redacted run identifiers, terminal classes, official binary outcomes, resource use, and costs.
- Eight stage-specific uncertainty proxies, missingness flags, and aggregate controller decisions.
- Grouped cross-validated post-hoc signal predictions.
- Identifier-only task manifests, analysis-ready tables, figure sources, and checksums.

## Missingness And Redaction

Blank numeric fields mean the quantity was unavailable or not applicable. `missing_*` indicators
record absent uncertainty evidence. Redaction removed raw prompts, patches, free-form trajectories,
authorization material, provider request identifiers, account metadata, personal information,
private endpoints, host paths, and machine metadata. The release contains aggregate controller
action counts but no raw trajectory text. No scientific outcome or uncertainty value was changed.

## Upstream Sources And Licensing

Task identifiers refer to SWE-bench Lite and Verified. Benchmark payloads, hidden tests, gold
patches, repositories, and evaluator images are not redistributed; users obtain them from upstream
under the applicable terms. Original HelloLoop code is MIT licensed. This data card does not
relicense third-party identifiers or facts. See `THIRD_PARTY_NOTICES.md`.

## Limitations

One model snapshot and provider route were studied. Provider/evaluator drift and benchmark
contamination are possible. Uncertainty fields are engineered proxies, not calibrated
probabilities. Post-hoc predictions did not control the original trajectories. Failures and
abstentions must not be silently dropped or reclassified.

## Appropriate Uses

Reproduce aggregate results, audit denominators and missingness, compare controller behavior,
evaluate alternative statistical summaries, and prototype controllers against synthetic fixtures.

## Inappropriate Uses

Do not infer autonomous safety, causal effects of interventions, statistical equivalence, or
general performance beyond the studied setting. Do not join task identifiers to private data,
attempt to reconstruct removed trajectory content, or treat post-hoc models as deployed policies.
