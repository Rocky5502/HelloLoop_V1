# Data Dictionary

## `data/processed/run_level.csv`

Each row is a preregistered or appended terminal run. Core identifiers are `experiment`, `run_id`,
`benchmark`, `task_id`, `loop_variant`, `repetition_id`, and `seed`. Outcome fields are
`registered_success`, `evaluator_reached`, `patch_submitted`, `abstention`,
`official_evaluation_status`, `official_resolved`, and `terminal_classification`.

Cost fields distinguish conservative `budget_cost_usd`, `reserved_pre_model_usd`,
`measured_full_rate_usd`, and runtime-reported `runtime_cost_usd`. Resource fields contain requests,
tokens, steps, tool calls, duration, inspections, test executions, revisions, and verification
actions. `u_*` fields are the eight proxy values plus their composite. `missing_*` fields indicate
that corresponding evidence was unavailable. `controller_decisions` is an aggregate action count,
not a raw trajectory.

Boolean CSV values use `0` and `1`; blank means not observed or not applicable. A failed or stopped
registered row remains in registered-denominator analyses and is not recoded as evaluator-reached.

## `results/processed/grouped_cv_predictions.csv`

Each evaluated feature row contains task/run identifiers, registered outcome, the frozen
unweighted signal, component baselines, and grouped cross-validated post-hoc predictions. These
predictions were derived from fixed trajectories and did not control the original runs.

## Identifier manifests

Files in `data/manifests/` list benchmark scope and task identifiers only. They intentionally omit
problem statements, hints, tests, patches, repository snapshots, and other upstream payloads.
