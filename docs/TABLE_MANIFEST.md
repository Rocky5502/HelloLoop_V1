# Table Manifest

All source tables are CSV and use SHA-256 checksums. `scripts/reproduce_analysis.py` independently
generates `policy_resolution.csv` and validates the headline counts, rates, signals, and cost totals.

| File | Source/input | Purpose | SHA-256 |
|---|---|---|---|
| `cost_decomposition.csv` | Redacted run table | Cost accounting by policy | `764a3d847cc59e725678c03838b3f9edbd7867aa4865dd62757ae36b9ba1d462` |
| `denominator_sensitivity.csv` | Redacted run table | Registered/evaluated denominator comparison | `45284af29de369976e0b86268c49f83b4fbbbf1aeaa124d875c99ca3836423c8` |
| `failure_breakdown.csv` | Redacted run table | Terminal classifications | `546123298b8704901f527b66a12058aeaf99c04edc004ee56530baf5a9ef33d3` |
| `intervention_diagnostics.csv` | Redacted run table | Descriptive controller actions | `08c471dd72a6f84c859bdffeaa2217c8667f97ed91fb71e394e4983ea0c85631` |
| `replication_transition_table.csv` | Redacted run table | Task-level replication transitions | `20ef6bf923e617143ed744925a077b1cf15d50d0834151e1dfbec8f6dfc2b741` |
| `signal_risk_coverage.csv` | Grouped predictions | Selective prediction diagnostic | `a3e0a18e22e2618e4434684f0a338f5356684177a6fcf00e6c64da65671a8214` |
| `signal_validity_metrics.csv` | Grouped predictions | Frozen and post-hoc signal metrics | `96801a466eb306c014ad9295b76e2d3cd38eb7f18390c2a5ceac3d628fe09437` |
| `threshold_search.csv` | Fixed trajectories | Post-hoc threshold sensitivity | `5d07dcf3974be3e2316c30609651149934f538b0199db6d346b6e78395518a22` |
| `uncertainty_definitions.csv` | Controller specification | Proxy definitions | `73a480964a7a9d1cb410bf4909ae1b310d595c497d09552688d2e7e47c46312e` |
