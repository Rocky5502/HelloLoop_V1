# Figure Manifest

All checksums are SHA-256. PNG metadata was inspected before release. The framework is rendered
from the final supplied vector source. Frozen audit displays were generated from the same redacted
run table and processed signal sources shipped here; the public reproduction script regenerates
the headline policy and cost displays in `results/generated/figures/`.

| File | Source | Input | Purpose | Use | SHA-256 |
|---|---|---|---|---|---|
| `docs/figures/helloloop_framework.png` | Final framework PDF render | Vector framework source | System architecture | Project overview | `795ee6d11b0d6fc5b47842d8f2123ebb757ef017b241c3a0697f3e021a8524d8` |
| `results/figures/fig_cost_decomposition.png` | Frozen audit pipeline | `run_level.csv` | Measured and reserved cost components | Public artifact | `fa688ff6f9e7b56ef0f091e6729a28f2b2315f986a28b0df77a9d254d5924951` |
| `results/figures/fig_cost_transition.png` | Frozen audit pipeline | `run_level.csv` | Cost transition diagnostic | Public artifact | `fedbdb34b719d09973c7e2f0a12a82f91cc3b2a29636ce2d2d30ae870ffd5a47` |
| `results/figures/fig_failure_breakdown.png` | Frozen audit pipeline | `run_level.csv` | Terminal failure classes | Public artifact | `94a026571e5e75c3568a65602df91334bbcb5dd8afc9145f44233abcd70575ea` |
| `results/figures/fig_intervention_cost_outcome.png` | Frozen audit pipeline | `run_level.csv` | Descriptive intervention outcomes | Public artifact | `6581a32fd6e22a6005fc898e7b33c883d699bd6dd2f891c3f78b96a419ff24d6` |
| `results/figures/fig_replication_stability.png` | Frozen audit pipeline | `run_level.csv` | Same-task replication transitions | Public artifact | `dc39a03c0257d85831efbf187033aeb1067cbb1e5340fbb6d39ffffc55ed663e` |
| `results/figures/fig_signal_reliability.png` | Frozen audit pipeline | grouped predictions | Reliability curve | Public artifact | `5348ef0d059c353559e1fe7e6a3527e37c70ce253da9482d42e5847b6edeaa32` |
| `results/figures/fig_signal_validity.png` | Frozen audit pipeline | grouped predictions | Signal discrimination | Public artifact | `0b58e4d45bd0673225170a6e88fd478ae57b21ab26c3330c52a770d50d9a6edd` |
