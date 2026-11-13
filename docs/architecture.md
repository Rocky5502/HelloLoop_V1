# Architecture

HelloLoop wraps a base coding agent with a policy layer and a structured event schema. A run moves
through understanding, planning, retrieval/inspection, execution, verification, and termination.
The trajectory state contains only agent-visible evidence. The benchmark evaluator remains outside
that loop and supplies the official outcome after a patch or abstention is submitted.

The `uncertainty` package computes eight deterministic, engineering-chosen proxies. The `controller`
package converts those values into policy decisions. L0 follows the base linear path; L1 adds
reflection; L2 applies fixed verification; and L3 selects adaptive actions from threshold rules.
Adapters isolate base-agent behavior, while the experiment registry fixes task, policy, seed, model
configuration, and terminal classification. Shards emit isolated records that are merged
deterministically before analysis.

Held-out outcome fields must never flow into controller inputs. Schema validators redact
credential-shaped values and make missing evidence explicit instead of silently imputing it.
