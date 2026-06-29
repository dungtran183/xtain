# CrossTaint

CrossTaint is an open-source framework for probabilistically bounded multi-hop cross-chain taint tracking in bridge-exploit forensics. The implementation contains the bridge intermediate representation, value-conserving propagation engine, bridge-event matcher, heterogeneous pseudonym resolver, cold-start fallback, synthetic trajectory generator, and evaluation utilities described in the accompanying anonymous artifact.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Runtime configuration is stored in `config/`. RPC endpoints and API keys are read from environment variables referenced by the YAML files. Set `CROSSTAINT_CONFIG_DIR` to use an external configuration directory.

## Minimal Use

```python
from crosstaint.pipeline import InferenceStack

stack = InferenceStack.from_config(
    matcher_checkpoint=matcher_checkpoint,
    resolver_checkpoint=resolver_checkpoint,
    event_index=event_index,
    nodes=nodes,
    edges=edges,
)

result = stack.propagate(
    origin=origin_node_id,
    origin_chain=origin_chain,
    origin_value=origin_value,
    graph=nodes,
    edges=edges,
    case_id=case_id,
)
```

`result.suspect_set` contains ranked suspect addresses and `result.soundness_certificate` records the per-case bounded-mismatch certificate metadata.

## Artifact Scope

This repository intentionally ships source code and public configuration only. Historical exploit manifests, benign-user background pools, checkpoints, indexed graph snapshots, local experiment outputs, manuscripts, and bibliography files are not part of this artifact.

## License

Apache License 2.0. See `LICENSE`.
