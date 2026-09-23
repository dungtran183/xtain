# CrossTaint

CrossTaint is an open-source framework for probabilistically bounded
multi-hop cross-chain taint tracking in bridge-exploit forensics. The source
tree implements the bridge intermediate representation, value-conserving
propagation, bridge-event matching, heterogeneous pseudonym resolution,
cold-start handling, synthetic trajectory generation, and benchmark tooling.

## Release identity

This repository contains the derived CrossTaint v1.1.0 artifact-hardening
release, published on 1 August 2026.

- Repository: <https://github.com/dungtran183/xtain>
- v1.1.0 tag: <https://github.com/dungtran183/xtain/tree/v1.1.0>
- Source license: Apache-2.0 (`LICENSE`)
- Upstream provenance and modification notice: `NOTICE`
- Machine-readable release inventory: `artifact/manifest.json`

The offline validator checks that these identifiers agree with the package
metadata and that every declared release file is present:

```bash
python scripts/validate_artifact.py
```

This command uses only the Python standard library and does not access the
network, download a dataset, or execute a model.

## Installation

CrossTaint requires Python 3.11 or newer. A clean editable installation is:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e.
```

The default configuration is installed inside the Python package. Resolution
order is: `CROSSTAINT_CONFIG_DIR`, `./config`, a source-checkout `config/`
directory, then the packaged defaults. RPC URLs in `config/chains.yaml` are
environment-variable placeholders and are not needed for the offline smoke
paths below.

## Fast deterministic checks

The lightweight validation suite checks release metadata, source/default
configuration equivalence, the benchmark loader, and fixed-weight propagation
over a two-hop synthetic IR fixture. It needs PyYAML; the optional
`jsonschema` test dependency adds full schema-instance conformance checks. It
needs no chain data, checkpoints, PyTorch model execution, or network access:

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
python -m compileall -q src scripts tests
```

After an editable or wheel installation, the corresponding deterministic CLI
smoke path is:

```bash
crosstaint-artifact --root . --smoke
```

This command validates the release and propagates over the bundled
synthetic-oracle fixture. It does not invoke the model-training or
historical-corpus paths.

After installing all runtime dependencies, exercise the complete synthetic
benchmark path with a small deterministic run:

```bash
crosstaint \
  --local-benchmark \
  --num-benign 20 \
  --num-exploit 2 \
  --num-runs 1 \
  --seed 42 \
  --output-dir results/synthetic-smoke
```

This synthetic command is an implementation smoke test. Its output is not a
reproduction of historical-incident results and should not be reported as
such.

## Benchmark manifests

`schemas/benchmark-manifest.schema.json` documents the JSON/YAML structure
accepted by the corpus loader. `examples/smoke-benchmark.json` is a synthetic,
non-sensitive oracle fixture for deterministic implementation checks. Its
expected path is authored fixture metadata, not evidence from two independent
provenance sources, and it must not be treated as an incident corpus. Exercise
it with `crosstaint-artifact --root . --smoke` as shown above.

For an authorized corpus, replace the example path and optionally provide a
separate benign-address file (`.json`, `.csv`, or `.parquet`):

```bash
crosstaint \
  --full-benchmark \
  --benchmark-manifest /path/to/authorized-manifest.json \
  --benign-addresses /path/to/authorized-benign-addresses.parquet \
  --num-runs 5 \
  --seed 42 \
  --output-dir results/authorized-corpus
```

Result JSON records the public configuration, seed, configuration hash,
manifest hash, dependency versions, platform, evidence scope, per-case
metrics, and aggregate statistics. External baseline adapters can be supplied
with `--external-baselines`; see the CLI help for all options:

```bash
crosstaint --help
```

## Library entry point

`crosstaint.pipeline.InferenceStack` is the main propagation API. Matcher and
pseudonym checkpoints are optional; when supplied, pass the corresponding
event index or graph objects to `InferenceStack.from_config`. The propagation
result contains ranked suspect entries and a per-case soundness-certificate
record when calibrated bound parameters are configured.

## Citation

Citation metadata is available in `CITATION.cff`. Cite the derived release as:

> CrossTaint contributors. *CrossTaint*, version 1.1.0, derived
> artifact-hardening release. <https://github.com/dungtran183/xtain/tree/v1.1.0>

## Responsible use and license

CrossTaint is forensic decision-support software, not an autonomous attribution
or enforcement system. Review `USAGE_POLICY.md` before working with real
addresses or incident labels.

The source code is licensed under the Apache License 2.0. `USAGE_POLICY.md` is
non-binding guidance and does not change that license. See `LICENSE` and
`NOTICE`.
