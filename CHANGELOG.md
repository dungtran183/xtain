# Changelog

## 1.1.0 — 2026-08-01

Derived artifact-hardening release targeting
<https://github.com/dungtran183/xtain/tree/v1.1.0>.

- Restored the exact official Apache License 2.0 text and added upstream
  provenance plus modification attribution in `NOTICE`.
- Added coherent release and citation metadata without claiming a new DOI.
- Added an offline artifact validator, release inventory, benchmark-manifest
  schema, and deterministic synthetic smoke fixture.
- Packaged the default YAML configuration so installed distributions can run
  outside a source checkout.
- Decoupled fixed-weight propagation from the optional trainable PyTorch edge
  weight module so the deterministic propagation smoke test remains lightweight.
- Added lightweight tests and continuous integration for artifact structure,
  configuration loading, manifest parsing, propagation, and wheel packaging.
- Clarified that responsible-use guidance does not modify Apache-2.0 rights.

No scientific algorithm, trained parameter, benchmark result, or paper claim
was changed by this hardening release.

## 1.0.0 — 2026-06-29 (upstream lineage)

- Upstream source tag: <https://github.com/fuondai/xtain/tree/v1.0.0>.
- Upstream commit: `320883a61888ba7a92a6ad738d951a7dcc7ffd23`.
- Upstream archive: <https://doi.org/10.5281/zenodo.21021751>.
