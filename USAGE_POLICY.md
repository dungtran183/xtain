# CrossTaint Usage Policy

CrossTaint is forensic decision-support software. It must not be used as the sole basis for asset freezing, sanctions screening, account closure, travel-rule enforcement, or legal attribution.

## Allowed Uses

- Reproducible academic evaluation on documented exploit corpora.
- Internal incident-response triage with human analyst review.
- Synthetic-data experimentation and adapter development.
- Benchmarking against independently verified external baselines.

## Disallowed Uses

- Automated enforcement against a person or account using raw CrossTaint scores alone.
- Republishing real benign-user addresses or address tables without a privacy review.
- Presenting synthetic benchmark output as historical exploit evidence.
- Removing provenance metadata from benchmark manifests or result artifacts.

Any deployment should present confidence scores, cold-start markers, mixer-termination markers, input provenance, and analyst override state together with every suspect address.