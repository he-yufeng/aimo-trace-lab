# AIMO Trace Lab — source-only pilot

Original implementation of temporal uncertainty summaries: relative-position
slopes, front/tail changes, tail variability and excess path variation for
entropy, top-two margins and selected-token log probabilities. Twelve scalars
plus a valid-token count; no trained predictor or novelty claim for the general
idea of temporal uncertainty. Nine mathematical/boundary controls pass.

The first real CPU preflight **passed** on October6,2026:30upstream unit/contract
tests (including mocks),9original controls, byte-identical Main/Small packages,
and actual135M-model generation with repeatable14aggregate and13temporal features.
It is **not** an8B model, CUDA, official Docker, competition accuracy or award
proof. See [the actual result and scope](RUNTIME_READBACK_R00.md) and
[the frozen pilot](PUBLIC_PILOT_PROTOCOL.md).

The separately attributed, unchanged official uncertainty baseline was submitted
once to CodaBench 16180 Small task 36241 as 963852. At 05:37 UTC October 6,
the official status was Finished: accuracy 0.5714285714, coverage 1,
invalid predictions 0. This is a current-stage baseline score, not a final
test result, rank, award, or original-method improvement. Its pretrained
artifact and fourteen-feature classifier are upstream work, not our original
contribution. Unsupported current models returnFalse; interface coverage is
not successful cross-model modeling.

The manual-only workflow is now disabled after its one original terminal run.
It used a free standard public Linux runner, read-onlypermissions, no secrets,
paid runner, artifact/cache upload, external inference API or scheduled jobs.

Only original source and bounded validation/reporting code are published.
Workspace registrations, contact/proposal receipts, raw datasets, frozen
held-out labels, models, environments, caches and credentials remain excluded
by a strict allowlist. Upstreamcode/artifacts are downloaded for the permitted
example test, not mirrored or claimed as original; no new redistribution
license is asserted for upstream code or raw datasets.

```sh
python -B -m unittest test_trace_shape_r01.py -v
```
