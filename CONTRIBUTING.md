# Contributing

The history of this repository is part of the deliverable. Every change follows the same cadence: a labeled issue, a branch linked to that issue, conventional commits with a rationale, a pull request with evidence, a self review, and a squash merge on green CI. This document is the reference for that cadence.

## Labels and milestones (one-time setup)

```bash
gh label create "type:infra"     --color 5319e7 --description "Scaffold, CI, docker, tooling"
gh label create "type:feat"      --color 0e8a16 --description "New capability"
gh label create "type:exp"       --color d93f0b --description "Experiment with measured result"
gh label create "type:fix"       --color b60205 --description "Defect repair"
gh label create "type:docs"      --color 0075ca --description "Documentation and report"
gh label create "area:data"      --color c2e0c6 --description "Manifest, splits, EDA"
gh label create "area:forensics" --color fef2c0 --description "Signal extractors, feature store"
gh label create "area:baseline"  --color bfd4f2 --description "Classifier baseline"
gh label create "area:reasoner"  --color d4c5f9 --description "VLM prompt, parser, grounding"
gh label create "area:eval"      --color f9d0c4 --description "Harness, calibration, report"
gh label create "area:demo"      --color c5def5 --description "Endpoint and page"

for m in "M0 Scaffold" "M1 Data and EDA" "M2 Forensics" "M3 Baseline" "M4 Reasoner" "M5 Eval, calibration, video"; do
  gh api "repos/AmosBunde/provenance-lens/milestones" -f title="$m"
done
```

## Issues

Every task starts as an issue with four sections: Problem, Proposal, Acceptance criteria, Out of scope. Each issue gets one `type:` label, one `area:` label where one applies, and a milestone. A worked example:

> **Title:** Manifest builder with hashing, dedupe, and license recording
>
> **Problem**
> Assets currently enter the project ad hoc. Without a canonical manifest there is no stable asset key, no dedupe, and no license audit trail, and the frozen test split cannot be defined.
>
> **Proposal**
> Add `provenance_lens.data.manifest` with a builder that walks the raw data directories, records SHA-256, source, license, label, manipulation type, resolution, and perceptual hash per asset, collapses exact and near duplicates, and writes `data/manifest.parquet`.
>
> **Acceptance criteria**
> - `make data` produces the manifest from a clean checkout.
> - Duplicate files with different names appear once.
> - Every row has a non-empty license string.
> - Unit tests cover hashing stability and dedupe.
>
> **Out of scope**
> Split assignment (separate issue), video assets.

## Branches

Branches are created from the issue so the two stay linked:

```bash
gh issue develop <n> --name <type>/<n>-<slug> --checkout
# example: gh issue develop 7 --name feat/7-manifest-builder --checkout
```

## Commits

Conventional commit subject, a body that explains why before what, and a `Refs #<n>` footer. Closing keywords never appear in commits; the pull request closes the issue.

```
feat(data): record perceptual hash at manifest time

Near-duplicate collapse needs a similarity key that survives
re-encoding, which the byte hash does not. Computing the perceptual
hash once at manifest time avoids re-reading every asset during split
construction.

Refs #7
```

Where a reviewer would otherwise have to guess (benchmarks, threshold choices, tradeoffs), the numbers are attached to the commit itself:

```bash
gh api repos/AmosBunde/provenance-lens/commits/<sha>/comments \
  -f body="Dedupe threshold sweep on 2k sample: pHash distance 4 removes 3.1% as near-dupes with 0 false merges on manual check of 50; distance 8 removes 9.4% with 3 false merges. Chose 4."
```

## Pull requests

The pull request body contains `Closes #<n>` and five sections: Summary, What changed, How it was validated, Risks and follow-ups, Reviewer notes. Experiment pull requests lead with the measured result. A worked example:

> Closes #7
>
> **Summary**
> Adds the manifest builder: canonical SHA-256 asset keys, near-duplicate collapse, per-asset license recording.
>
> **What changed**
> New module `data/manifest.py`, config wiring in `data_sources.yaml`, `make data` target, 6 unit tests.
>
> **How it was validated**
> `make data` on a clean checkout builds a 12,408 row manifest in 74 s; rerun is a no-op. Dedupe verified against 50 hand-checked pairs.
>
> **Risks and follow-ups**
> pHash threshold fixed at 4; revisit if EDA shows cross-split near-dupes. Split assignment lands in #8.
>
> **Reviewer notes**
> The dedupe union-find is O(n^2) in the worst case but bucketed by hash prefix; see the commit comment on `<sha>` for the sweep.

## Review cadence

Before merging, the pull request gets a substantive self-review comment: what a skeptical reviewer would probe, and the answer. After merging, the issue gets a closing comment with post-merge evidence.

> **Self-review (on the pull request):** The riskiest piece is the near-dupe collapse silently merging distinct scenes. I re-ran with the threshold at 0 and compared class balance: delta under 0.2% per class, so the collapse is not driving the label distribution. Second concern: license strings come from the source config, not per-file metadata; acceptable because every planned source is single-license.

> **Closing comment (on the issue, after merge):** Merged in #12. Post-merge `make data` from a fresh clone on the CPU box: 12,408 assets, 41 s warm, manifest SHA matches the CI artifact. Follow-up for split assignment tracked in #8.

## Merging

Only with CI green, always squash-merged, with the pull request title as the commit subject. A milestone begins only after the previous one is fully merged to main with CI green.

## Style

Prose in this repository (documentation, reports, issues, pull requests, comments) uses full forms rather than contractions and avoids em dashes in favor of commas, colons, and parentheses. Diagrams are authored as Archify specifications under `architecture/` and regenerated, never hand-edited; a pull request that changes a component boundary regenerates the affected diagram in the same pull request.
