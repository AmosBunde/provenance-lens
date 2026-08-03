# Work breakdown

One issue equals one branch equals one pull request. Every pull request targets a reviewable diff of roughly 150 to 400 changed lines (excluding lockfiles, generated plots, and notebook outputs) and merges only with green CI. Planning numbers below are stable identifiers for this document; GitHub assigns the real numbers at creation time, and branch names use the real number. The real-issue column is filled in as issues are created.

Branch convention: `<type>/<issue>-<slug>` with types `chore`, `feat`, `exp`, `fix`. Pull request bodies follow CONTRIBUTING.md; experiment pull requests lead with the measured result.

Total: 29 planned issues, branches, and pull requests across 6 milestones.

## M0: Scaffold (complete)

M0 was executed before this plan was adopted and landed as five infrastructure pull requests plus one documentation pull request, covering the same scope as planning items 1 to 3.

| Plan # | Scope | Real issues | Merged PRs |
|---|---|---|---|
| 1 | Package skeleton, pyproject, Makefile | #1, #2 | #6, #7 |
| 2 | Pre-commit, pytest, CI pipeline | #3, #4 | #8, #9 |
| 3 | Docker images for CPU and CUDA | #5 | #10 |
| (extra) | README and CONTRIBUTING separation | #11 | #12 |

## M1: Data and EDA (5 PRs)

| Plan # | Issue title | Labels | Branch | PR deliverable | Depends on | Real issue |
|---|---|---|---|---|---|---|
| 4 | Dataset source selection, download scripts, per-asset license manifest | phase:data, type:feat | feat/N-dataset-sources | `make data` downloads and records source plus license per asset | M0 | |
| 5 | Manifest builder: SHA-256 hashing and near-duplicate detection | phase:data, type:feat | feat/N-manifest-dedupe | Deterministic manifest with dedupe report | 4 | |
| 6 | Frozen train/val/test split with scene-level leakage guard and committed test hash | phase:data, type:feat | feat/N-frozen-splits | Pure-function splits, committed SHA-256, CI hash verification | 5 | |
| 7 | Data loader with test-split gating enforced in code | phase:data, type:feat | feat/N-gated-loader | Training imports of the test loader raise; unit test proves it | 6 | |
| 8 | EDA notebook: class-separating signals and dataset artifact audit | phase:data, type:research | exp/N-eda-artifact-audit | Executed notebook exported to docs/report, written findings on artifact features (resolution, quality factor, camera) | 7 | |

## M2: Forensic signal extraction (5 PRs)

| Plan # | Issue title | Labels | Branch | PR deliverable | Depends on | Real issue |
|---|---|---|---|---|---|---|
| 9 | Compression analysis: JPEG ghost detection | phase:forensics, type:feat | feat/N-jpeg-ghosts | Ghost map plus scalar inconsistency score, unit tests on synthetic recompressions | 7 | |
| 10 | Compression analysis: blocking-artifact grid estimation | phase:forensics, type:feat | feat/N-blocking-grid | Grid alignment score, tests on shifted crops | 9 | |
| 11 | Noise residuals: high-pass and PRNU-style with cross-region mismatch | phase:forensics, type:feat | feat/N-noise-residuals | Per-region residual statistics and mismatch score | 7 | |
| 12 | Edge coherence and lighting direction estimators | phase:forensics, type:feat | feat/N-edge-lighting | Both estimators with visual debug output | 7 | |
| 13 | Parquet feature store keyed by asset hash, wired into make features | phase:forensics, type:feat | feat/N-feature-store | One command computes and stores all signals idempotently | 9, 10, 11, 12 | |

## M3: Baseline classifier (4 PRs)

| Plan # | Issue title | Labels | Branch | PR deliverable | Depends on | Real issue |
|---|---|---|---|---|---|---|
| 14 | Frozen backbone embedding extraction pipeline | phase:baseline, type:feat | feat/N-backbone-embeddings | Cached embeddings for all assets, CPU smoke path | 13 | |
| 15 | Detection head model and training loop in PyTorch | phase:baseline, type:feat | feat/N-detection-head | Trainable head over embedding plus structured features, val metrics logged | 14 | |
| 16 | Hyperparameter sweep, early stopping, checkpoint freezing | phase:baseline, type:exp | exp/N-baseline-sweep | Sweep log, frozen best checkpoint, val curve plots | 15 | |
| 17 | Single-shot test scoring: record the number to beat | phase:baseline, type:exp | exp/N-number-to-beat | One test run, score committed to the results table, protocol note that the test set is now closed to the baseline | 16 | |

## M4: VLM reasoning layer (5 PRs)

| Plan # | Issue title | Labels | Branch | PR deliverable | Depends on | Real issue |
|---|---|---|---|---|---|---|
| 18 | VLM client abstraction with API and local backends | phase:reasoner, type:feat | feat/N-vlm-client | One interface, backend chosen in configs/reasoner.yaml, mocked tests | M0 | |
| 19 | Prompt template rendering forensic measurements as structured context | phase:reasoner, type:feat | feat/N-signal-prompt | Deterministic rendering from the feature store, golden-file tests | 13, 18 | |
| 20 | Strict JSON output contract and drift-rejecting parser | phase:reasoner, type:feat | feat/N-json-contract | Schema, parser, failure taxonomy, tests on malformed outputs | 19 | |
| 21 | Evidence grounding check wired into scoring | phase:reasoner, type:feat | feat/N-grounding-check | Cited signals verified against the feature store, ungrounded citations scored as parse failures | 20 | |
| 22 | Batch inference over val and test with per-call cost logging | phase:reasoner, type:feat | feat/N-batch-inference | Resumable batch runner, cost report per split | 21 | |

## M5: Evaluation, calibration, video, demo (7 PRs)

| Plan # | Issue title | Labels | Branch | PR deliverable | Depends on | Real issue |
|---|---|---|---|---|---|---|
| 23 | Scorer: accuracy, precision, recall, F1 with bootstrap CIs | phase:eval, type:feat | feat/N-scorer | Scores both tracks from prediction files on the same frozen split | 17, 22 | |
| 24 | Per-manipulation-type breakdown and confusion matrices | phase:eval, type:feat | feat/N-type-breakdown | Sliced results plus plots | 23 | |
| 25 | Calibration: reliability diagrams, ECE, temperature scaling on val only | phase:eval, type:feat | feat/N-calibration | Calibration module with the val-only fitting rule enforced in code | 23 | |
| 26 | Video frame sampler: uniform plus scene-change | phase:eval, type:feat | feat/N-frame-sampler | Sampler with tests on synthetic clips | 7 | |
| 27 | Temporal consistency scoring and clip-level aggregation | phase:eval, type:feat | feat/N-temporal-consistency | Verdict and signal drift metrics, clip accuracy | 22, 26 | |
| 28 | Report generator and written findings including failure analysis | phase:eval, type:research | exp/N-results-report | make eval produces docs/report end to end, negative results included | 24, 25, 27 | |
| 29 | Demo: FastAPI single-asset verdict endpoint and static page | type:infra, type:feat | feat/N-demo-endpoint | make demo serves upload-and-verdict on :8000 | 21 | |

## Review plan

**Ordering and parallelism.** The dependency column defines the merge order. Within M2, planning items 9 to 12 are independent and can run as four parallel branches; everything else is largely sequential. Never stack more than two unmerged branches, because stacked review comments become unmanageable.

**Size budget.** If a branch grows past roughly 400 reviewable lines, split it: close the branch, split the issue into two issues, and open two pull requests. The EDA (8), sweep (16), and report (28) pull requests are exempt on generated artifacts only; their reviewable source must stay in budget.

**Checklist for infrastructure and feature pull requests** (planning 1 to 15, 18 to 27, 29):

1. Does the diff do only what the linked issue's acceptance criteria say?
2. Is there a test that would fail if the core behavior regressed?
3. Can the stage be re-run idempotently from a clean checkout with one make command?
4. Do names, configs, and docs match the README, and are the architecture diagrams regenerated if a boundary moved?

**Checklist for experiment pull requests** (planning 8, 16, 17, 28):

1. Does the pull request body lead with the measured result?
2. Is the protocol untouched (no test-set reads outside the harness, calibration fitted on validation only)?
3. Are negative or flat results stated rather than smoothed over?
4. Is the result reproducible from the commands listed in the pull request body?

**Cadence.** At one to two pull requests reviewed per day, the remaining work is roughly 3 to 5 weeks of elapsed review time. Independent same-milestone pull requests are batched into a single review sitting (planning 9 to 12 in one session).
