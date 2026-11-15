# Decision Log — detector-reliability

## 2026-11-15 — Step 1 (repo skeleton)

| # | Decision | Rationale |
|---|---|---|
| D1 | Code lives in public GitHub repo `dhrgam79/detector-reliability`; Drive holds brief, decision log, result backups | Portfolio is judged on the repo (history, CI, README) |
| D2 | D-ECE reported at two score thresholds: 0.05 and 0.3 | 0.05 is dominated by low-confidence FPs; 0.3 is closer to a deployed operating point |
| D3 | Primary analysis on driving classes (person, bicycle, car, motorcycle, bus, truck, traffic light, stop sign); all-80 as secondary | Stronger ADAS framing at zero compute cost |
| D4 | 500-image subset sampled only from val2017 images containing >= 1 driving-class object | Otherwise many images have no relevant GT. NOTE: person dominates COCO; report per-class counts |
| D5 | Image-level bootstrap 95% CIs (1000 resamples) on mAP and D-ECE | 250 eval images is statistically thin; claims need error bars |
| D6 | Calibration method named "confidence-logit temperature scaling", not softmax TS | FRCNN scores are post-NMS softmax probs; logit(p) is a heuristic and must be labeled as such |
| D7 | Temperature scaling applied after the 0.05 threshold; no re-thresholding after scaling | Otherwise detection set changes and "mAP unchanged" breaks |
| D8 | Added `src/detrel/config.py` (not in brief §4) | Single place for YAML loading + validation |
| D9 | CI matrix Python 3.10 + 3.12, CPU-only torch wheels | Covers min supported version and current Colab |
| D10 | Commit email = GitHub noreply address | Keeps personal email out of public history |

## 2026-11-15 — Step 2 (corruptions)

| # | Decision | Rationale |
|---|---|---|
| D11 | Vendor 5 corruptions from `imagecorruptions` 1.1.2 into `src/detrel/_vendor/` (Apache-2.0 text + change list included); drop the dependency | Package crashes on NumPy 2 (`fog` uses removed `np.float_`); also imports `pkg_resources` and deprecated `scipy.ndimage.interpolation` |
| D12 | Randomness via per-call `numpy.random.RandomState(seed)`, not `Generator` | NEP 19: RandomState stream is frozen across NumPy versions; Generator streams may change. Also enables exact parity tests vs the reference |
| D13 | Seed per (base_seed, image_id, corruption, severity) via SHA-256, not `hash()` | `hash()` is salted per process; SHA-256 is stable everywhere |
| D14 | Replaced `cv2.cvtColor(RGB2GRAY)` with NumPy weighted sum; dropped OpenCV | One trivial call; lighter CI. Snow differs by <= 1 intensity level on a negligible fraction of pixels |
| D15 | Inputs restricted to RGB uint8 H x W x 3, sides >= 32 px; grayscale branches removed | Matches brief §6.1; less untested code |
| D16 | Truncating uint8 cast kept (not rounding) | Matches reference implementation; required for parity |

### Verification
- Parity vs original `imagecorruptions` (severities 1, 3, 5): fog, motion_blur, gaussian_noise, brightness pixel-exact; snow max diff 1.
- 102 tests pass; vendored module and `corruptions.py` at 100% line coverage.
- CPU cost ~0.06 s/image/condition at 640x480; 250 images x 25 conditions ~6.4 min on a fast CPU (expect ~2x on Colab's 2 vCPUs).

## 2026-11-15 — Step 3 (data)

| # | Decision | Rationale |
|---|---|---|
| D17 | Eligible image = >= 1 **non-crowd** instance of a driving class | A crowd-only "person" region is not a detectable object; would inflate eligibility |
| D18 | Candidates sorted before a single `RandomState(seed).permutation`; first 250 -> calibration, next 250 -> evaluation | Independent of annotation-file order; stable across NumPy versions |
| D19 | Split written to `results/split.json` and **committed**; every run recomputes and aborts on any mismatch (ids, seed, sizes, classes, annotation SHA-256) | Turns "the subset is fixed" from a claim into a checked invariant |
| D20 | Download only the 500 subset images + the annotation zip (~250 MB, only `instances_val2017.json` kept), not the 778 MB val2017 zip | Saves ~10 min on Colab |
| D21 | Split is random, not stratified by class; per-class counts logged and stored in manifest | Stratifying multi-label images is messy; counts let us flag rare classes (stop sign, bus) honestly |
| D22 | `CocoSubset` keeps **all** GT (all classes + crowd flags); class filtering happens at metric time | Same loader serves primary (driving) and secondary (all-80) analyses |
| D23 | `tests/__init__.py` added so test modules can share fixtures | `test_cli` reuses the synthetic COCO builder |

### Open question for Step 5
- How to treat detections that overlap **crowd** GT in D-ECE. COCO mAP ignores them; counting them as FP would penalise correct detections. Proposal: ignore them, matching pycocotools.

## 2026-11-15 — Split results and class groups

Real split (Colab, annotations sha256 `e8c7f790…`): 3028 eligible images. Evaluation GT
instances: person 960, car 182, traffic light 54, truck 31, bicycle 25, motorcycle 20, bus 20,
stop sign 4. Person = 74% of instances in both halves.

| # | Decision | Rationale |
|---|---|---|
| D24 | Report by safety-relevant group: **VRU** (person, bicycle, motorcycle; 1005 eval GT), **vehicle** (car, bus, truck; 233), **infrastructure** (traffic light, stop sign; 58) | Per-class results for rare classes are noise; groups map to SOTIF hazard categories |
| D25 | Headline = standard COCO mAP over 8 classes with bootstrap CI, explicitly labelled class-averaged; D-ECE overall + per VRU/vehicle group | mAP weights stop sign (4 GT) equal to person (960); D-ECE is detection-weighted (~74% person). Readers must know which population each number describes |
| D26 | `min_instances_for_claim: 100`; groups below it reported but flagged; README gets an "Evidence sufficiency" table | SOTIF-style argument about evidence sufficiency; infrastructure (58) is flagged |
| D27 | Evaluation-set size revisited after Step 4 timing (candidate 750–1000) | Narrower CIs for VRU if the 30-min budget allows |

## 2026-11-15 — Step 4 (model + inference + cache)

| # | Decision | Rationale |
|---|---|---|
| D28 | Store post-NMS `boxes, labels, scores` only; no logits | torchvision applies softmax + per-class NMS inside `roi_heads`; exposing the kept detection's logit needs patching internals. Brief §6.4 default path (score-based) |
| D29 | Cache path `cache/{split}/{condition}.pt` (brief said `{corruption}_{severity}.pt`) | Calibration split is also cached (clean); split must be in the key |
| D30 | Cache stores metadata (model, weights, thresholds, AMP, seed, split, SHA-256 of image ids, condition); mismatch -> warn and recompute | Prevents silently reusing detections from a different config |
| D31 | Corruption runs in DataLoader workers; per-image seed makes output independent of worker count / batch size (tested) | Overlaps CPU corruption with GPU inference for the 30-min budget |
| D32 | `amp: false` default | fp16 changes scores slightly; enable only if T4 timing requires it, and document |
| D33 | mAP wrapper (pycocotools, unchanged protocol) pulled forward from Step 6; pycocotools' `-1` (no GT) mapped to `nan`, not 0 | Step 4 needs a baseline number; `nan` avoids silently averaging an undefined value as zero |
| D34 | Sanity reference: torchvision's published 46.7 mAP (all classes, full val2017) logged next to our all-class baseline | Large deviation = pipeline bug (e.g., box format, label mapping) |
| D35 | Label ids verified offline against weights metadata (index = COCO category id) | No remapping layer needed; test guards it |

## 2026-11-15 — First real run (Colab T4, fp32)

Clean all-class mAP 0.444 (eval) / 0.481 (calib) vs torchvision 0.467 -> pipeline validated.
Throughput 4.7 img/s clean, 4.3 img/s brightness-5 -> GPU-bound; CPU corruption overhead ~8%.
Driving 8-class mAP differed by 0.167 between the two random halves (infrastructure 0.633 vs
0.235); infrastructure mAP *rose* under brightness-5 (0.235 -> 0.299) -> noise, 58 GT.

| # | Decision | Rationale |
|---|---|---|
| D36 | Headline = **VRU mAP** + all-80 mAP; 8-class driving mAP demoted to secondary (with CI). Supersedes D25's headline | 8-class mAP swung 17 points between random halves; VRU (Δ 0.038) and all-80 (Δ 0.037) are stable |
| D37 | Runtime budget relaxed **30 -> 45 min** on T4 (brief §1, §9 updated accordingly) | Author's call: spend budget on a larger evaluation set to narrow CIs |
| D38 | Image download parallelised (16 threads) | Sequential took 10 min on Colab |
| D39 | Eval-set size rule: if AMP >= 1.6x faster and every mAP row moves < 0.005 -> AMP + 750 eval images; else fp32 + 400 | Maximise eval size within 45 min without letting fp16 change results; prefix-stable split keeps existing images |

## 2026-11-15 — AMP test (Colab T4)

fp16 autocast: 12.3–12.8 img/s vs 4.6–4.8 fp32 (2.7x). mAP changes <= 0.005 on all rows;
mAP50 changes <= 0.002 except **VRU calibration mAP50 0.795 -> 0.786 (0.009)**.

| # | Decision | Rationale |
|---|---|---|
| D40 | **Adopt AMP despite D39's rule failing on one row** (VRU calib mAP50, 0.009 > 0.005). Breach recorded here, not hidden by moving the threshold | 0.009 is 5–20x below observed between-half sampling noise (0.04–0.17); precision is identical across all conditions, so relative degradation is unaffected; the 2.7x speed buys a 3x larger evaluation set (CI width ~0.58x) |
| D41 | Subset 500 -> 1000: calibration 250 (unchanged), evaluation 250 -> 750 | Prefix-stable selection: calibration identical and previous 250 eval images retained (tested) |
| D42 | README must state: all inference in fp16 autocast; fp32 vs fp16 differences from this test | Disclosure |

## Open risks
- Runtime budget: corruption (~10-12 min on Colab CPU) + inference must fit in 30 min. Step 4 should overlap corruption with GPU inference (DataLoader workers) or cache corrupted images.
- Default config path resolves relative to the repo checkout; package is meant to be run from a clone, not a wheel install.
