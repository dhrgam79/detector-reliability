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

## Open risks
- Runtime budget: corruption (~10-12 min on Colab CPU) + inference must fit in 30 min. Step 4 should overlap corruption with GPU inference (DataLoader workers) or cache corrupted images.
- Default config path resolves relative to the repo checkout; package is meant to be run from a clone, not a wheel install.
