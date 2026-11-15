# Detector Reliability Under Corruption

> Work in progress — results, figures, and the SOTIF analysis will be added as milestones complete.

Evaluates a pretrained Faster R-CNN (torchvision) on COCO images degraded by camera corruptions
(fog, snow, motion blur, noise, brightness; severities 1–5), measuring detection accuracy (mAP)
and confidence calibration (D-ECE), then testing whether temperature scaling fit on clean data
survives distribution shift.

## Setup

```bash
pip install -e ".[dev]"
pytest
```

## License

MIT — see [LICENSE](LICENSE).
