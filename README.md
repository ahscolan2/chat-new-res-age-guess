# Age Guess Service

This repository provides a reference implementation of a privacy-aware age
estimation API. The API performs the following steps:

1. Validates incoming JSON payloads, enforcing consent and configurable
   confidence levels.
2. Loads and normalises RGB portrait images supplied as base64 strings.
3. Detects a single, unobstructed face before cropping to the detected region.
4. Estimates the age with a calibrated model checkpoint and reports confidence
   intervals alongside metadata, fairness considerations, and known
   limitations.

## Project layout

```
.
├── config/
│   └── age_service.json       # Runtime configuration & metadata
├── src/
│   ├── age_service/           # Core service package
│   │   ├── api.py             # Inference entry point
│   │   ├── config.py          # Configuration loader
│   │   ├── detector.py        # Detector abstractions and cropping
│   │   ├── exceptions.py      # Domain-specific error types
│   │   ├── image.py           # Image loading and normalisation utilities
│   │   ├── model.py           # Calibrated model wrapper & confidence maths
│   │   └── validation.py      # Request validation helpers
│   └── api.py                 # Public `run_inference` helper
└── tests/
    └── test_inference.py      # Behavioural tests covering success & rejections
```

The service reads configuration from `config/age_service.json`, which contains
model metadata, fairness warnings, limitations, and operational constraints such
as minimum resolution requirements.

### Payload format

Requests sent to `run_inference` should provide the portrait image as a
base64-encoded ASCII string formatted as `"width,height,intensity"`. The
intensity value is interpreted as a uniform greyscale level from 0 to 255 and is
used by the reference detector and model implementation.

## Running the tests

```bash
pip install -r requirements.txt  # if needed
pytest
```

The provided test suite exercises the happy path along with rejection scenarios
(no faces, multiple faces, occlusion, and low resolution) and input validation
failures to guard against regressions.
