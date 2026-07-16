# Antara

*Antara* (अंतर) — Sanskrit/Hindi for "interval" or "the space between" —
fills the interval between satellite frames.

Enhancing the temporal resolution of geostationary satellite imagery using
AI/ML techniques based on optical flow — a capstone project targeting the
ISRO problem statement *"Fill in the Frames Seamlessly."*

Geostationary satellites (INSAT-3D/3DR, GOES, Himawari) image the same
region every 10–30 minutes — too coarse to track fast, non-linear
phenomena (cyclones, thunderstorm cells, wildfire plumes) in near-real-time.
This project generates a synthetic intermediate frame between two real
consecutive satellite images, and evaluates where a learned, optical-flow
based method holds up better than classical interpolation — specifically
on fast, non-linear cloud dynamics.

## Status

- [x] Data pipeline: fetch GOES-16 ABI-L1b radiance scans from NOAA's public
      AWS Open Data bucket, extract patch triplets `(t-1, t, t+1)`.
- [x] Classical baseline: Farneback dense optical flow + bidirectional warp/blend.
- [x] Evaluation: PSNR/SSIM against the real held-out middle frame.
- [ ] Deep optical-flow interpolation (fine-tuned RIFE/FILM) — the core contribution.
- [ ] Cyclone/calm evaluation split via IBTrACS.
- [ ] INSAT-3D/3DR validation via MOSDAC (stretch).

See the full plan and rationale for each step at
`~/.claude/plans/sorted-frolicking-nebula.md`.

## Why this is self-supervised

Training/eval data is just real triplets pulled from the satellite
archive — `frame_t` is held out and used only as ground truth. No manual
labeling anywhere in this project.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Usage

```bash
# 1. Download a window of GOES-16 band 13 (clean IR) full-disk scans
.venv/bin/python -m src.data.fetch_goes \
  --start 2024-04-09T12:00 --end 2024-04-09T13:00 --band 13 --out data/raw

# 2. Extract (t-1, t, t+1) patch triplets
.venv/bin/python -m src.data.extract_triplets \
  --raw-dir data/raw --out-dir data/processed/triplets --size 256

# 3. Run the classical Farneback baseline and get PSNR/SSIM
.venv/bin/python -m src.eval.evaluate_baseline \
  --triplets-dir data/processed/triplets --out-csv data/processed/baseline_results.csv

# Run tests
.venv/bin/python -m pytest tests/ -v
```

## Layout

```
src/
  data/       fetch_goes.py        — download GOES-16 scans (public AWS Open Data)
              extract_triplets.py  — NetCDF radiance -> normalized patch triplets
  baseline/   farneback_interpolate.py — classical optical-flow interpolation
  eval/       metrics.py           — PSNR/SSIM
              evaluate_baseline.py — run baseline over all triplets, report metrics
tests/        unit tests for metrics + baseline interpolation
```

## Data source

[NOAA GOES-16 on AWS Open Data](https://registry.opendata.aws/noaa-goes/) —
public, unsigned S3 requests, no registration required. Band 13 (clean
longwave IR, ~10.3μm) is used by default: it works day and night and is
~25MB/scan vs. 300MB+ for the high-resolution visible band.
