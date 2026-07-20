# Antara

*Antara* (अंतर) — It means "interval" or "the space between" —
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
- [x] Pretrained deep interpolation (FILM, zero-shot): 30.7 dB / 0.87 SSIM vs. baseline's 24.4 dB / 0.57 SSIM.
- [x] Fine-tuning proof-of-concept on CPU: 28.7 dB / 0.82 SSIM on a held-out test
      set, vs. 27.8 dB / 0.80 SSIM zero-shot on the same set. Small-scale (31
      training triplets, 5 epochs) -- see `docs/PLAN.md` for what's left.
- [x] GPU support added to fine-tuning/inference (`--device`, auto-detects cuda).
- [ ] Full fine-tuning run on more data on a Colab/Kaggle GPU — see
      `notebooks/finetune_on_colab.ipynb`, ready to run. The core contribution.
- [x] Cyclone/calm evaluation split via IBTrACS: `build_cyclone_dataset.py`
      finds a named storm's peak position and pulls GOES-16 scans centered on
      it; `build_calm_dataset.py` pulls the same geographic crop from an
      off-season window; `evaluate_stratified.py` runs Farneback + FILM over
      both and reports PSNR/SSIM per subset — the project's headline
      comparison.
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

# 4. Download the pretrained FILM checkpoint and evaluate it (no fine-tuning yet)
.venv/bin/python -m src.deep.download_film --out models/film_net_fp32.pt
.venv/bin/python -m src.eval.evaluate_film \
  --triplets-dir data/processed/triplets --model-path models/film_net_fp32.pt --out-csv data/processed/film_results.csv

# 5. Fine-tune FILM on a directory of triplets (exclude your test set from this dir!)
.venv/bin/python -m src.deep.finetune_film \
  --model-path models/film_net_fp32.pt --triplets-dir data/processed/triplets_finetune \
  --out-path models/film_net_finetuned.pt --epochs 5 --batch-size 2

# 6. Evaluate the fine-tuned model the same way as step 4, on a held-out set
.venv/bin/python -m src.eval.evaluate_film \
  --triplets-dir data/processed/triplets_test --model-path models/film_net_finetuned.pt \
  --out-csv data/processed/test_film_finetuned_results.csv

# 7. Build the cyclone/calm evaluation split and get the stratified comparison
.venv/bin/python -m src.data.build_cyclone_dataset --storm-name MILTON --season 2024
.venv/bin/python -m src.data.build_calm_dataset
.venv/bin/python -m src.eval.evaluate_stratified --film-model-path models/film_net_finetuned.pt

# Run tests
.venv/bin/python -m pytest tests/ -v
```

## Scaling up fine-tuning on Colab

`notebooks/finetune_on_colab.ipynb` reuses the same pipeline code above
(nothing reimplemented) to fine-tune on 3 separate GOES-16 days and
evaluate on a 4th, fully disjoint day, on a free Colab GPU. Open it in
Colab via GitHub, enable a GPU runtime, and run top to bottom -- it'll
prompt for a GitHub personal access token to clone this private repo,
and saves the fine-tuned checkpoint + result CSVs to Google Drive so
they survive when the session ends.

## Layout

```
src/
  data/       fetch_goes.py            — download GOES-16 scans (public AWS Open Data)
              extract_triplets.py      — NetCDF radiance -> normalized patch triplets
              geo_projection.py        — lat/lon -> GOES full-disk pixel coordinates
              ibtracs.py                — load cyclone track data from IBTrACS
              build_cyclone_dataset.py — triplets centered on a storm's peak position
              build_calm_dataset.py    — off-season triplets at the same crop, for comparison
  baseline/   farneback_interpolate.py — classical optical-flow interpolation
  deep/       download_film.py     — fetch pretrained FILM TorchScript checkpoint
              film_interpolate.py  — deep frame interpolation (FILM)
              dataset.py           — PyTorch Dataset over triplet directories
              finetune_film.py     — fine-tune FILM on satellite triplets
  eval/       metrics.py               — PSNR/SSIM
              evaluate_baseline.py     — run Farneback baseline over all triplets
              evaluate_film.py         — run FILM (pretrained or fine-tuned) over all triplets
              evaluate_stratified.py   — Farneback + FILM, calm vs. cyclone subsets
tests/        unit tests for metrics, baseline, FILM interpolation, and the data pipeline
```

## Data source

[NOAA GOES-16 on AWS Open Data](https://registry.opendata.aws/noaa-goes/) —
public, unsigned S3 requests, no registration required. Band 13 (clean
longwave IR, ~10.3μm) is used by default: it works day and night and is
~25MB/scan vs. 300MB+ for the high-resolution visible band.
