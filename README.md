# Antara

[![tests](https://github.com/HimanshuSolo/Antara/actions/workflows/tests.yml/badge.svg)](https://github.com/HimanshuSolo/Antara/actions/workflows/tests.yml)

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
- [x] Evaluation: PSNR/SSIM/LPIPS against the real held-out middle frame.
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
      both and reports PSNR/SSIM/LPIPS per subset — the project's headline
      comparison. Run against the small-scale fine-tuned checkpoint (52 calm
      / 28 cyclone triplets):

      | Subset  | Method     | PSNR    | SSIM   | LPIPS  |
      |---------|------------|---------|--------|--------|
      | calm    | Farneback  | 31.44 dB | 0.823 | 0.103 |
      | calm    | FILM       | 36.88 dB | 0.917 | 0.041 |
      | cyclone | Farneback  | 24.98 dB | 0.729 | 0.125 |
      | cyclone | FILM       | 32.05 dB | 0.932 | 0.044 |

      Farneback drops 6.46 dB PSNR going from calm to cyclone conditions;
      FILM drops only 4.83 dB and its SSIM/LPIPS barely move — the
      degrade-sharply-vs-hold-up split the ISRO problem statement predicts.
      Same caveat as the fine-tuning result above: small-scale checkpoint,
      not yet the full Colab GPU run.
- [x] Patch-size ablation (`src/eval/ablate_patch_size.py`), run on 46 real
      GOES-16 triplets at 128/256/512px against the pretrained checkpoint:

      | Size | Method     | PSNR    | SSIM   | LPIPS  |
      |------|------------|---------|--------|--------|
      | 128  | Farneback  | 22.80 dB | 0.537 | 0.181 |
      | 128  | FILM       | 31.60 dB | 0.886 | 0.062 |
      | 256  | Farneback  | 23.73 dB | 0.558 | 0.186 |
      | 256  | FILM       | 31.53 dB | 0.875 | 0.084 |
      | 512  | Farneback  | 24.65 dB | 0.602 | 0.184 |
      | 512  | FILM       | 31.89 dB | 0.886 | 0.090 |

      FILM's PSNR/SSIM barely move across a 16x range in patch area (128px
      to 512px) — it isn't relying on extra spatial context. Farneback
      improves somewhat with more context (22.8 → 24.65 dB), consistent
      with its pyramidal flow estimation benefiting from a larger search
      window, but still stays well below FILM at every size.
- [x] Multi-frame interpolation (stretch): `interpolate_multi()` in
      `src/deep/film_interpolate.py` generalizes FILM inference to any
      normalized time `t` (it's a continuous-time interpolator, not just a
      midpoint one) and calls it at N evenly-spaced points, turning one
      real frame gap into Nx the temporal resolution.
      `src/eval/plot_multiframe.py` renders the qualitative panel; run for
      real on a cyclone triplet at 3x (t=0.25/0.5/0.75) — the eye and
      cloud bands progress smoothly across all 5 frames. Qualitative only:
      this dataset has ground truth solely at the true midpoint, so only
      the num_frames=1 case is quantitatively checkable (that's what
      `evaluate_film.py` already does).
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

# 3. Run the classical Farneback baseline and get PSNR/SSIM/LPIPS
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

# 8. Render the report figures: the calm-vs-cyclone bar chart and per-triplet
#    qualitative side-by-sides
.venv/bin/python -m src.eval.plot_stratified
.venv/bin/python -m src.eval.plot_comparison \
  --triplets-dir data/processed/triplets_cyclone --film-model-path models/film_net_finetuned.pt

# 9. Patch-size ablation: re-extract triplets at several sizes from the same
#    raw scans and compare Farneback/FILM at each
.venv/bin/python -m src.eval.ablate_patch_size \
  --raw-dir data/raw --sizes 128 256 512 --film-model-path models/film_net_finetuned.pt

# 10. Fine-tuning data-volume ablation: fine-tune on increasing triplet
#     counts from the same pool, evaluate every checkpoint on the same
#     held-out test set
.venv/bin/python -m src.deep.ablate_finetune_data \
  --model-path models/film_net_fp32.pt --triplets-finetune-dir data/processed/triplets_finetune \
  --triplets-test-dir data/processed/triplets_test --counts 8 16 24 31

# 11. Interactive demo: a single self-contained HTML page with a
#     before/after slider comparing Farneback vs. FILM on one triplet
.venv/bin/python -m src.eval.generate_demo \
  --triplets-dir data/processed/triplets_cyclone --film-model-path models/film_net_finetuned.pt

# 12. Multi-frame interpolation (stretch): N evenly-spaced synthesized
#     frames instead of just the midpoint, qualitative only
.venv/bin/python -m src.eval.plot_multiframe \
  --triplets-dir data/processed/triplets_cyclone --film-model-path models/film_net_finetuned.pt --num-frames 3

# 13. Assemble everything above into a report skeleton (narrative sections
#     left as TODOs -- see docs/PLAN.md for the deliverable this fills in;
#     ablation/multiframe sections are included automatically if present)
.venv/bin/python -m src.eval.generate_report

# Run tests
.venv/bin/python -m pytest tests/ -v

# Lint + type check (same checks CI runs; config lives in pyproject.toml)
.venv/bin/pip install ruff mypy
.venv/bin/ruff check src/ tests/
.venv/bin/python -m mypy src/ tests/
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
              ablate_finetune_data.py — fine-tune on increasing triplet counts, evaluate each checkpoint
  eval/       metrics.py               — PSNR/SSIM/LPIPS
              evaluate_baseline.py     — run Farneback baseline over all triplets
              evaluate_film.py         — run FILM (pretrained or fine-tuned) over all triplets
              evaluate_stratified.py   — Farneback + FILM, calm vs. cyclone subsets
              plot_stratified.py       — bar chart: PSNR/SSIM/LPIPS, calm vs. cyclone
              plot_comparison.py       — qualitative side-by-side panels per triplet
              plot_multiframe.py       — Nx multi-frame interpolation panel (stretch)
              ablate_patch_size.py     — Farneback/FILM at several patch sizes
              generate_report.py       — assemble results + ablations + figures into a report skeleton
              generate_demo.py         — self-contained before/after slider demo per triplet
tests/        unit tests for metrics, baseline, FILM interpolation, and the data pipeline
```

## Data source

[NOAA GOES-16 on AWS Open Data](https://registry.opendata.aws/noaa-goes/) —
public, unsigned S3 requests, no registration required. Band 13 (clean
longwave IR, ~10.3μm) is used by default: it works day and night and is
~25MB/scan vs. 300MB+ for the high-resolution visible band.
