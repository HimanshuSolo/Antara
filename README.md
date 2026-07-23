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
- [x] Full fine-tuning run on a Colab T4 GPU — `notebooks/finetune_on_colab.ipynb`,
      run for real: 30 epochs on 207 triplets pooled from 3 GOES-16 days
      (2024-04-09, 2024-04-11, 2024-04-14), evaluated on a 4th day
      (2024-04-19) never seen during fine-tuning. The core contribution:

      | Method                        | PSNR    | SSIM   | LPIPS  |
      |-------------------------------|---------|--------|--------|
      | Farneback (classical)         | 26.60 dB | 0.6999 | 0.1324 |
      | FILM pretrained (zero-shot)   | 32.66 dB | 0.9242 | 0.0371 |
      | FILM fine-tuned (full run)    | 33.25 dB | 0.9314 | 0.0595 |

      Fine-tuning improves PSNR/SSIM over zero-shot on a fully disjoint test
      day; LPIPS ticks up slightly, a real (small) perceptual-vs-pixel
      trade-off rather than a straight win on every metric.
- [x] Cyclone/calm evaluation split via IBTrACS: `build_cyclone_dataset.py`
      finds a named storm's peak position and pulls GOES-16 scans centered on
      it; `build_calm_dataset.py` pulls the same geographic crop from an
      off-season window; `evaluate_stratified.py` runs Farneback + FILM over
      both and reports PSNR/SSIM/LPIPS per subset — the project's headline
      comparison. Run against the small-scale fine-tuned checkpoint (52 calm
      / 28 cyclone triplets):

      | Subset  | Method     | PSNR    | SSIM   | LPIPS  |
      |---------|------------|---------|--------|--------|
      | calm    | Farneback  | 31.44 dB | 0.8227 | 0.1029 |
      | calm    | FILM       | 36.62 dB | 0.9117 | 0.0472 |
      | cyclone | Farneback  | 24.98 dB | 0.7292 | 0.1250 |
      | cyclone | FILM       | 32.06 dB | 0.9310 | 0.0505 |

      Farneback drops 6.46 dB PSNR going from calm to cyclone conditions;
      FILM drops only 4.56 dB and its SSIM/LPIPS barely move — the
      degrade-sharply-vs-hold-up split the ISRO problem statement predicts.
      Run against the full-scale Colab checkpoint above (52 calm / 28
      cyclone triplets).
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
- [x] Online/continual fine-tuning (stretch): `src/deep/continual_finetune.py`'s
      `continual_update()` fine-tunes an existing checkpoint on just the
      `window_size` most-recently-added triplets from a growing pool,
      rather than retraining from scratch each time new satellite passes
      stream in — adapts to recent conditions (e.g. seasonal cloud-pattern
      shift) instead of being pulled back toward stale ones. Reuses
      `finetune_film.finetune()` and `ablate_finetune_data.materialize_subset()`
      unchanged; windowing is the only new logic. An optional `--test-dir`
      evaluates the updated checkpoint (PSNR/SSIM/LPIPS) so you can tell
      whether a given update actually helped before trusting it. Run for
      real: applied to the existing fine-tuned checkpoint (28.7 dB
      baseline above), windowed to the 16 most-recently-added triplets
      from the same fine-tuning pool, 5 more epochs -- PSNR nudges to
      **28.83 dB** / 0.8254 SSIM on the same held-out test set. Same
      caveat as continual fine-tuning's design intent: this reuses the
      existing pool as a stand-in "growing pool" rather than genuinely new
      streaming data (none was available), so it demonstrates the
      mechanism works end-to-end on real GOES-16 data, not seasonal
      adaptation specifically.
- [x] Live pipeline (stretch): `src/api/live.py` is a small FastAPI
      service that fetches the two most recently published GOES-19 band 13
      scans (GOES-16's successor -- see `src/data/fetch_goes.py`'s
      `latest_scan_pair`) and runs the real Farneback and fine-tuned FILM
      methods on them. There is no real frame between those two scans
      yet, so this is a genuinely live run of the pipeline, not a canned
      one -- everywhere else on the site is static, numbers/images baked
      in ahead of time. The `web/live` page (`LivePipeline` component)
      calls it and renders a before/after slider on the synthesized
      midpoint. Run for real against `noaa-goes19`: a scan pair 10
      minutes apart end to end in ~140s on CPU (download + Farneback +
      FILM inference).
- [x] Gallery (stretch): `src/api/gallery.py` mounts onto the same app
      and serves a handful of curated real GOES-16 (t-1, t+1) pairs --
      Hurricane Milton's 2024 eyewall plus calm off-season weather --
      instantly, with no model involved. Synthesizing the middle frame is
      a separate, on-demand endpoint the `web/live` page's `Gallery`
      component calls when the user clicks "Generate", running the real
      Farneback + FILM pipeline live. Unlike the live pipeline above,
      these triplets have a real ground-truth middle frame on disk, so
      the response includes genuine PSNR/SSIM/LPIPS for both methods, not
      just images.
- [x] Satellite loop endpoint (stretch, real-world usage): `POST
      /api/gallery/{id}/loop` reuses `interpolate_multi()` to generate
      several evenly-spaced FILM frames between the real t-1 and t+1
      scans, then assembles `[t-1, ...interpolated, t+1]` into a single
      looping GIF with Pillow -- a higher-effective-frame-rate satellite
      motion loop, the actual format forecasters watch to track storm
      motion (e.g. the National Hurricane Center's animated loops),
      rather than a single static comparison frame. The `Gallery`
      component's "Generate satellite loop" button calls it after the
      single-frame result. Run for real against the Milton eyewall
      triplets: 5 interpolated frames end to end in ~7s on CPU. The same
      response also includes a downloadable H.264 MP4 of the same
      sequence (encoded via `ffmpeg`'s `libopenh264`, from the same
      generated frames -- no extra FILM inference) -- smaller and more
      shareable than the GIF, and the `Gallery` component exposes it as
      a "Download MP4" link. Degrades gracefully (`loop_mp4: null`) if
      `ffmpeg` isn't installed on the machine running the API.
- [x] Per-triplet PDF report endpoint (stretch, real-world usage): `POST
      /api/gallery/{id}/report` (`src/api/report.py`) renders the same
      t-1/t/t+1 images, Farneback/FILM predictions, and PSNR/SSIM/LPIPS
      figures from `generate()` into a single-page PDF -- the kind of
      shareable, archivable artifact an analyst would attach to an
      incident report, not just something viewed on a live web page.
      Reuses `generate()`'s cache, so it doesn't rerun the model if the
      single-frame result already exists. The `Gallery` component's
      "Generate PDF report" button calls it and exposes the result as a
      "Download PDF report" link.
- [x] Cyclone eye tracking (stretch, new ML application): a second
      application on the same triplets, answering a question PSNR/SSIM/LPIPS
      can't -- does interpolation preserve *where* the storm actually was?
      `src/data/build_eye_labels.py` derives per-frame ground-truth eye
      pixel coordinates from IBTrACS best-track positions (linearly
      interpolated to each scan's exact timestamp) and the crop metadata
      `extract_triplets.py` now writes per triplet. Two detectors run
      against that ground truth: a classical two-stage heuristic
      (`src/baseline/eye_detect.py`, cold-cloud-shield centroid then
      brightest-blob-within-radius, ~5.5px mean error on Milton) and a small
      CNN trained from scratch (`src/deep/eye_detect.py` /
      `eye_detect_train.py`, ~6.3px). Real result on the full 28-triplet
      Milton set (`src/eval/evaluate_eye_detect.py`): running each detector
      on the FILM-synthesized middle frame instead of the real one moves its
      output by only 0.35px (classical) / 1.45px (CNN) on average, vs.
      12.33px / 4.64px for Farneback -- FILM preserves the storm's actual
      position far better than the classical baseline, not just pixel
      similarity. Exposed at `POST /api/track/{id}/detect`
      (`src/api/track.py`) and a new `/track` page (`EyeTracker` component),
      separate from `/live`'s Gallery. The trained detector is a
      single-storm proof of concept, not a generalization test -- pooling
      several storms via `build_cyclone_dataset.py` before training is the
      natural next step, the same path this project already took with
      FILM's fine-tuning.
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

Optional: install `ffmpeg` (system package, not in `requirements.txt`) to
enable MP4 downloads from the gallery's satellite-loop endpoint. Without
it, the endpoint still works and returns the animated GIF preview; it
just returns `loop_mp4: null` instead of a video file.

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

# 13. Continual fine-tuning (stretch): incrementally update an existing
#     checkpoint on just the most-recently-added triplets in a growing
#     pool, instead of retraining from scratch as new passes stream in.
#     --test-dir is optional -- evaluates the updated checkpoint so you
#     can tell whether the update actually helped before trusting it.
.venv/bin/python -m src.deep.continual_finetune \
  --model-path models/film_net_finetuned.pt --pool-dir data/processed/triplets_stream \
  --out-path models/film_net_finetuned_updated.pt --window-size 100 \
  --test-dir data/processed/triplets_test

# 14. Assemble the full report -- narrative (Problem, Related Work, Method,
#     Experiments, Conclusion) plus results tables/figures pulled live from
#     the CSVs/PNGs above, so the numbers can't drift from what was measured.
#     Ablation/multiframe/continual-fine-tuning sections are included
#     automatically if present.
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

## Web frontend

`web/` is a minimal Next.js site: an overview page, a results page (the
same PSNR/SSIM/LPIPS numbers as the Status section above), and an
interactive before/after slider comparing Farneback vs. fine-tuned FILM
on a real cyclone frame. Monochrome by design (white/black, with a
dark-mode variant) -- no component library. Every page except `/live` is
fully static, numbers and images baked in ahead of time by the Python
pipeline above; `/live` is the one exception, with two runtime pieces
served by `src/api/live.py`: the live monitor, which runs the real
pipeline against whatever GOES-19 scans were published most recently
(see "Live pipeline" in the Status section above), and a gallery of
curated cyclone/calm pairs (`src/api/gallery.py`) that generates its
middle frame on demand when you click "Generate", animated as t-1 and
t+1 materializing in with the FILM-synthesized frame resolving between
them (see "Gallery" in the Status section above).

```bash
cd web
npm install
npm run dev   # http://localhost:3000

# in a second terminal, to make the /live page work:
cd ..
.venv/bin/uvicorn src.api.live:app --reload --port 8000
```

See `web/README.md` for its structure.

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
              continual_finetune.py — windowed incremental fine-tuning on a growing triplet pool (stretch)
  eval/       metrics.py               — PSNR/SSIM/LPIPS
              evaluate_baseline.py     — run Farneback baseline over all triplets
              evaluate_film.py         — run FILM (pretrained or fine-tuned) over all triplets
              evaluate_stratified.py   — Farneback + FILM, calm vs. cyclone subsets
              plot_stratified.py       — bar chart: PSNR/SSIM/LPIPS, calm vs. cyclone
              plot_comparison.py       — qualitative side-by-side panels per triplet
              plot_multiframe.py       — Nx multi-frame interpolation panel (stretch)
              ablate_patch_size.py     — Farneback/FILM at several patch sizes
              generate_report.py       — assemble the full report: narrative + results/ablations/figures
              generate_demo.py         — self-contained before/after slider demo per triplet
  api/        live.py                  — FastAPI service backing the web/live page (stretch)
              gallery.py               — on-demand generation over curated cyclone/calm pairs, mounted on live.py's app (stretch)
              report.py                — one-page PDF report builder for a single gallery triplet (stretch)
tests/        unit tests for metrics, baseline, FILM interpolation, and the data pipeline
web/          mostly-static Next.js frontend, one live page — see "Web frontend" above and web/README.md
```

## Data source

[NOAA GOES-16 on AWS Open Data](https://registry.opendata.aws/noaa-goes/) —
public, unsigned S3 requests, no registration required. Band 13 (clean
longwave IR, ~10.3μm) is used by default: it works day and night and is
~25MB/scan vs. 300MB+ for the high-resolution visible band.
