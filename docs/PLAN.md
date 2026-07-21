# 7th Sem Capstone — Satellite Temporal Super-Resolution via Optical-Flow-Based Frame Interpolation

## Context
This is a semester/capstone ML project (not a change to the current codebase — no repo files are touched by this plan). The project targets a real ISRO problem statement: **"Fill in the Frames Seamlessly — Enhancing Temporal Resolution of Satellite Imagery using AI/ML based on Optical Flow."**

Geostationary satellites (INSAT-3D/3DR, GOES, Himawari) image the same region every 10–30 minutes. That's too coarse to track fast, non-linear phenomena — cyclones, thunderstorm cells, wildfire smoke plumes, flash floods — in near-real-time. Classical optical-flow interpolation (linear warping along a flow field) produces blur/artifacts on exactly these fast/non-linear cases, because it assumes locally-linear motion. The goal: generate a synthetic intermediate frame between two real consecutive satellite images using a learned, optical-flow-based method, and show it holds up specifically where classical methods break — fast, non-linear cloud dynamics.

This framing is deliberately chosen to satisfy the earlier stated constraints (CV, research-flavored, free-tier Colab/Kaggle compute, space domain) while matching the ISRO problem statement's explicit ask for an "AI/ML based on Optical Flow" technique.

## Core approach

### 1. Baseline (must build first — this is the thing you're beating)
Classical optical-flow interpolation: compute dense flow with Farneback (OpenCV, `cv2.calcOpticalFlowFarneback`) between frame t-1 and t+1, then warp/blend at t=0.5 to synthesize the middle frame. This is fast to implement and gives you the "traditional method" reference the problem statement explicitly calls out as inadequate.

### 2. Deep optical-flow interpolation (the core contribution)
Replace Farneback with a **pretrained deep frame-interpolation model** to estimate flow between t-1 and t+1 and *learn* how to blend/repair the warped result — this mirrors the architecture used by Super-SloMo / DAIN / RIFE / FILM. Practically:
- **Model: Google's FILM**, via the TorchScript port at `dajes/frame-interpolation-pytorch` (self-contained `.pt` checkpoint, architecture + weights together). **RIFE was tried first and dropped**: its practical/HD checkpoints (the ones every tool actually uses) ship as Python-3.7-only compiled bytecode with no source available anywhere, which blocks both loading on a modern interpreter and, more importantly, the fine-tuning this project needs to do later. FILM has no such issue and is at least as strong a pretrained baseline.
- **Verified**: running the pretrained FILM checkpoint as-is (zero satellite-specific training) on our real GOES-16 test triplets already gives **30.7 dB PSNR / 0.87 SSIM**, vs. the Farneback baseline's 24.4 dB / 0.57 SSIM — confirms the architecture transfers to satellite imagery before any fine-tuning investment (`src/deep/film_interpolate.py`, `src/eval/evaluate_film.py`).
- **Fine-tune** that pretrained model on satellite imagery triplets next. This is the key move that makes free-tier compute feasible — you're adapting an existing strong prior to a new visual domain (clouds/weather), not training flow estimation + synthesis from zero.

### 3. Why this is self-supervised (no labeling cost)
Training data is just real triplets `(frame_{t-1}, frame_t, frame_{t+1})` pulled straight from the satellite archive — `frame_t` is the "label," held out during inference and used only to compute the loss during training. No manual annotation needed anywhere in this project.

## Data plan
- **Primary/prototyping source: GOES-16/17 full-disk imagery** on AWS Open Data (`noaa-goes16`/`noaa-goes17` buckets) — free, no registration, well-documented, ~10–15 min cadence. Use this to build and validate the whole pipeline first.
- **Secondary**: Himawari-8/9 (JMA), similar cadence, also mirrored on AWS Open Data — good for a second-domain generalization check.
- **Stretch goal**: INSAT-3D/3DR via ISRO's MOSDAC portal, to validate directly on the satellite the problem statement names. Registration/access can be slow, so treat this as a late-stage validation step, not a pipeline dependency.
- **Event curation for the "hard test set"**: use **IBTrACS** (International Best Track Archive for Climate Stewardship) cyclone track data to identify known cyclone dates/locations, then pull satellite imagery for those windows. This gives you a principled "fast, non-linear motion" evaluation subset — exactly the case the problem statement says traditional methods fail on — versus a "calm weather" subset for comparison.
- Work on **cropped patches** (e.g. 256–512px around events of interest), not full-disk frames, to keep fine-tuning and inference feasible on a free Colab/Kaggle T4.

## Evaluation (this produces your paper's headline result)
- ~~Standard interpolation metrics: **PSNR, SSIM**, optionally **LPIPS** (perceptual) between the generated middle frame and the real held-out middle frame.~~ **Done** — `src/eval/metrics.py` now reports all three; `evaluate_baseline.py`, `evaluate_film.py`, `evaluate_stratified.py`, and `plot_stratified.py` all carry LPIPS alongside PSNR/SSIM.
- **Stratify results into two subsets**: calm/slow-moving weather vs. cyclone/storm (fast, non-linear) windows from the IBTrACS-curated set. The core result is a table/plot showing classical Farneback degrading sharply on the fast/non-linear subset while the fine-tuned deep model holds up — this stratified comparison IS the novelty contribution.
- Qualitative side-by-sides (real t-1, generated t, real t+1, ground-truth-t) — especially on cyclone frames — for the report/demo.

## Suggested semester timeline
1. ~~**Weeks 1–2**: Data pipeline — pull GOES/Himawari triplets from AWS Open Data, build patch extraction, implement the Farneback baseline end-to-end.~~ **Done.**
2. ~~**Weeks 3–5**: Get a pretrained FILM checkpoint running inference on satellite patches (no fine-tuning yet) — confirms the architecture transfers at all before you invest in training.~~ **Done** — 30.7 dB / 0.87 SSIM, well above the Farneback baseline.
3. ~~**Weeks 6–8**: Fine-tune on satellite triplets; get PSNR/SSIM numbers for baseline vs. fine-tuned model.~~ **Proof-of-concept done on CPU** (`src/deep/finetune_film.py`, `src/deep/dataset.py`) — 46 real triplets from a contiguous 8-hour GOES-16 window, split chronologically into a 38-triplet fine-tuning pool and an 8-triplet **held-out test set** (excluded from fine-tuning entirely, so the comparison below is fair):

   | Method | PSNR | SSIM |
   |---|---|---|
   | Farneback (classical) | 22.6 dB | 0.46 |
   | FILM, pretrained (zero-shot) | 27.8 dB | 0.80 |
   | FILM, fine-tuned (5 epochs, 31 train triplets, CPU) | **28.7 dB** | **0.82** |

   Train loss fell steadily (0.0151→0.0140) and val loss too (0.0232→0.0227) — real signal, not noise. This is a small-scale correctness proof, not the final result: 31 training triplets and 5 epochs on CPU is nowhere near enough data/training to claim the real improvement this architecture can deliver. **Remaining for this milestone**: rerun with far more triplets (multiple days/events, not one 8-hour window) and more epochs on a Colab/Kaggle GPU — `notebooks/finetune_on_colab.ipynb` is ready for this.

   ~~Build the IBTrACS-based cyclone/calm evaluation split.~~ **Done** — `src/data/ibtracs.py` + `src/data/geo_projection.py` locate a named storm's peak position and project it to GOES pixel coordinates; `src/data/build_cyclone_dataset.py` and `src/data/build_calm_dataset.py` pull matched storm-centered and off-season triplets; `src/eval/evaluate_stratified.py` runs Farneback + FILM over both and reports PSNR/SSIM/LPIPS per subset. **Run end-to-end** against the small-scale fine-tuned checkpoint (52 calm / 28 cyclone triplets) — see the table in `README.md`'s Status section. Confirms the hypothesis: Farneback drops 6.46 dB PSNR calm→cyclone, FILM only 4.83 dB. Still needs a rerun against the full-scale checkpoint once the Colab GPU run above lands, to get the real headline number.
4. **Weeks 9–10**: Ablations — ~~patch size~~ **Done** (`src/eval/ablate_patch_size.py`; run on 46 real GOES-16 triplets at 128/256/512px against the pretrained checkpoint -- see the table in README.md's Status section. FILM's PSNR/SSIM barely move across a 16x range in patch area; Farneback improves somewhat with more context but stays well below FILM throughout), ~~fine-tuning data volume~~ **script done** (`src/deep/ablate_finetune_data.py`, fine-tunes on increasing triplet counts from the same pool and evaluates each checkpoint on the same held-out test set; smoke-tested with 4 and 8 triplets at 1 epoch: 27.96 dB → 28.11 dB PSNR, right direction but too small a sweep to read anything into yet -- a real sweep belongs on a Colab/Kaggle GPU, same as fine-tuning itself), ~~single vs. attempting 2x/4x multi-frame interpolation~~ **done as a stretch goal**, see below.
5. **Weeks 11–13**: INSAT/MOSDAC validation attempt (if access came through in time), write-up, plots, demo notebook polish.

## Stretch goals (only if core pipeline lands early)
- ~~Multi-frame interpolation (predict 3 intermediate frames for 4x temporal resolution, not just 1).~~ **Done** — FILM turns out to already be a continuous-time interpolator (its `dt` input isn't hardcoded to 0.5 anywhere in the architecture), so `interpolate_at()`/`interpolate_multi()` in `src/deep/film_interpolate.py` expose that directly rather than needing new architecture. `src/eval/plot_multiframe.py` renders the qualitative panel; run for real on a cyclone triplet at 3x -- see README.md's Status section. Quantitatively checkable only at the true midpoint, since that's the only frame this dataset has ground truth for.
- ~~Online/continual fine-tuning as new satellite passes stream in, adapting to seasonal cloud-pattern shift — revives the "self-improving system" angle discussed earlier, as an optional extension rather than the core deliverable.~~ **Done** — `src/deep/continual_finetune.py`'s `continual_update()` fine-tunes an existing checkpoint on just the `window_size` most-recently-added triplets from a growing pool, rather than retraining from scratch each time, so the model tracks recent conditions instead of stale ones. Reuses `finetune_film.finetune()` and `ablate_finetune_data.materialize_subset()` unchanged; windowing is the only new logic. Smoke-tested (windowing, small-pool, and empty-pool cases); a real run showing measurable seasonal adaptation would still need a genuine time-ordered stream of satellite passes over a real season, which is beyond this project's data budget.
- ~~Small interactive demo (notebook widget or lightweight web page) with a before/after slider comparing classical vs. learned interpolation on a cyclone event.~~ **Done** — `src/eval/generate_demo.py` renders a single self-contained HTML file (images embedded as base64) with a drag slider between the Farneback and FILM predictions for one triplet.

## Deliverable / verification
- Reproducible Colab/Kaggle notebook (pinned versions, fixed seeds) + small repo.
- ~~Core plot: PSNR/SSIM, baseline vs. fine-tuned model, split by calm vs. cyclone subset.~~ **Done** — `src/eval/plot_stratified.py` renders it from `evaluate_stratified.py`'s output CSVs; run for real against the small-scale fine-tuned checkpoint (see README.md's Status section). Still needs a rerun against the full-scale checkpoint once the Colab GPU run lands.
- ~~Qualitative frame comparisons on 2–3 named cyclone events.~~ **Done** — `src/eval/plot_comparison.py` renders real t-1/t+1, ground-truth t, and the Farneback/FILM predictions side by side with PSNR/SSIM/LPIPS in the titles, run for real against `data/processed/triplets_cyclone`. Same caveat: needs the full-scale checkpoint for the final report figures.
- 4–6 page paper-style report (Problem, Related Work — video frame interpolation + remote sensing, Method, Experiments, Results, Conclusion), explicitly framed against the ISRO problem statement. `src/eval/generate_report.py` mechanically assembles the Results section (table + figure references) from the CSVs/PNGs above so those numbers can't drift from what was measured; the narrative sections still need to be written by hand.
