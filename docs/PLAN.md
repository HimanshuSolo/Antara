# 7th Sem Capstone — Satellite Temporal Super-Resolution via Optical-Flow-Based Frame Interpolation

## Context
This is a semester/capstone ML project (not a change to the current codebase — no repo files are touched by this plan). The project targets a real ISRO problem statement: **"Fill in the Frames Seamlessly — Enhancing Temporal Resolution of Satellite Imagery using AI/ML based on Optical Flow."**

Geostationary satellites (INSAT-3D/3DR, GOES, Himawari) image the same region every 10–30 minutes. That's too coarse to track fast, non-linear phenomena — cyclones, thunderstorm cells, wildfire smoke plumes, flash floods — in near-real-time. Classical optical-flow interpolation (linear warping along a flow field) produces blur/artifacts on exactly these fast/non-linear cases, because it assumes locally-linear motion. The goal: generate a synthetic intermediate frame between two real consecutive satellite images using a learned, optical-flow-based method, and show it holds up specifically where classical methods break — fast, non-linear cloud dynamics.

This framing is deliberately chosen to satisfy the earlier stated constraints (CV, research-flavored, free-tier Colab/Kaggle compute, space domain) while matching the ISRO problem statement's explicit ask for an "AI/ML based on Optical Flow" technique.

## Core approach

### 1. Baseline (must build first — this is the thing you're beating)
Classical optical-flow interpolation: compute dense flow with Farneback (OpenCV, `cv2.calcOpticalFlowFarneback`) between frame t-1 and t+1, then warp/blend at t=0.5 to synthesize the middle frame. This is fast to implement and gives you the "traditional method" reference the problem statement explicitly calls out as inadequate.

### 2. Deep optical-flow interpolation (the core contribution)
Replace Farneback with a **pretrained deep optical-flow model** (RAFT, available via `torchvision.models.optical_flow.raft_large`, pretrained on Sintel/KITTI/FlyingChairs) to estimate flow between t-1 and t+1, then use a small learned **synthesis/refinement network** to warp + blend + fix occlusion artifacts (holes where content appears/disappears) — this mirrors the architecture used by Super-SloMo / DAIN / RIFE. Practically:
- Start from a pretrained frame-interpolation model checkpoint (RIFE — repo `hzwer/ECCV2022-RIFE` — or Google's FILM) rather than building the flow+synthesis pipeline from scratch.
- **Fine-tune** that pretrained model on satellite imagery triplets. This is the key move that makes free-tier compute feasible — you're adapting an existing strong prior to a new visual domain (clouds/weather), not training flow estimation + synthesis from zero.

### 3. Why this is self-supervised (no labeling cost)
Training data is just real triplets `(frame_{t-1}, frame_t, frame_{t+1})` pulled straight from the satellite archive — `frame_t` is the "label," held out during inference and used only to compute the loss during training. No manual annotation needed anywhere in this project.

## Data plan
- **Primary/prototyping source: GOES-16/17 full-disk imagery** on AWS Open Data (`noaa-goes16`/`noaa-goes17` buckets) — free, no registration, well-documented, ~10–15 min cadence. Use this to build and validate the whole pipeline first.
- **Secondary**: Himawari-8/9 (JMA), similar cadence, also mirrored on AWS Open Data — good for a second-domain generalization check.
- **Stretch goal**: INSAT-3D/3DR via ISRO's MOSDAC portal, to validate directly on the satellite the problem statement names. Registration/access can be slow, so treat this as a late-stage validation step, not a pipeline dependency.
- **Event curation for the "hard test set"**: use **IBTrACS** (International Best Track Archive for Climate Stewardship) cyclone track data to identify known cyclone dates/locations, then pull satellite imagery for those windows. This gives you a principled "fast, non-linear motion" evaluation subset — exactly the case the problem statement says traditional methods fail on — versus a "calm weather" subset for comparison.
- Work on **cropped patches** (e.g. 256–512px around events of interest), not full-disk frames, to keep fine-tuning and inference feasible on a free Colab/Kaggle T4.

## Evaluation (this produces your paper's headline result)
- Standard interpolation metrics: **PSNR, SSIM**, optionally **LPIPS** (perceptual) between the generated middle frame and the real held-out middle frame.
- **Stratify results into two subsets**: calm/slow-moving weather vs. cyclone/storm (fast, non-linear) windows from the IBTrACS-curated set. The core result is a table/plot showing classical Farneback degrading sharply on the fast/non-linear subset while the fine-tuned deep model holds up — this stratified comparison IS the novelty contribution.
- Qualitative side-by-sides (real t-1, generated t, real t+1, ground-truth-t) — especially on cyclone frames — for the report/demo.

## Suggested semester timeline
1. **Weeks 1–2**: Data pipeline — pull GOES/Himawari triplets from AWS Open Data, build patch extraction, implement the Farneback baseline end-to-end.
2. **Weeks 3–5**: Get a pretrained RIFE/FILM checkpoint running inference on satellite patches (no fine-tuning yet) — confirms the architecture transfers at all before you invest in training.
3. **Weeks 6–8**: Fine-tune on satellite triplets; build the IBTrACS-based cyclone/calm evaluation split; get PSNR/SSIM numbers for baseline vs. fine-tuned model on both subsets.
4. **Weeks 9–10**: Ablations — patch size, fine-tuning data volume, single vs. attempting 2x/4x multi-frame interpolation as a stretch.
5. **Weeks 11–13**: INSAT/MOSDAC validation attempt (if access came through in time), write-up, plots, demo notebook polish.

## Stretch goals (only if core pipeline lands early)
- Multi-frame interpolation (predict 3 intermediate frames for 4x temporal resolution, not just 1).
- Online/continual fine-tuning as new satellite passes stream in, adapting to seasonal cloud-pattern shift — revives the "self-improving system" angle discussed earlier, as an optional extension rather than the core deliverable.
- Small interactive demo (notebook widget or lightweight web page) with a before/after slider comparing classical vs. learned interpolation on a cyclone event.

## Deliverable / verification
- Reproducible Colab/Kaggle notebook (pinned versions, fixed seeds) + small repo.
- Core plot: PSNR/SSIM, baseline vs. fine-tuned model, split by calm vs. cyclone subset.
- Qualitative frame comparisons on 2–3 named cyclone events.
- 4–6 page paper-style report (Problem, Related Work — video frame interpolation + remote sensing, Method, Experiments, Results, Conclusion), explicitly framed against the ISRO problem statement.
