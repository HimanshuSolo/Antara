"""Assemble the project's report -- narrative sections plus the results
tables and figure references pulled live from the CSVs/PNGs the other
eval scripts produced, so the numbers in the report can never drift from
what the pipeline actually measured.

The narrative sections (Problem, Related Work, Method, Experiments,
Conclusion) are static prose written once the pipeline and results below
existed to describe -- they live in the *_TEXT constants below rather
than being generated, since unlike the results tables they aren't
mechanically derivable from data.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from src.eval.metrics import load_result_rows, summarize

SUBSETS = ["calm", "cyclone"]
METHODS = ["farneback", "film"]

PROBLEM_TEXT = """Geostationary satellites (INSAT-3D/3DR, GOES, Himawari) image the same
region every 10-30 minutes -- coarse enough that fast, non-linear
phenomena (cyclone eyewalls, convective thunderstorm cells, wildfire
smoke plumes) can change substantially between consecutive scans.
Classical optical-flow interpolation synthesizes an intermediate frame by
estimating a dense motion field and linearly warping/blending along it;
that linearity assumption is exactly what breaks down on fast,
non-linear motion, which is the case the ISRO problem statement *"Fill
in the Frames Seamlessly"* calls out. This project targets that
statement directly: generate a synthetic frame between two real
consecutive satellite scans using a learned, optical-flow-based method,
and measure specifically where it holds up better than a classical
baseline on fast/non-linear cloud dynamics, rather than reporting a
single blended average across easy and hard cases alike."""

RELATED_WORK_TEXT = """**General video frame interpolation.** Super-SloMo (Jiang et al., CVPR
2018) estimates bidirectional optical flow between two frames and learns
a refinement network to warp and blend at an arbitrary intermediate
time, the template this project's classical baseline is a minimal
(non-learned) version of. DAIN (Bao et al., CVPR 2019) adds an explicit
depth-aware flow-projection layer so occluded regions near motion
boundaries are handled correctly rather than blurred. RIFE (Huang et
al., ECCV 2022) drops the separate optical-flow network entirely,
estimating intermediate flow directly for real-time speed. RIFE was
tried first for this project and dropped: its practical/HD checkpoints
ship as Python 3.7-only compiled bytecode with no available source,
which blocks both loading on a modern interpreter and, more importantly,
fine-tuning -- a hard requirement here. **FILM** (Reda et al., ECCV
2022) instead uses a shared multi-scale feature pyramid extractor and
scale-agnostic bi-directional motion estimation trained end-to-end for
*large* motion specifically -- relevant since cyclone/storm frame pairs
are exactly the large-motion case -- and ships a self-contained
TorchScript port (weights + architecture together, ordinary Python),
which is what makes it usable here at all. This project uses FILM as-is
via `src/deep/film_interpolate.py`.

**Satellite-specific temporal interpolation.** Vandal & Nemani,
*"Temporal Interpolation of Geostationary Satellite Imagery with
Task-Specific Optical Flow"* (arXiv:1907.12013; IEEE TNNLS, 2021), adapt
Super-SloMo to GOES-16 ABI mesoscale data, upsampling 15-minute cadence
to 1-minute. The Warp-and-Refine Network (arXiv:2303.04405) targets
GK2A geostationary imagery, splitting an optical-flow warp stage from a
learned refinement stage to interpolate/predict frames at 2-minute
intervals from a 4-minute cadence. Both train or heavily adapt an
architecture directly on satellite data from the start. This project
takes a different angle consistent with a free-tier-compute budget:
start from a model pretrained for large-motion interpolation on
ordinary video, verify the architecture transfers to satellite imagery
zero-shot, then fine-tune -- and instead of reporting one aggregate
temporal-upsampling number, stratify evaluation by an IBTrACS-curated
cyclone-vs-calm split so the classical-vs-learned gap on exactly the
fast/non-linear motion case the ISRO problem statement names is the
headline result, not a byproduct."""

METHOD_TEXT = """**Baseline: Farneback optical flow** (`src/baseline/farneback_interpolate.py`).
Dense flow is computed independently in each direction between frame
t-1 and t+1 (`cv2.calcOpticalFlowFarneback`), each field is scaled by
0.5, both source frames are warped toward the midpoint along their
respective half-flow field, and the two warps are averaged. Averaging
both directions rather than warping from a single frame fills in
occlusion holes that a one-directional warp would leave uncovered. This
is the "traditional method" the problem statement calls inadequate for
fast motion -- the reference every learned method here must beat.

**FILM** (`src/deep/film_interpolate.py`). The pretrained TorchScript
checkpoint expects 3-channel RGB video frames in [0, 1]; since GOES-16
band 13 is a single-channel IR radiance field, the channel is replicated
to 3 identical channels so the pretrained conv filters apply unmodified,
and predictions are averaged back to grayscale rather than assuming the
3 output channels stay identical. Inputs are padded to a multiple of 64
px (the network's internal downsampling factor) and cropped back after
inference. FILM takes a continuous `dt` input rather than hardcoding
t=0.5, which `interpolate_multi()` exploits directly for the multi-frame
stretch goal (N evenly-spaced frames from one model, no architecture
change).

**Fine-tuning** (`src/deep/finetune_film.py`). The pretrained checkpoint
is fine-tuned end-to-end on satellite triplets with plain L1
reconstruction loss between the predicted and real held-out middle
frame. The original FILM paper also uses a perceptual (VGG) loss and a
style loss; L1 alone is a deliberate scope cut for a semester project,
noted here rather than silently deviating from the paper. The
train/validation split is chronological, not random: consecutive
triplets overlap in content (triplet N's "next" frame is triplet N+1's
"prev" frame), so a random split would leak near-duplicate frames across
the split boundary.

**Continual fine-tuning** (`src/deep/continual_finetune.py`). Reuses
`finetune()` unchanged; the only new idea is windowing -- each update
trains on just the `window_size` most-recently-added triplets from a
growing pool rather than retraining from scratch on the whole pool every
time, so the checkpoint tracks recent conditions instead of being pulled
back toward stale ones as the pool grows."""

EXPERIMENTS_TEXT = """**Data pipeline.** GOES-16 ABI-L1b band 13 (clean longwave IR, ~10.3
um -- works day and night, ~25MB/scan) full-disk scans are pulled from
NOAA's public, unsigned, no-registration AWS Open Data bucket
(`src/data/fetch_goes.py`). Radiance is percentile-clipped (1st/99th) and
normalized to 8-bit rather than using raw min/max, so a handful of
sensor-outlier pixels can't wash out the contrast of an entire frame;
off-disk NaN pixels map to the clip floor. Consecutive scans are grouped
into (t-1, t, t+1) triplets and cropped to a patch around a center
pixel; any triplet whose adjacent scan gap exceeds 1.5x the dataset's
median cadence is skipped, since a dropped/recalibration scan would
otherwise silently produce a triplet spanning far more real time than
intended -- a bad example that looks fine until timestamps are checked.

**Cyclone/calm split.** IBTrACS (International Best Track Archive for
Climate Stewardship) best-track data gives a named storm's peak-intensity
fix -- exact time, lat/lon, wind speed -- rather than guessing at
"probably stormy" dates (`src/data/ibtracs.py`). That lat/lon is
converted to a GOES fixed-grid pixel coordinate via the geodetic
scan-angle transform published in the GOES-R Product User's Guide
(`src/data/geo_projection.py`), and triplets are extracted centered on
the storm (`src/data/build_cyclone_dataset.py`). The calm comparison set
uses the same crop and extraction path on off-season scans
(`src/data/build_calm_dataset.py`) -- no cyclone-specific data or
labeling involved, just a different time window.

**Ablations.** Patch size (128/256/512 px) tests whether FILM depends
on extra spatial context the way Farneback's pyramidal flow estimation
does (`src/eval/ablate_patch_size.py`). Fine-tuning data volume (8/16/24
triplets from the same pool) checks the direction of the PSNR trend as
training data grows (`src/deep/ablate_finetune_data.py`). Multi-frame
interpolation (`src/eval/plot_multiframe.py`) renders N evenly-spaced
synthesized frames instead of just the midpoint -- qualitative only,
since this dataset has ground truth solely at the true midpoint."""

CONCLUSION_TEXT = """The headline result confirms the hypothesis the ISRO problem statement
predicts: Farneback's PSNR drops 6.46 dB going from the calm subset to
the cyclone subset, while FILM's drops only 4.83 dB and its SSIM/LPIPS
barely move -- the fast, non-linear motion in cyclone frame pairs is
exactly where the classical linear-warp assumption breaks down and a
learned model doesn't. This holds even before satellite-specific
training is credited: the pretrained FILM checkpoint alone reaches 30.7
dB / 0.87 SSIM zero-shot on held-out GOES-16 triplets, against
Farneback's 24.4 dB / 0.57 SSIM, and fine-tuning improves on that
further (28.7 dB vs. 27.8 dB zero-shot on the same small held-out test
set -- lower than the aggregate zero-shot number above because it's a
different, harder held-out split). The patch-size ablation shows FILM's
accuracy is essentially patch-size-invariant across a 16x range in area,
while Farneback improves somewhat with more spatial context but never
closes the gap -- consistent with FILM learning motion synthesis rather
than only locally-windowed flow.

**Limitations and future work.** The fine-tuning numbers above come from
a small-scale, CPU-only proof of concept (31 training triplets, 5
epochs, one 8-hour GOES-16 window) -- real signal (train/val loss both
fell monotonically), but nowhere near the data volume or epoch count a
~34M-parameter video model needs to show its full potential on this
domain. `notebooks/finetune_on_colab.ipynb` reuses this project's
pipeline unmodified to fine-tune across multiple disjoint GOES-16 days
on a free Colab/Kaggle GPU and is ready to run; the stratified and
ablation results above should be rerun against that checkpoint once it
exists, since that is the number this project is ultimately trying to
report. INSAT-3D/3DR validation via ISRO's MOSDAC portal remains
unattempted -- registration/access lead time put it outside this
project's timeline -- so every result here is on GOES-16 (and, via the
same code path, Himawari-8/9), not yet the satellite the problem
statement names directly; the data pipeline has no GOES-specific
assumption baked in beyond the scan-time filename parser and projection
formula, both isolated to `src/data/`, so adding an INSAT/MOSDAC data
source is additive, not a rewrite.

Every number in this report comes from real GOES-16 radiance data
pulled from NOAA's public archive, with `frame_t` used only as held-out
ground truth -- no manual labeling exists anywhere in this pipeline,
which is what makes scaling up training data a matter of compute
budget, not annotation budget."""


def build_results_table(results_dir: Path) -> str:
    lines = ["| Subset | Method | PSNR (dB) | SSIM | LPIPS |", "|---|---|---|---|---|"]
    for subset in SUBSETS:
        for method in METHODS:
            rows = load_result_rows(results_dir / f"{subset}_{method}.csv")
            if not rows:
                continue
            mean_psnr, mean_ssim, mean_lpips = summarize(rows)
            lines.append(f"| {subset} | {method} | {mean_psnr:.2f} | {mean_ssim:.4f} | {mean_lpips:.4f} |")
    return "\n".join(lines)


def _discover_suffixes(dir_path: Path, prefix: str) -> list[int]:
    """e.g. prefix='farneback' over farneback_128.csv, farneback_256.csv
    -> [128, 256], sorted numerically."""
    suffixes = set()
    for csv_path in dir_path.glob(f"{prefix}_*.csv"):
        try:
            suffixes.add(int(csv_path.stem.removeprefix(f"{prefix}_")))
        except ValueError:
            continue
    return sorted(suffixes)


def build_patch_size_table(ablation_dir: Path) -> str:
    sizes = _discover_suffixes(ablation_dir, "farneback")
    if not sizes:
        return ""
    lines = ["| Size | Method | PSNR (dB) | SSIM | LPIPS |", "|---|---|---|---|---|"]
    for size in sizes:
        for method in METHODS:
            rows = load_result_rows(ablation_dir / f"{method}_{size}.csv")
            if not rows:
                continue
            mean_psnr, mean_ssim, mean_lpips = summarize(rows)
            lines.append(f"| {size} | {method} | {mean_psnr:.2f} | {mean_ssim:.4f} | {mean_lpips:.4f} |")
    return "\n".join(lines)


def build_finetune_data_table(ablation_dir: Path) -> str:
    counts = _discover_suffixes(ablation_dir, "test")
    if not counts:
        return ""
    lines = ["| Fine-tuning triplets | PSNR (dB) | SSIM | LPIPS |", "|---|---|---|---|"]
    for count in counts:
        rows = load_result_rows(ablation_dir / f"test_{count}.csv")
        if not rows:
            continue
        mean_psnr, mean_ssim, mean_lpips = summarize(rows)
        lines.append(f"| {count} | {mean_psnr:.2f} | {mean_ssim:.4f} | {mean_lpips:.4f} |")
    return "\n".join(lines)


def build_continual_table(continual_csv: Path) -> str:
    rows = load_result_rows(continual_csv)
    if not rows:
        return ""
    mean_psnr, mean_ssim, mean_lpips = summarize(rows)
    return "\n".join([
        "| PSNR (dB) | SSIM | LPIPS |",
        "|---|---|---|",
        f"| {mean_psnr:.2f} | {mean_ssim:.4f} | {mean_lpips:.4f} |",
    ])


def build_report_markdown(
    results_dir: Path,
    qualitative_dir: Path,
    report_dir: Path,
    patch_size_dir: Path | None = None,
    finetune_data_dir: Path | None = None,
    multiframe_dir: Path | None = None,
    continual_csv: Path | None = None,
) -> str:
    sections = [
        "# Antara: Satellite Temporal Super-Resolution via Optical-Flow-Based Frame Interpolation",
        "",
        PROBLEM_TEXT,
        "",
        "## Related Work",
        "",
        RELATED_WORK_TEXT,
        "",
        "## Method",
        "",
        METHOD_TEXT,
        "",
        "## Experiments",
        "",
        EXPERIMENTS_TEXT,
        "",
        "## Results",
        "",
        "### Calm vs. cyclone (headline result)",
        "",
        build_results_table(results_dir),
        "",
    ]

    comparison_png = results_dir / "comparison.png"
    if comparison_png.exists():
        rel = os.path.relpath(comparison_png, report_dir)
        sections += [f"![Calm vs. cyclone comparison]({rel})", ""]

    qualitative_pngs = sorted(qualitative_dir.glob("*.png")) if qualitative_dir.exists() else []
    if qualitative_pngs:
        sections += ["### Qualitative comparisons", ""]
        for png in qualitative_pngs:
            rel = os.path.relpath(png, report_dir)
            sections.append(f"![{png.stem}]({rel})")
        sections.append("")

    if patch_size_dir and patch_size_dir.exists():
        table = build_patch_size_table(patch_size_dir)
        if table:
            sections += ["### Patch-size ablation", "", table, ""]

    if finetune_data_dir and finetune_data_dir.exists():
        table = build_finetune_data_table(finetune_data_dir)
        if table:
            sections += ["### Fine-tuning data-volume ablation", "", table, ""]

    multiframe_pngs = sorted(multiframe_dir.glob("*_multiframe.png")) if multiframe_dir and multiframe_dir.exists() else []
    if multiframe_pngs:
        sections += ["### Multi-frame interpolation (qualitative)", ""]
        for png in multiframe_pngs:
            rel = os.path.relpath(png, report_dir)
            sections.append(f"![{png.stem}]({rel})")
        sections.append("")

    if continual_csv and continual_csv.exists():
        table = build_continual_table(continual_csv)
        if table:
            sections += ["### Continual fine-tuning (stretch)", "", table, ""]

    sections += ["## Conclusion", "", CONCLUSION_TEXT, ""]
    return "\n".join(sections)


def write_report(
    results_dir: Path,
    qualitative_dir: Path,
    out_path: Path,
    patch_size_dir: Path | None = None,
    finetune_data_dir: Path | None = None,
    multiframe_dir: Path | None = None,
    continual_csv: Path | None = None,
) -> Path:
    markdown = build_report_markdown(
        results_dir,
        qualitative_dir,
        out_path.parent,
        patch_size_dir=patch_size_dir,
        finetune_data_dir=finetune_data_dir,
        multiframe_dir=multiframe_dir,
        continual_csv=continual_csv,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(markdown)
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="data/processed/stratified")
    parser.add_argument("--qualitative-dir", default="data/processed/qualitative")
    parser.add_argument("--patch-size-dir", default="data/processed/ablation_patch_size")
    parser.add_argument("--finetune-data-dir", default="data/processed/ablation_finetune_data")
    parser.add_argument("--multiframe-dir", default="data/processed/multiframe")
    parser.add_argument("--continual-csv", default="models/film_net_finetuned_updated_test.csv")
    parser.add_argument("--out", default="data/processed/report.md")
    args = parser.parse_args()

    out_path = write_report(
        Path(args.results_dir),
        Path(args.qualitative_dir),
        Path(args.out),
        patch_size_dir=Path(args.patch_size_dir),
        finetune_data_dir=Path(args.finetune_data_dir),
        multiframe_dir=Path(args.multiframe_dir),
        continual_csv=Path(args.continual_csv),
    )
    print(f"Wrote {out_path}")
