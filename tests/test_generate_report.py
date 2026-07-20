import csv
from pathlib import Path

from src.eval.generate_report import build_report_markdown, write_report


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["triplet", "psnr", "ssim", "lpips"])
        writer.writeheader()
        writer.writerows(rows)


def test_build_report_markdown_includes_results_table(tmp_path):
    results_dir = tmp_path / "stratified"
    _write_csv(results_dir / "calm_farneback.csv", [{"triplet": "a", "psnr": 20.0, "ssim": 0.5, "lpips": 0.3}])
    _write_csv(results_dir / "calm_film.csv", [{"triplet": "a", "psnr": 30.0, "ssim": 0.9, "lpips": 0.1}])

    markdown = build_report_markdown(results_dir, tmp_path / "qualitative", tmp_path)

    assert "| calm | farneback | 20.00 | 0.5000 | 0.3000 |" in markdown
    assert "| calm | film | 30.00 | 0.9000 | 0.1000 |" in markdown
    assert "## Results" in markdown
    assert "<!-- TODO -->" in markdown


def test_build_report_markdown_skips_missing_subsets(tmp_path):
    markdown = build_report_markdown(tmp_path / "missing", tmp_path / "also_missing", tmp_path)
    assert "| calm |" not in markdown


def test_build_report_markdown_embeds_stratified_chart(tmp_path):
    results_dir = tmp_path / "stratified"
    results_dir.mkdir()
    (results_dir / "comparison.png").write_bytes(b"fake png")

    markdown = build_report_markdown(results_dir, tmp_path / "qualitative", tmp_path)

    assert "![Calm vs. cyclone comparison](stratified/comparison.png)" in markdown


def test_build_report_markdown_embeds_qualitative_panels(tmp_path):
    qualitative_dir = tmp_path / "qualitative"
    qualitative_dir.mkdir()
    (qualitative_dir / "triplet_0000.png").write_bytes(b"fake png")

    markdown = build_report_markdown(tmp_path / "stratified", qualitative_dir, tmp_path)

    assert "![triplet_0000](qualitative/triplet_0000.png)" in markdown


def test_build_report_markdown_includes_patch_size_ablation(tmp_path):
    ablation_dir = tmp_path / "ablation_patch_size"
    _write_csv(ablation_dir / "farneback_128.csv", [{"triplet": "a", "psnr": 20.0, "ssim": 0.5, "lpips": 0.3}])
    _write_csv(ablation_dir / "film_128.csv", [{"triplet": "a", "psnr": 30.0, "ssim": 0.9, "lpips": 0.1}])

    markdown = build_report_markdown(
        tmp_path / "stratified", tmp_path / "qualitative", tmp_path, patch_size_dir=ablation_dir
    )

    assert "| 128 | farneback | 20.00 | 0.5000 | 0.3000 |" in markdown
    assert "| 128 | film | 30.00 | 0.9000 | 0.1000 |" in markdown
    assert "### Patch-size ablation" in markdown


def test_build_report_markdown_includes_finetune_data_ablation(tmp_path):
    ablation_dir = tmp_path / "ablation_finetune_data"
    _write_csv(ablation_dir / "test_8.csv", [{"triplet": "a", "psnr": 27.0, "ssim": 0.8, "lpips": 0.08}])

    markdown = build_report_markdown(
        tmp_path / "stratified", tmp_path / "qualitative", tmp_path, finetune_data_dir=ablation_dir
    )

    assert "| 8 | 27.00 | 0.8000 | 0.0800 |" in markdown
    assert "### Fine-tuning data-volume ablation" in markdown


def test_build_report_markdown_embeds_multiframe_panels(tmp_path):
    multiframe_dir = tmp_path / "multiframe"
    multiframe_dir.mkdir()
    (multiframe_dir / "triplet_0000_multiframe.png").write_bytes(b"fake png")

    markdown = build_report_markdown(
        tmp_path / "stratified", tmp_path / "qualitative", tmp_path, multiframe_dir=multiframe_dir
    )

    assert "![triplet_0000_multiframe](multiframe/triplet_0000_multiframe.png)" in markdown
    assert "### Multi-frame interpolation" in markdown


def test_build_report_markdown_omits_ablation_sections_when_absent(tmp_path):
    markdown = build_report_markdown(tmp_path / "stratified", tmp_path / "qualitative", tmp_path)

    assert "### Patch-size ablation" not in markdown
    assert "### Fine-tuning data-volume ablation" not in markdown
    assert "### Multi-frame interpolation" not in markdown


def test_write_report_writes_file(tmp_path):
    out_path = tmp_path / "out" / "report.md"

    written = write_report(tmp_path / "stratified", tmp_path / "qualitative", out_path)

    assert written == out_path
    assert out_path.exists()
