import type { Metadata } from "next";
import ResultsTable from "@/components/ResultsTable";
import PsnrBarChart from "@/components/PsnrBarChart";
import {
  stratifiedResults,
  zeroShotResults,
  finetuneProofOfConcept,
  finetuneFullScale,
  patchSizeAblation,
} from "@/lib/results";

export const metadata: Metadata = {
  title: "Results — Antara",
};

export default function ResultsPage() {
  return (
    <div className="container section">
      <div className="section-label">Results</div>
      <h1>Evaluation results</h1>
      <p className="lede">
        All figures below are derived from real GOES-16 radiance data obtained from
        NOAA&apos;s public archive. No synthetic or held-in data is used at any stage of this
        pipeline.
      </p>

      <section className="section--tight">
        <h2>Stratified evaluation: calm vs. cyclone conditions</h2>
        <p className="prose">
          The Farneback baseline shows a 6.46 dB drop in PSNR when moving from calm to cyclone
          conditions, while FILM shows a smaller drop of only 4.56 dB, with SSIM and LPIPS
          remaining largely stable. This confirms the hypothesis underlying the ISRO problem
          statement: classical interpolation degrades sharply under fast, non-linear motion,
          while the learned model remains robust.
        </p>
        <PsnrBarChart />
        <div style={{ marginTop: "2rem" }}>
          <ResultsTable
            columns={["Subset", "Method", "PSNR (dB)", "SSIM", "LPIPS"]}
            rows={stratifiedResults.flatMap((subset) =>
              subset.results.map((r) => ({
                key: `${subset.subset}-${r.method}`,
                cells: [
                  subset.subset,
                  r.method,
                  r.psnr.toFixed(2),
                  r.ssim.toFixed(3),
                  r.lpips.toFixed(3),
                ],
              })),
            )}
          />
        </div>
        <p className="caveat">
          Evaluated using the full-scale, Colab-fine-tuned checkpoint across 52 calm and 28
          cyclone triplets.
        </p>
      </section>

      <section className="section--tight">
        <h2>Zero-shot transfer</h2>
        <p className="prose">
          The pretrained FILM checkpoint, applied without any satellite-specific training,
          already outperforms the classical baseline. This confirms that the architecture
          transfers effectively to satellite imagery prior to any fine-tuning.
        </p>
        <ResultsTable
          columns={["Method", "PSNR (dB)", "SSIM"]}
          rows={zeroShotResults.map((r) => ({
            key: r.method,
            cells: [r.method, r.psnr.toFixed(1), r.ssim.toFixed(2)],
          }))}
        />
      </section>

      <section className="section--tight">
        <h2>Fine-tuning proof of concept</h2>
        <p className="prose">
          This proof of concept uses 46 real triplets from a contiguous 8-hour GOES-16 window,
          split chronologically into a 38-triplet fine-tuning pool and an 8-triplet held-out
          test set that is excluded from fine-tuning entirely.
        </p>
        <ResultsTable
          columns={["Method", "PSNR (dB)", "SSIM"]}
          rows={finetuneProofOfConcept.map((r) => ({
            key: r.method,
            cells: [r.method, r.psnr.toFixed(1), r.ssim.toFixed(2)],
          }))}
        />
        <p className="caveat">
          Small-scale run: 31 training triplets, 5 epochs, CPU only. See the full-scale GPU run
          below.
        </p>
      </section>

      <section className="section--tight">
        <h2>Full-scale fine-tuning (Colab GPU)</h2>
        <p className="prose">
          This is the core result of the project: fine-tuning for 30 epochs on 207 triplets
          pooled from 3 GOES-16 days, evaluated on a 4th, fully disjoint day never seen during
          training. Fine-tuning improves PSNR and SSIM over the zero-shot baseline; LPIPS
          increases marginally, reflecting a small perceptual-versus-pixel trade-off rather than
          a uniform improvement across all metrics.
        </p>
        <ResultsTable
          columns={["Method", "PSNR (dB)", "SSIM", "LPIPS"]}
          rows={finetuneFullScale.map((r) => ({
            key: r.method,
            cells: [r.method, r.psnr.toFixed(2), r.ssim.toFixed(4), r.lpips.toFixed(4)],
          }))}
        />
        <p className="caveat">
          Executed in <code>notebooks/finetune_on_colab.ipynb</code> on a Colab T4 GPU.
        </p>
      </section>

      <section className="section--tight">
        <h2>Patch-size ablation</h2>
        <p className="prose">
          FILM&apos;s PSNR and SSIM remain nearly constant across a 16&times; range in patch area
          (128px to 512px), indicating that the model does not depend on additional spatial
          context. Farneback improves modestly with more context but remains well below FILM at
          every patch size.
        </p>
        <ResultsTable
          columns={["Size", "Method", "PSNR (dB)", "SSIM", "LPIPS"]}
          rows={patchSizeAblation.map((r) => ({
            key: `${r.size}-${r.method}`,
            cells: [r.size, r.method, r.psnr.toFixed(2), r.ssim.toFixed(3), r.lpips.toFixed(3)],
          }))}
        />
      </section>
    </div>
  );
}
