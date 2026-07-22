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
      <h1>Calm weather holds up for both. Cyclones don&apos;t.</h1>
      <p className="lede">
        Every number below comes from real GOES-16 radiance data pulled from NOAA&apos;s public
        archive — no synthetic or held-in data anywhere in this pipeline.
      </p>

      <section className="section--tight">
        <h2>Calm vs. cyclone (headline result)</h2>
        <p className="prose">
          Farneback drops 6.46 dB PSNR going from calm to cyclone conditions; FILM drops only
          4.56 dB and its SSIM/LPIPS barely move — the degrade-sharply-vs-hold-up split the ISRO
          problem statement predicts.
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
          Run against the full-scale Colab fine-tuned checkpoint (52 calm / 28 cyclone triplets).
        </p>
      </section>

      <section className="section--tight">
        <h2>Zero-shot transfer</h2>
        <p className="prose">
          Running the pretrained FILM checkpoint as-is — zero satellite-specific training —
          already outperforms the classical baseline, confirming the architecture transfers to
          satellite imagery before any fine-tuning investment.
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
          46 real triplets from a contiguous 8-hour GOES-16 window, split chronologically into a
          38-triplet fine-tuning pool and an 8-triplet held-out test set excluded from
          fine-tuning entirely.
        </p>
        <ResultsTable
          columns={["Method", "PSNR (dB)", "SSIM"]}
          rows={finetuneProofOfConcept.map((r) => ({
            key: r.method,
            cells: [r.method, r.psnr.toFixed(1), r.ssim.toFixed(2)],
          }))}
        />
        <p className="caveat">
          Small-scale (31 training triplets, 5 epochs on CPU) — see the full-scale GPU run below.
        </p>
      </section>

      <section className="section--tight">
        <h2>Full-scale fine-tuning (Colab GPU)</h2>
        <p className="prose">
          30 epochs on 207 triplets pooled from 3 GOES-16 days, evaluated on a 4th day never seen
          during fine-tuning — the core contribution. Fine-tuning improves PSNR/SSIM over
          zero-shot on this fully disjoint test day; LPIPS ticks up slightly, a real (small)
          perceptual-vs-pixel trade-off rather than a straight win on every metric.
        </p>
        <ResultsTable
          columns={["Method", "PSNR (dB)", "SSIM", "LPIPS"]}
          rows={finetuneFullScale.map((r) => ({
            key: r.method,
            cells: [r.method, r.psnr.toFixed(2), r.ssim.toFixed(4), r.lpips.toFixed(4)],
          }))}
        />
        <p className="caveat">
          Run for real in <code>notebooks/finetune_on_colab.ipynb</code> on a Colab T4 GPU.
        </p>
      </section>

      <section className="section--tight">
        <h2>Patch-size ablation</h2>
        <p className="prose">
          FILM&apos;s PSNR/SSIM barely move across a 16× range in patch area (128px → 512px) — it
          isn&apos;t relying on extra spatial context. Farneback improves somewhat with more
          context but stays well below FILM at every size.
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
