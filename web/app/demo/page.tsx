import type { Metadata } from "next";
import BeforeAfterSlider from "@/components/BeforeAfterSlider";
import { demoTripletMetrics } from "@/lib/results";

export const metadata: Metadata = {
  title: "Demo — Antara",
};

export default function DemoPage() {
  const { farneback, film } = demoTripletMetrics;

  return (
    <div className="container section">
      <div className="section-label">Interactive demo</div>
      <h1>Classical vs. learned interpolation</h1>
      <p className="lede">
        Drag the slider to compare the classical Farneback prediction against the fine-tuned
        FILM prediction for the same real, held-out cyclone frame. Both predictions are
        generated using only the real frames immediately before and after; neither method has
        access to this middle frame during inference.
      </p>

      <div className="section--tight">
        <BeforeAfterSlider
          leftSrc="/demo/farneback.png"
          rightSrc="/demo/film.png"
          leftLabel="Farneback"
          rightLabel="FILM"
          alt="Synthesized cyclone frame"
        />
      </div>

      <div className="card-grid">
        <div className="card">
          <h3>Farneback (classical)</h3>
          <div className="prose">
            <p>PSNR: {farneback.psnr.toFixed(2)} dB</p>
            <p>SSIM: {farneback.ssim.toFixed(3)}</p>
            <p>LPIPS: {farneback.lpips.toFixed(3)}</p>
          </div>
        </div>
        <div className="card">
          <h3>FILM (fine-tuned)</h3>
          <div className="prose">
            <p>PSNR: {film.psnr.toFixed(2)} dB</p>
            <p>SSIM: {film.ssim.toFixed(3)}</p>
            <p>LPIPS: {film.lpips.toFixed(3)}</p>
          </div>
        </div>
        <div className="card">
          <h3>Ground truth</h3>
          <p className="prose">
            The real middle frame, held out during inference and used only to compute the
            metrics above.
          </p>
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src="/demo/ground-truth.png"
            alt="Real held-out middle frame"
            style={{ width: "100%", border: "1px solid var(--border)", marginTop: "0.75rem" }}
          />
        </div>
      </div>

      <p className="caveat">
        Generated using <code>src/baseline/farneback_interpolate.py</code> and{" "}
        <code>src/deep/film_interpolate.py</code> on a real cyclone triplet from{" "}
        <code>data/processed/triplets_cyclone</code>.
      </p>
    </div>
  );
}
