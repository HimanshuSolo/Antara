import type { Metadata } from "next";
import Gallery from "@/components/Gallery";
import LivePipeline from "@/components/LivePipeline";

export const metadata: Metadata = {
  title: "Live — Antara",
};

export default function LivePage() {
  return (
    <div className="container section">
      <div className="section-label">Live pipeline</div>
      <h1>Live inference on real-time satellite data</h1>
      <p className="lede">
        Every other page on this site is static, with all figures and images computed in
        advance. This page is the exception: it calls a local API that polls NOAA&apos;s public 
        archive for the two most recently published GOES-19 band 13 scans, and automatically runs 
        the Farneback baseline and the fine-tuned FILM model on them as soon as a new pair becomes available.
      </p>

      <LivePipeline />

      <div className="section--tight" style={{ marginTop: "3.5rem" }}>
        <div className="section-label">Gallery</div>
        <h2>Curated examples: on-demand frame generation</h2>
        <p className="prose">
          These (t&minus;1, t+1) pairs are curated from real GOES-16 events: Hurricane
          Milton&apos;s eyewall in 2024, and calm, off-season weather for comparison, served by{" "}
          <code>src/api/gallery.py</code>. No computation occurs until the Generate button is
          clicked, which triggers the real Farneback and FILM pipeline for that pair. Unlike the
          live pipeline above, these triplets have a real ground-truth middle frame available on
          disk, so each result includes genuine PSNR, SSIM, and LPIPS accuracy figures rather
          than a qualitative estimate.
        </p>
        <Gallery />
      </div>
    </div>
  );
}
