import type { Metadata } from "next";
import Gallery from "@/components/Gallery";
import LivePipeline from "@/components/LivePipeline";

export const metadata: Metadata = {
  title: "Live — Antara",
};

export default function LivePage() {
  return (
    <div className="container section">
      <div className="section-label">Live monitoring</div>
      <h1>Watching GOES-19 for the next scan, live.</h1>
      <p className="lede">
        Every other page on this site is static — numbers and images baked in ahead of
        time. This one calls a small local API (<code>src/api/live.py</code>) that polls
        for the two most recently published GOES-19 band 13 scans from NOAA&apos;s public
        archive, then runs the real Farneback baseline and the fine-tuned FILM model on
        them — automatically, as soon as a new pair lands.
      </p>

      <LivePipeline />

      <p className="caveat">
        Requires the API server running locally: <code>.venv/bin/uvicorn src.api.live:app
        --reload --port 8000</code> from the repo root (needs <code>models/</code> to
        contain a FILM checkpoint — see the root README&apos;s Setup section). See{" "}
        <code>web/README.md</code> for details.
      </p>

      <div className="section--tight" style={{ marginTop: "3.5rem" }}>
        <div className="section-label">Gallery</div>
        <h2>Real cyclone and calm-weather pairs, generated on demand.</h2>
        <p className="prose">
          These (t&minus;1, t+1) pairs are curated from real GOES-16 events — Hurricane
          Milton&apos;s eyewall in 2024, and calm off-season weather for comparison — via{" "}
          <code>src/api/gallery.py</code>. Nothing runs until you click Generate: that
          request triggers the real Farneback + FILM pipeline for that pair. Unlike the
          pipeline above, these triplets have a real ground-truth middle frame on disk, so
          each result comes back with genuine PSNR/SSIM/LPIPS accuracy numbers, not just a
          plausible-looking guess.
        </p>
        <Gallery />
      </div>
    </div>
  );
}
