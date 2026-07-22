import type { Metadata } from "next";
import LivePipeline from "@/components/LivePipeline";

export const metadata: Metadata = {
  title: "Live — Antara",
};

export default function LivePage() {
  return (
    <div className="container section">
      <div className="section-label">Live pipeline</div>
      <h1>Run it on whatever GOES-16 just published.</h1>
      <p className="lede">
        Every other page on this site is static — numbers and images baked in ahead of
        time. This one calls a small local API (<code>src/api/live.py</code>) that fetches
        the two most recently published GOES-16 band 13 scans from NOAA&apos;s public
        archive right now, then runs the real Farneback baseline and the fine-tuned FILM
        model on them, live.
      </p>

      <LivePipeline />

      <p className="caveat">
        Requires the API server running locally: <code>.venv/bin/uvicorn src.api.live:app
        --reload --port 8000</code> from the repo root (needs <code>models/</code> to
        contain a FILM checkpoint — see the root README&apos;s Setup section). See{" "}
        <code>web/README.md</code> for details.
      </p>
    </div>
  );
}
