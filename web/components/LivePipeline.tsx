"use client";

import { useState } from "react";
import BeforeAfterSlider from "@/components/BeforeAfterSlider";
import { fetchLiveResult, LIVE_API_URL, type LiveResult } from "@/lib/liveApi";

type Status = "idle" | "loading" | "error" | "done";

function formatUtc(iso: string): string {
  return new Date(iso).toUTCString().replace(" GMT", " UTC");
}

export default function LivePipeline() {
  const [status, setStatus] = useState<Status>("idle");
  const [result, setResult] = useState<LiveResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function run() {
    setStatus("loading");
    setError(null);
    try {
      const data = await fetchLiveResult();
      setResult(data);
      setStatus("done");
    } catch (e) {
      setError(
        e instanceof Error
          ? e.message
          : "Could not reach the live pipeline API.",
      );
      setStatus("error");
    }
  }

  return (
    <div>
      <div className="btn-row" style={{ marginTop: "1.5rem" }}>
        <button
          className="btn btn--primary"
          onClick={run}
          disabled={status === "loading"}
        >
          {status === "loading" ? "Running pipeline…" : "Run pipeline on latest scan"}
        </button>
      </div>

      {status === "loading" && (
        <p className="prose" style={{ marginTop: "1.25rem" }}>
          Downloading the latest published GOES-16 band 13 scans and running Farneback +
          FILM on them — this fetches real data and runs a real model, so it can take
          anywhere from a few seconds to about a minute.
        </p>
      )}

      {status === "error" && (
        <div className="caveat" style={{ marginTop: "1.25rem" }}>
          {error}
          <br />
          The live API isn&apos;t running or isn&apos;t reachable at{" "}
          <code>{LIVE_API_URL}</code>. Start it with{" "}
          <code>.venv/bin/uvicorn src.api.live:app --reload --port 8000</code>{" "}
          from the repo root.
        </div>
      )}

      {status === "done" && result && (
        <div style={{ marginTop: "2rem" }}>
          <div className="stat-grid">
            <div className="stat">
              <div className="stat__value">{formatUtc(result.prev_time)}</div>
              <div className="stat__label">frame t&minus;1 (real)</div>
            </div>
            <div className="stat">
              <div className="stat__value">{formatUtc(result.next_time)}</div>
              <div className="stat__label">frame t+1 (real)</div>
            </div>
            <div className="stat">
              <div className="stat__value">{result.cadence_minutes.toFixed(1)} min</div>
              <div className="stat__label">native scan cadence</div>
            </div>
            <div className="stat">
              <div className="stat__value">{result.processing_seconds.toFixed(1)}s</div>
              <div className="stat__label">pipeline time ({result.model})</div>
            </div>
          </div>

          <p className="prose" style={{ marginTop: "1.5rem" }}>
            There is no real frame between the two above yet — that gap is exactly what
            gets synthesized below. Drag to compare the classical Farneback prediction
            against the fine-tuned FILM prediction for the midpoint.
          </p>

          <div className="section--tight">
            <BeforeAfterSlider
              leftSrc={result.farneback_mid}
              rightSrc={result.film_mid}
              leftLabel="Farneback"
              rightLabel="FILM"
              alt="Synthesized frame between the two most recent real scans"
            />
          </div>

          <div className="card-grid">
            <div className="card">
              <h3>t&minus;1 (real)</h3>
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                src={result.frame_prev}
                alt="Most recent real scan before the gap"
                style={{ width: "100%", border: "1px solid var(--border)" }}
              />
            </div>
            <div className="card">
              <h3>t+1 (real)</h3>
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                src={result.frame_next}
                alt="Most recent real scan after the gap"
                style={{ width: "100%", border: "1px solid var(--border)" }}
              />
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
