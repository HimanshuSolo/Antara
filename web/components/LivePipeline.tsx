"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import BeforeAfterSlider from "@/components/BeforeAfterSlider";
import { fetchLiveResult, LIVE_API_URL, type LiveResult } from "@/lib/liveApi";

// GOES-19 full-disk scans publish roughly every 10 minutes. Polling this
// often is cheap: the API caches by scan-pair, so most polls just confirm
// nothing new has landed yet rather than re-running Farneback/FILM.
const POLL_INTERVAL_MS = 45_000;

type Status = "idle" | "loading" | "error" | "done";

function formatUtc(iso: string): string {
  return new Date(iso).toUTCString().replace(" GMT", " UTC");
}

function formatClock(date: Date): string {
  return date.toLocaleTimeString([], { hour12: false });
}

export default function LivePipeline() {
  const [status, setStatus] = useState<Status>("idle");
  const [result, setResult] = useState<LiveResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [monitoring, setMonitoring] = useState(true);
  const [lastChecked, setLastChecked] = useState<Date | null>(null);
  const [justUpdated, setJustUpdated] = useState(false);
  const resultRef = useRef<LiveResult | null>(null);

  const check = useCallback(async () => {
    if (!resultRef.current) setStatus("loading");
    try {
      const data = await fetchLiveResult();
      const isNewPair = data.next_time !== resultRef.current?.next_time;
      resultRef.current = data;
      setResult(data);
      setStatus("done");
      setError(null);
      if (isNewPair) {
        setJustUpdated(true);
        setTimeout(() => setJustUpdated(false), 4000);
      }
    } catch (e) {
      setError(
        e instanceof Error ? e.message : "Could not reach the live pipeline API.",
      );
      if (!resultRef.current) setStatus("error");
    } finally {
      setLastChecked(new Date());
    }
  }, []);

  // Runs an initial check on mount, then keeps polling while monitoring is
  // on. The mount check fires exactly once regardless of `monitoring` so
  // pausing right away still shows the current scan pair.
  useEffect(() => {
    const id = setTimeout(check, 0);
    return () => clearTimeout(id);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    if (!monitoring) return;
    const id = setInterval(check, POLL_INTERVAL_MS);
    return () => clearInterval(id);
  }, [monitoring, check]);

  return (
    <div>
      <div className="monitor-status" style={{ marginTop: "1.5rem" }}>
        <span className={`monitor-dot ${monitoring ? "monitor-dot--live" : ""}`} />
        <span>
          {monitoring
            ? `Live — checking for a new GOES-19 scan pair every ${POLL_INTERVAL_MS / 1000}s`
            : "Monitoring paused"}
        </span>
        {lastChecked && <span>· last checked {formatClock(lastChecked)}</span>}
      </div>

      <div className="btn-row" style={{ marginTop: "1rem" }}>
        <button
          className="btn btn--secondary"
          onClick={() => setMonitoring((m) => !m)}
        >
          {monitoring ? "Pause monitoring" : "Resume monitoring"}
        </button>
        <button
          className="btn btn--secondary"
          onClick={check}
          disabled={status === "loading"}
        >
          Check now
        </button>
      </div>

      {status === "loading" && (
        <p className="prose" style={{ marginTop: "1.25rem" }}>
          Downloading the latest published GOES-19 band 13 scans and running Farneback +
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

      {status === "done" && error && (
        <div className="caveat" style={{ marginTop: "1.25rem" }}>
          Last check failed ({error}) — still showing the most recent successful
          result below. Retrying automatically.
        </div>
      )}

      {status === "done" && result && (
        <div style={{ marginTop: "2rem" }}>
          {justUpdated && (
            <p className="caveat" style={{ marginTop: 0, marginBottom: "1.5rem" }}>
              New scan pair just landed — frame regenerated at {formatClock(new Date())}.
            </p>
          )}
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
            There is no real frame between the two below yet — that gap is exactly what
            gets synthesized in the middle panel.
          </p>

          <div className="live-sequence" key={result.next_time}>
            <div className="live-frame live-frame--real live-frame--t1">
              <div className="live-frame__label">
                <span>T&minus;1 (real)</span>
                <span>{formatClock(new Date(result.prev_time))}</span>
              </div>
              <div className="live-frame__imgwrap">
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img src={result.frame_prev} alt="Most recent real scan before the gap" />
              </div>
            </div>

            <span className="live-sequence__arrow" aria-hidden="true">
              &rarr;
            </span>

            <div className="live-frame">
              <div className="live-frame__label">
                <span>Synthesized (FILM)</span>
                <span className="live-frame__generating-label">GENERATING</span>
              </div>
              <div className="live-frame__imgwrap live-frame__result">
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img src={result.film_mid} alt="ML-synthesized middle frame" />
                <div className="live-frame__scan" aria-hidden="true" />
              </div>
            </div>

            <span className="live-sequence__arrow" aria-hidden="true">
              &larr;
            </span>

            <div className="live-frame live-frame--real live-frame--t2">
              <div className="live-frame__label">
                <span>T+1 (real)</span>
                <span>{formatClock(new Date(result.next_time))}</span>
              </div>
              <div className="live-frame__imgwrap">
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img src={result.frame_next} alt="Most recent real scan after the gap" />
              </div>
            </div>
          </div>

          <p className="prose" style={{ marginTop: "1.5rem" }}>
            Drag to compare the classical Farneback prediction against the fine-tuned FILM
            prediction for that same midpoint.
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
        </div>
      )}
    </div>
  );
}
