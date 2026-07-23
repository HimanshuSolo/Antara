"use client";

import { useEffect, useState } from "react";
import { detectEye, fetchTrackItems, type TrackItem, type TrackResult } from "@/lib/trackApi";
import { LIVE_API_URL } from "@/lib/liveApi";

type CardStatus = "idle" | "loading" | "error" | "done";

function TrackCard({ item }: { item: TrackItem }) {
  const [status, setStatus] = useState<CardStatus>("idle");
  const [result, setResult] = useState<TrackResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const onDetect = async () => {
    setStatus("loading");
    setError(null);
    try {
      const data = await detectEye(item.id);
      setResult(data);
      setStatus("done");
    } catch (e) {
      setError(e instanceof Error ? e.message : "Eye detection failed.");
      setStatus("error");
    }
  };

  return (
    <div className="gallery-card">
      <div className="gallery-card__label">
        <h3>{item.label}</h3>
      </div>

      {status !== "done" && (
        <>
          <button
            className="btn btn--secondary"
            style={{ marginTop: "1.25rem" }}
            onClick={onDetect}
            disabled={status === "loading"}
          >
            {status === "loading" ? "Detecting…" : "Detect eye position"}
          </button>

          {status === "loading" && (
            <p className="caveat" style={{ marginTop: "1rem" }}>
              Running Farneback and FILM, then both eye detectors, on this triplet &mdash;
              typically a few seconds on CPU.
            </p>
          )}

          {status === "error" && (
            <div className="caveat" style={{ marginTop: "1rem" }}>
              {error}
            </div>
          )}
        </>
      )}

      {status === "done" && result && (
        <>
          <div className="live-sequence">
            <div className="live-frame">
              <div className="live-frame__label">
                <span>Real (ground truth)</span>
              </div>
              <div className="live-frame__imgwrap">
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img src={result.annotated_real} alt="Detected eye position on the real frame" />
              </div>
            </div>
            <span className="live-sequence__arrow" aria-hidden="true">
              &rarr;
            </span>
            <div className="live-frame">
              <div className="live-frame__label">
                <span>Farneback (synthesized)</span>
              </div>
              <div className="live-frame__imgwrap">
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  src={result.annotated_farneback}
                  alt="Detected eye position on the Farneback-synthesized frame"
                />
              </div>
            </div>
            <span className="live-sequence__arrow" aria-hidden="true">
              &rarr;
            </span>
            <div className="live-frame">
              <div className="live-frame__label">
                <span>FILM (synthesized)</span>
              </div>
              <div className="live-frame__imgwrap">
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img src={result.annotated_film} alt="Detected eye position on the FILM-synthesized frame" />
              </div>
            </div>
          </div>

          <p className="caveat" style={{ marginTop: "0.75rem" }}>
            Orange cross = classical detector, blue cross = trained CNN
            {result.ground_truth ? ", green cross = IBTrACS ground truth" : ""}.
          </p>

          <div className="stat-grid" style={{ marginTop: "1.5rem" }}>
            {result.classical_accuracy_px !== null && (
              <div className="stat">
                <div className="stat__value">{result.classical_accuracy_px.toFixed(1)}px</div>
                <div className="stat__label">classical accuracy (vs. ground truth)</div>
              </div>
            )}
            {result.cnn_accuracy_px !== null && (
              <div className="stat">
                <div className="stat__value">{result.cnn_accuracy_px.toFixed(1)}px</div>
                <div className="stat__label">CNN accuracy (vs. ground truth)</div>
              </div>
            )}
            <div className="stat">
              <div className="stat__value">{result.classical_farneback_drift_px.toFixed(1)}px</div>
              <div className="stat__label">classical drift, Farneback</div>
            </div>
            <div className="stat">
              <div className="stat__value">{result.classical_film_drift_px.toFixed(1)}px</div>
              <div className="stat__label">classical drift, FILM</div>
            </div>
            <div className="stat">
              <div className="stat__value">{result.cnn_farneback_drift_px.toFixed(1)}px</div>
              <div className="stat__label">CNN drift, Farneback</div>
            </div>
            <div className="stat">
              <div className="stat__value">{result.cnn_film_drift_px.toFixed(1)}px</div>
              <div className="stat__label">CNN drift, FILM</div>
            </div>
          </div>

          <p className="prose" style={{ marginTop: "1.25rem" }}>
            &ldquo;Drift&rdquo; is how far a detector&apos;s output moves when run on a synthesized
            middle frame instead of the real one &mdash; a lower number means interpolation
            preserved the storm&apos;s actual position more faithfully, independent of whether the
            detector itself is accurate.
          </p>
        </>
      )}
    </div>
  );
}

export default function EyeTracker() {
  const [items, setItems] = useState<TrackItem[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchTrackItems()
      .then(setItems)
      .catch((e) =>
        setError(e instanceof Error ? e.message : "Could not reach the tracking API."),
      );
  }, []);

  if (error) {
    return (
      <div className="caveat" style={{ marginTop: "1.5rem" }}>
        {error}
        <br />
        The tracking API is not running or is not reachable at{" "}
        <code>{LIVE_API_URL}</code>. Start it with{" "}
        <code>.venv/bin/uvicorn src.api.live:app --reload --port 8000</code> from the
        repository root.
      </div>
    );
  }

  if (!items) {
    return (
      <p className="prose" style={{ marginTop: "1.5rem" }}>
        Loading trackable triplets&hellip;
      </p>
    );
  }

  return (
    <div className="gallery-grid">
      {items.map((item) => (
        <TrackCard key={item.id} item={item} />
      ))}
    </div>
  );
}
