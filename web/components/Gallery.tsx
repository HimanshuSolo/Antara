"use client";

import { useEffect, useState } from "react";
import BeforeAfterSlider from "@/components/BeforeAfterSlider";
import {
  fetchGallery,
  generateGalleryItem,
  type GalleryItem,
  type GenerateResult,
} from "@/lib/galleryApi";
import { LIVE_API_URL } from "@/lib/liveApi";

type CardStatus = "idle" | "loading" | "error" | "done";

function GalleryCard({ item }: { item: GalleryItem }) {
  const [status, setStatus] = useState<CardStatus>("idle");
  const [result, setResult] = useState<GenerateResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const onGenerate = async () => {
    setStatus("loading");
    setError(null);
    try {
      const data = await generateGalleryItem(item.id);
      setResult(data);
      setStatus("done");
    } catch (e) {
      setError(e instanceof Error ? e.message : "Generation failed.");
      setStatus("error");
    }
  };

  return (
    <div className="gallery-card">
      <div className="gallery-card__label">
        <h3>{item.label}</h3>
        <span className="gallery-card__subset">{item.subset}</span>
      </div>

      {status !== "done" && (
        <div className="live-sequence">
          <div className="live-frame live-frame--real live-frame--t1">
            <div className="live-frame__label">
              <span>T&minus;1 (real)</span>
            </div>
            <div className="live-frame__imgwrap">
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img src={item.frame_prev} alt={`${item.label} at t-1`} />
            </div>
          </div>
          <span className="live-sequence__arrow" aria-hidden="true">
            &hellip;
          </span>
          <div className="live-frame live-frame--real live-frame--t2">
            <div className="live-frame__label">
              <span>T+1 (real)</span>
            </div>
            <div className="live-frame__imgwrap">
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img src={item.frame_next} alt={`${item.label} at t+1`} />
            </div>
          </div>
        </div>
      )}

      {status !== "done" && (
        <>
          <button
            className="btn btn--secondary"
            style={{ marginTop: "1.25rem" }}
            onClick={onGenerate}
            disabled={status === "loading"}
          >
            {status === "loading" ? "Generating…" : "Generate middle frame"}
          </button>

          {status === "loading" && (
            <p className="caveat" style={{ marginTop: "1rem" }}>
              Running Farneback and FILM on this pair &mdash; typically a few seconds on CPU.
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
          <div className="live-sequence" key={result.model}>
            <div className="live-frame live-frame--real live-frame--t1">
              <div className="live-frame__label">
                <span>T&minus;1 (real)</span>
              </div>
              <div className="live-frame__imgwrap">
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img src={item.frame_prev} alt={`${item.label} at t-1`} />
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
              </div>
              <div className="live-frame__imgwrap">
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img src={item.frame_next} alt={`${item.label} at t+1`} />
              </div>
            </div>
          </div>

          <div className="stat-grid" style={{ marginTop: "1.5rem" }}>
            <div className="stat">
              <div className="stat__value">{result.film_psnr.toFixed(2)} dB</div>
              <div className="stat__label">FILM PSNR (vs. real t)</div>
            </div>
            <div className="stat">
              <div className="stat__value">{result.farneback_psnr.toFixed(2)} dB</div>
              <div className="stat__label">Farneback PSNR (vs. real t)</div>
            </div>
            <div className="stat">
              <div className="stat__value">{result.film_ssim.toFixed(3)}</div>
              <div className="stat__label">FILM SSIM</div>
            </div>
            <div className="stat">
              <div className="stat__value">{result.processing_seconds.toFixed(1)}s</div>
              <div className="stat__label">pipeline time</div>
            </div>
          </div>

          <p className="prose" style={{ marginTop: "1.25rem" }}>
            This triplet has a real ground-truth middle frame, unlike the live pipeline above
            where the true midpoint does not yet exist. The figures below are therefore
            genuine accuracy measurements rather than a qualitative estimate.
          </p>

          <div className="section--tight">
            <BeforeAfterSlider
              leftSrc={result.farneback_mid}
              rightSrc={result.film_mid}
              leftLabel="Farneback"
              rightLabel="FILM"
              alt={`Synthesized frame for ${item.label}`}
            />
          </div>
        </>
      )}
    </div>
  );
}

export default function Gallery() {
  const [items, setItems] = useState<GalleryItem[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchGallery()
      .then(setItems)
      .catch((e) =>
        setError(e instanceof Error ? e.message : "Could not reach the gallery API."),
      );
  }, []);

  if (error) {
    return (
      <div className="caveat" style={{ marginTop: "1.5rem" }}>
        {error}
        <br />
        The gallery API is not running or is not reachable at{" "}
        <code>{LIVE_API_URL}</code>. Start it with{" "}
        <code>.venv/bin/uvicorn src.api.live:app --reload --port 8000</code> from the
        repository root.
      </div>
    );
  }

  if (!items) {
    return (
      <p className="prose" style={{ marginTop: "1.5rem" }}>
        Loading the gallery&hellip;
      </p>
    );
  }

  return (
    <div className="gallery-grid">
      {items.map((item) => (
        <GalleryCard key={item.id} item={item} />
      ))}
    </div>
  );
}
