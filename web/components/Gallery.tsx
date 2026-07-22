"use client";

import { useEffect, useState } from "react";
import BeforeAfterSlider from "@/components/BeforeAfterSlider";
import {
  fetchGallery,
  generateGalleryItem,
  generateGalleryLoop,
  generateGalleryReport,
  type GalleryItem,
  type GenerateResult,
  type LoopResult,
  type ReportResult,
} from "@/lib/galleryApi";
import { LIVE_API_URL } from "@/lib/liveApi";

type CardStatus = "idle" | "loading" | "error" | "done";

function GalleryCard({ item }: { item: GalleryItem }) {
  const [status, setStatus] = useState<CardStatus>("idle");
  const [result, setResult] = useState<GenerateResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const [loopStatus, setLoopStatus] = useState<CardStatus>("idle");
  const [loop, setLoop] = useState<LoopResult | null>(null);
  const [loopError, setLoopError] = useState<string | null>(null);

  const [reportStatus, setReportStatus] = useState<CardStatus>("idle");
  const [report, setReport] = useState<ReportResult | null>(null);
  const [reportError, setReportError] = useState<string | null>(null);

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

  const onGenerateLoop = async () => {
    setLoopStatus("loading");
    setLoopError(null);
    try {
      const data = await generateGalleryLoop(item.id);
      setLoop(data);
      setLoopStatus("done");
    } catch (e) {
      setLoopError(e instanceof Error ? e.message : "Loop generation failed.");
      setLoopStatus("error");
    }
  };

  const onGenerateReport = async () => {
    setReportStatus("loading");
    setReportError(null);
    try {
      const data = await generateGalleryReport(item.id);
      setReport(data);
      setReportStatus("done");
    } catch (e) {
      setReportError(e instanceof Error ? e.message : "Report generation failed.");
      setReportStatus("error");
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

          <div className="section--tight">
            <p className="prose">
              An analyst archiving this comparison &mdash; or attaching it to an incident
              report &mdash; needs something more portable than a live web page. This renders
              the images and accuracy figures above into a single-page PDF.
            </p>

            {reportStatus !== "done" && (
              <button
                className="btn btn--secondary"
                onClick={onGenerateReport}
                disabled={reportStatus === "loading"}
              >
                {reportStatus === "loading" ? "Generating report…" : "Generate PDF report"}
              </button>
            )}

            {reportStatus === "error" && (
              <div className="caveat" style={{ marginTop: "1rem" }}>
                {reportError}
              </div>
            )}

            {reportStatus === "done" && report && (
              <div style={{ marginTop: "0.5rem" }}>
                <a
                  className="btn btn--secondary"
                  style={{ display: "inline-block" }}
                  href={report.report_pdf}
                  download={`${item.id}-report.pdf`}
                >
                  Download PDF report
                </a>
                <p className="caveat" style={{ marginTop: "0.75rem" }}>
                  Generated {report.generated_at}.
                </p>
              </div>
            )}
          </div>

          <div className="section--tight">
            <p className="prose">
              A single synthesized frame is useful for measurement, but a forecaster tracking
              a storm actually watches a satellite loop. Generating several evenly-spaced
              FILM frames instead of one turns this real frame gap into a smoother,
              higher-effective-frame-rate motion loop.
            </p>

            {loopStatus !== "done" && (
              <button
                className="btn btn--secondary"
                onClick={onGenerateLoop}
                disabled={loopStatus === "loading"}
              >
                {loopStatus === "loading" ? "Generating loop…" : "Generate satellite loop"}
              </button>
            )}

            {loopStatus === "loading" && (
              <p className="caveat" style={{ marginTop: "1rem" }}>
                Running FILM at 5 evenly-spaced intermediate times between t&minus;1 and
                t+1 &mdash; several forward passes, so this takes a bit longer than a single
                frame.
              </p>
            )}

            {loopStatus === "error" && (
              <div className="caveat" style={{ marginTop: "1rem" }}>
                {loopError}
              </div>
            )}

            {loopStatus === "done" && loop && (
              <div style={{ marginTop: "1rem" }}>
                <div className="live-frame__imgwrap" style={{ maxWidth: 360 }}>
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img src={loop.loop_gif} alt={`Satellite motion loop for ${item.label}`} />
                </div>
                <p className="caveat" style={{ marginTop: "0.75rem" }}>
                  {loop.num_frames}&times; FILM-interpolated frames between the two real
                  scans, generated in {loop.processing_seconds.toFixed(1)}s.
                </p>
                {loop.loop_mp4 ? (
                  <a
                    className="btn btn--secondary"
                    style={{ marginTop: "0.5rem", display: "inline-block" }}
                    href={loop.loop_mp4}
                    download={`${item.id}-satellite-loop.mp4`}
                  >
                    Download MP4
                  </a>
                ) : (
                  <p className="caveat" style={{ marginTop: "0.5rem" }}>
                    MP4 download unavailable &mdash; the API server has no ffmpeg installed.
                  </p>
                )}
              </div>
            )}
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
