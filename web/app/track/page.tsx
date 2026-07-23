import type { Metadata } from "next";
import EyeTracker from "@/components/EyeTracker";

export const metadata: Metadata = {
  title: "Track — Antara",
};

export default function TrackPage() {
  return (
    <div className="container section">
      <div className="section-label">Eye tracking</div>
      <h1>Does interpolation preserve the storm&apos;s position?</h1>
      <p className="lede">
        A separate application built on the same triplets as the rest of this project: locating
        a tropical cyclone&apos;s eye, using a classical heuristic (<code>src/baseline/eye_detect.py</code>)
        and a small trained CNN (<code>src/deep/eye_detect.py</code>), supervised by real IBTrACS
        best-track positions rather than manual labels. Running both detectors on a
        Farneback- or FILM-synthesized middle frame and comparing to the same detector&apos;s
        output on the real frame measures something PSNR/SSIM/LPIPS don&apos;t: whether
        interpolation keeps the storm where it actually was, not just whether the pixels look
        similar.
      </p>

      <EyeTracker />

      <p className="caveat" style={{ marginTop: "2rem" }}>
        Requires the API server to be running locally: run{" "}
        <code>.venv/bin/uvicorn src.api.live:app --reload --port 8000</code> from the repository
        root (<code>models/</code> must contain both a FILM checkpoint and an eye-detection
        checkpoint &mdash; run <code>.venv/bin/python -m src.deep.eye_detect_train</code> to
        produce the latter). The eye detector is trained on a single storm&apos;s (Hurricane
        Milton&apos;s) ~4.5-hour window, a proof of concept rather than a generalization test;
        see <code>src/deep/eye_detect_train.py</code> for the follow-up needed (pooling several
        storms) to make it one.
      </p>
    </div>
  );
}
