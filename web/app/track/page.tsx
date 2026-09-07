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
        Locating a tropical cyclone&apos;s eye, using a classical heuristic and a small trained CNN, supervised by real IBTrACS
        best-track positions rather than manual labels. Running both detectors on a
        Farneback- or FILM-synthesized middle frame and comparing to the same detector&apos;s
        output on the real frame measures something PSNR/SSIM/LPIPS don&apos;t: whether
        interpolation keeps the storm where it actually was, not just whether the pixels look
        similar.
      </p>

      <EyeTracker />
    </div>
  );
}
