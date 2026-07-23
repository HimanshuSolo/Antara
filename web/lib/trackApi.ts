// Cyclone eye-tracking: detects the storm eye's pixel location with both a
// classical heuristic and a trained CNN, on the real ground-truth middle
// frame and on the FILM/Farneback-synthesized frame from the same gallery
// triplet -- served by src/api/track.py, a separate endpoint from
// galleryApi.ts's /generate. Answers whether interpolation preserves where
// the storm actually was (the "drift" fields), not just how sharp the
// synthesized frame looks.

import { LIVE_API_URL } from "@/lib/liveApi";

export type TrackItem = {
  id: string;
  label: string;
  frame_prev: string;
  frame_next: string;
};

export type Detection = {
  row: number;
  col: number;
};

export type TrackResult = {
  id: string;
  ground_truth: Detection | null;
  classical_real: Detection;
  classical_farneback: Detection;
  classical_film: Detection;
  cnn_real: Detection;
  cnn_farneback: Detection;
  cnn_film: Detection;
  classical_accuracy_px: number | null;
  cnn_accuracy_px: number | null;
  classical_farneback_drift_px: number;
  classical_film_drift_px: number;
  cnn_farneback_drift_px: number;
  cnn_film_drift_px: number;
  annotated_real: string;
  annotated_farneback: string;
  annotated_film: string;
};

async function unwrap<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const body = await res.json().catch(() => null);
    throw new Error(body?.detail ?? `Request failed (${res.status})`);
  }
  return res.json();
}

export async function fetchTrackItems(): Promise<TrackItem[]> {
  return unwrap(await fetch(`${LIVE_API_URL}/api/track`));
}

export async function detectEye(id: string): Promise<TrackResult> {
  return unwrap(await fetch(`${LIVE_API_URL}/api/track/${id}/detect`, { method: "POST" }));
}
