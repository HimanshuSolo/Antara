// The only page in this site that talks to a server at runtime -- every
// other page is static, numbers baked in ahead of time by the Python
// pipeline. This one calls src/api/live.py, which runs that same pipeline
// (Farneback + FILM) against whichever two GOES-16 scans were published
// most recently, so there's a genuine gap being filled, not a canned one.

export const LIVE_API_URL =
  process.env.NEXT_PUBLIC_LIVE_API_URL ?? "http://localhost:8000";

export type LiveResult = {
  prev_time: string;
  next_time: string;
  cadence_minutes: number;
  frame_prev: string;
  frame_next: string;
  farneback_mid: string;
  film_mid: string;
  processing_seconds: number;
  model: string;
};

export async function fetchLiveResult(): Promise<LiveResult> {
  const res = await fetch(`${LIVE_API_URL}/api/live`);
  if (!res.ok) {
    const body = await res.json().catch(() => null);
    throw new Error(body?.detail ?? `Request failed (${res.status})`);
  }
  return res.json();
}
