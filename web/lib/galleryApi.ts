// The gallery's curated (t-1, t+1) pairs come from real GOES-16 events --
// some from Hurricane Milton's 2024 cyclone-centered eval set, some from
// the calm off-season set -- served instantly by src/api/gallery.py.
// Generating the middle frame is a separate, on-demand call: the real
// Farneback + FILM pipeline runs live when the user clicks "Generate",
// exactly like the /live page, except these triplets also have a real
// ground-truth middle frame on disk, so the response includes real
// PSNR/SSIM/LPIPS for both methods.

import { LIVE_API_URL } from "@/lib/liveApi";

export type GalleryItem = {
  id: string;
  label: string;
  subset: "cyclone" | "calm";
  frame_prev: string;
  frame_next: string;
};

export type GenerateResult = {
  id: string;
  farneback_mid: string;
  film_mid: string;
  ground_truth: string;
  farneback_psnr: number;
  farneback_ssim: number;
  farneback_lpips: number;
  film_psnr: number;
  film_ssim: number;
  film_lpips: number;
  processing_seconds: number;
  model: string;
};

// A real-world usage endpoint: assembles t-1, N FILM-interpolated
// intermediate frames, and t+1 into a single looping sequence -- a higher
// effective frame-rate satellite motion loop, the format forecasters
// actually watch to track storm motion, rather than one static frame.
// loop_gif is for the inline animated preview; loop_mp4 (H.264, smaller,
// shareable) is for download, and is null if the server has no ffmpeg.
export type LoopResult = {
  id: string;
  loop_gif: string;
  loop_mp4: string | null;
  num_frames: number;
  processing_seconds: number;
  model: string;
};

async function unwrap<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const body = await res.json().catch(() => null);
    throw new Error(body?.detail ?? `Request failed (${res.status})`);
  }
  return res.json();
}

export async function fetchGallery(): Promise<GalleryItem[]> {
  return unwrap(await fetch(`${LIVE_API_URL}/api/gallery`));
}

export async function generateGalleryItem(id: string): Promise<GenerateResult> {
  return unwrap(await fetch(`${LIVE_API_URL}/api/gallery/${id}/generate`, { method: "POST" }));
}

export async function generateGalleryLoop(id: string, numFrames = 5): Promise<LoopResult> {
  return unwrap(
    await fetch(`${LIVE_API_URL}/api/gallery/${id}/loop?num_frames=${numFrames}`, {
      method: "POST",
    }),
  );
}
