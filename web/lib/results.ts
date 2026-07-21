// Every number here mirrors the repo root README.md's "Status" section and
// docs/PLAN.md, both generated from real evaluation runs over real GOES-16
// satellite data (see src/eval/*.py). This file is the frontend's single
// source of truth for those numbers -- update it alongside README.md/PLAN.md
// whenever a pipeline run produces new results, rather than letting the two
// drift apart.

export type MethodResult = {
  method: string;
  psnr: number;
  ssim: number;
  lpips: number;
};

export type StratifiedSubset = {
  subset: "calm" | "cyclone";
  results: MethodResult[];
};

// src/eval/evaluate_stratified.py, run against the small-scale fine-tuned
// checkpoint (52 calm / 28 cyclone triplets).
export const stratifiedResults: StratifiedSubset[] = [
  {
    subset: "calm",
    results: [
      { method: "Farneback", psnr: 31.44, ssim: 0.823, lpips: 0.103 },
      { method: "FILM", psnr: 36.88, ssim: 0.917, lpips: 0.041 },
    ],
  },
  {
    subset: "cyclone",
    results: [
      { method: "Farneback", psnr: 24.98, ssim: 0.729, lpips: 0.125 },
      { method: "FILM", psnr: 32.05, ssim: 0.932, lpips: 0.044 },
    ],
  },
];

// Zero-shot pretrained FILM vs. Farneback, before any satellite-specific
// fine-tuning -- src/eval/evaluate_film.py / evaluate_baseline.py.
export const zeroShotResults: MethodResult[] = [
  { method: "Farneback (classical)", psnr: 24.4, ssim: 0.57, lpips: NaN },
  { method: "FILM (pretrained, zero-shot)", psnr: 30.7, ssim: 0.87, lpips: NaN },
];

// src/deep/finetune_film.py proof-of-concept: 46 real triplets from one
// contiguous 8-hour GOES-16 window, split chronologically into a 38-triplet
// fine-tuning pool and an 8-triplet held-out test set.
export const finetuneProofOfConcept: MethodResult[] = [
  { method: "Farneback (classical)", psnr: 22.6, ssim: 0.46, lpips: NaN },
  { method: "FILM, pretrained (zero-shot)", psnr: 27.8, ssim: 0.8, lpips: NaN },
  { method: "FILM, fine-tuned (5 epochs, 31 triplets, CPU)", psnr: 28.7, ssim: 0.82, lpips: NaN },
];

// src/eval/ablate_patch_size.py, run on 46 real GOES-16 triplets against
// the pretrained checkpoint.
export type PatchSizeRow = {
  size: number;
  method: string;
  psnr: number;
  ssim: number;
  lpips: number;
};

export const patchSizeAblation: PatchSizeRow[] = [
  { size: 128, method: "Farneback", psnr: 22.8, ssim: 0.537, lpips: 0.181 },
  { size: 128, method: "FILM", psnr: 31.6, ssim: 0.886, lpips: 0.062 },
  { size: 256, method: "Farneback", psnr: 23.73, ssim: 0.558, lpips: 0.186 },
  { size: 256, method: "FILM", psnr: 31.53, ssim: 0.875, lpips: 0.084 },
  { size: 512, method: "Farneback", psnr: 24.65, ssim: 0.602, lpips: 0.184 },
  { size: 512, method: "FILM", psnr: 31.89, ssim: 0.886, lpips: 0.09 },
];

export const headlineStats = [
  { value: "36.9 dB", label: "FILM PSNR, calm weather" },
  { value: "32.1 dB", label: "FILM PSNR, cyclone" },
  { value: "6.46 dB", label: "Farneback's calm→cyclone drop" },
  { value: "4.83 dB", label: "FILM's calm→cyclone drop" },
];

// This single real cyclone triplet's Farneback/FILM predictions, rendered
// via src/baseline/farneback_interpolate.py and src/deep/film_interpolate.py
// against models/film_net_finetuned.pt -- see web/public/demo/.
export const demoTripletMetrics = {
  farneback: { psnr: 25.23, ssim: 0.741, lpips: 0.121 },
  film: { psnr: 32.13, ssim: 0.933, lpips: 0.039 },
};
