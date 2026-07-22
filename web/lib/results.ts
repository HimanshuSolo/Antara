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

// src/eval/evaluate_stratified.py, run against the full-scale Colab
// fine-tuned checkpoint (52 calm / 28 cyclone triplets).
export const stratifiedResults: StratifiedSubset[] = [
  {
    subset: "calm",
    results: [
      { method: "Farneback", psnr: 31.44, ssim: 0.8227, lpips: 0.1029 },
      { method: "FILM", psnr: 36.62, ssim: 0.9117, lpips: 0.0472 },
    ],
  },
  {
    subset: "cyclone",
    results: [
      { method: "Farneback", psnr: 24.98, ssim: 0.7292, lpips: 0.125 },
      { method: "FILM", psnr: 32.06, ssim: 0.931, lpips: 0.0505 },
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

// notebooks/finetune_on_colab.ipynb, run for real on a Colab T4 GPU: 30
// epochs on 207 triplets pooled from 3 GOES-16 days, evaluated on a 4th day
// never seen during fine-tuning. The core contribution.
export const finetuneFullScale: MethodResult[] = [
  { method: "Farneback (classical)", psnr: 26.6, ssim: 0.6999, lpips: 0.1324 },
  { method: "FILM, pretrained (zero-shot)", psnr: 32.66, ssim: 0.9242, lpips: 0.0371 },
  { method: "FILM, fine-tuned (30 epochs, 207 triplets, GPU)", psnr: 33.25, ssim: 0.9314, lpips: 0.0595 },
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
  { value: "36.6 dB", label: "FILM PSNR, calm weather" },
  { value: "32.1 dB", label: "FILM PSNR, cyclone" },
  { value: "6.46 dB", label: "Farneback's calm→cyclone drop" },
  { value: "4.56 dB", label: "FILM's calm→cyclone drop" },
];

// This single real cyclone triplet's Farneback/FILM predictions, rendered
// via src/baseline/farneback_interpolate.py and src/deep/film_interpolate.py
// against models/film_net_finetuned.pt -- see web/public/demo/.
export const demoTripletMetrics = {
  farneback: { psnr: 25.23, ssim: 0.741, lpips: 0.121 },
  film: { psnr: 32.13, ssim: 0.933, lpips: 0.039 },
};
