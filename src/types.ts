/** Public types for the face-recognition API: results, options, flags, config. */

export type FraudFlag =
  | 'baseline_missing'
  | 'baseline_ambiguous'
  | 'face_missing'
  | 'multiple_faces'
  | 'identity_mismatch'
  | 'low_confidence'
  | 'liveness_fail';

export interface CompareResult {
  match: boolean;
  /** 0–1; margin below threshold. NOT a calibrated probability. */
  confidence: number;
  flags: FraudFlag[];
  details: {
    baselineFacesFound: number;
    currentFacesFound: number;
    cosineDistance: number;
    threshold: number;
    /** Liveness score [0–1] for the current image. Absent if liveness model not loaded or check skipped. */
    livenessScore?: number;
  };
}

export interface CompareOptions {
  /** Cosine distance threshold. Default: 0.5 (uncalibrated — run calibration before production use). */
  threshold?: number;
  /** Liveness score threshold for the current image. Score must exceed this to pass. Default: 0.5. */
  livenessThreshold?: number;
  /**
   * Whether to run liveness check on the current image. Default: true if liveness model is loaded.
   * Set to false to skip liveness even when the model is available.
   */
  checkLiveness?: boolean;
}

export interface ModelConfig {
  /** Path or URL to face_landmarker.task (MediaPipe). */
  faceLandmarkerPath?: string;
  /** Path or URL to mobilefacenet.onnx (Apache-2.0 weights). */
  recognitionModelPath?: string;
  /** Path or URL to Silent-Face anti-spoofing ONNX model (MiniFASNetV2). */
  livenessModelPath: string;
  /** Base URL/path for WASM blobs (MediaPipe + onnxruntime-web). */
  wasmBasePath?: string;
  /** Run a dummy inference after load to avoid first-call latency. Default: true. */
  warmup?: boolean;
}
