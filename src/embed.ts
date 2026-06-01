/**
 * Face embedding + similarity. Runs the recognition model on an aligned crop
 * and compares embeddings by the configured distance metric. Preprocessing and
 * metric are configurable to support bring-your-own recognition models.
 */
import * as ort from 'onnxruntime-web';
import type { RecognitionConfig, DistanceMetric } from './types.js';
import { buildInputTensor, tensorDims, triplet, type ResolvedTensorSpec } from './preprocess.js';

export interface ResolvedRecognitionConfig extends ResolvedTensorSpec {
  l2normalize: boolean;
  metric: DistanceMetric;
}

/** Defaults for the bundled facex_nano / MobileFaceNet weights. */
export const RECOGNITION_DEFAULTS: ResolvedRecognitionConfig = {
  inputSize: 112,
  layout: 'NCHW',
  channelOrder: 'RGB',
  mean: [127.5, 127.5, 127.5],
  std: [127.5, 127.5, 127.5],
  l2normalize: true,
  metric: 'cosine',
};

/** Merge a user RecognitionConfig over the defaults. */
export function resolveRecognitionConfig(c: RecognitionConfig = {}): ResolvedRecognitionConfig {
  return {
    inputSize: c.inputSize ?? RECOGNITION_DEFAULTS.inputSize,
    layout: c.layout ?? RECOGNITION_DEFAULTS.layout,
    channelOrder: c.channelOrder ?? RECOGNITION_DEFAULTS.channelOrder,
    mean: c.mean !== undefined ? triplet(c.mean) : RECOGNITION_DEFAULTS.mean,
    std: c.std !== undefined ? triplet(c.std) : RECOGNITION_DEFAULTS.std,
    l2normalize: c.l2normalize ?? RECOGNITION_DEFAULTS.l2normalize,
    metric: c.metric ?? RECOGNITION_DEFAULTS.metric,
  };
}

/** L2-normalize an embedding vector in-place. Returns the same array. */
export function l2normalize(v: Float32Array): Float32Array {
  let norm = 0;
  for (const x of v) norm += x * x;
  norm = Math.sqrt(norm);
  if (norm < 1e-10) return v;
  for (let i = 0; i < v.length; i++) v[i] /= norm;
  return v;
}

/** Cosine distance. Range [0, 2]; 0 = identical. Handles non-normalized inputs. */
export function cosineDistance(a: Float32Array, b: Float32Array): number {
  let dot = 0,
    na = 0,
    nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  const denom = Math.sqrt(na) * Math.sqrt(nb);
  return denom < 1e-10 ? 1 : 1 - dot / denom;
}

/** Euclidean (L2) distance between two embeddings. */
export function euclideanDistance(a: Float32Array, b: Float32Array): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) {
    const d = a[i] - b[i];
    s += d * d;
  }
  return Math.sqrt(s);
}

/** Distance under the configured metric. Lower = more similar. */
export function distance(
  a: Float32Array,
  b: Float32Array,
  metric: DistanceMetric = 'cosine',
): number {
  return metric === 'euclidean' ? euclideanDistance(a, b) : cosineDistance(a, b);
}

/**
 * Run the recognition model on an aligned face crop (sized cfg.inputSize²).
 * Returns the embedding, L2-normalized unless cfg.l2normalize is false.
 */
export async function embed(
  session: ort.InferenceSession,
  alignedFace: ImageData,
  cfg: ResolvedRecognitionConfig = RECOGNITION_DEFAULTS,
): Promise<Float32Array> {
  const input = buildInputTensor(alignedFace, cfg);
  const inputName = session.inputNames[0];
  const tensor = new ort.Tensor('float32', input, tensorDims(cfg.layout, cfg.inputSize));
  const outputs = await session.run({ [inputName]: tensor });
  const outputName = session.outputNames[0];
  const raw = new Float32Array(outputs[outputName].data as Float32Array);
  return cfg.l2normalize ? l2normalize(raw) : raw;
}
