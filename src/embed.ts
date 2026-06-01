/**
 * Face embedding + similarity. Runs the recognition model (MobileFaceNet /
 * facex_nano) on an aligned 112×112 crop and compares L2-normalized embeddings
 * by cosine distance.
 */
import * as ort from 'onnxruntime-web';

/**
 * Preprocess a 112×112 aligned face crop for MobileFaceNet inference.
 *
 * Layout: NCHW [1, 3, 112, 112], Float32, RGB, values in [-1, 1].
 *
 * IMPORTANT: Verify the exact input layout from model metadata before deploying.
 * Some facex variants may expect HWC [1, 112, 112, 3]. If cosine distances are
 * unreasonably high for the same person, swap to HWC layout here.
 */
function preprocess(imageData: ImageData): Float32Array {
  const { data } = imageData; // RGBA, uint8
  const size = 112 * 112;
  const tensor = new Float32Array(3 * size);

  // Aligned channel writes — keep the column layout.
  // prettier-ignore
  for (let i = 0; i < size; i++) {
    tensor[i]          = data[i * 4]     / 127.5 - 1.0; // R
    tensor[size + i]   = data[i * 4 + 1] / 127.5 - 1.0; // G
    tensor[size*2 + i] = data[i * 4 + 2] / 127.5 - 1.0; // B
  }

  return tensor;
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

/** Cosine distance between two L2-normalized embeddings. Range [0, 2]; 0 = identical. */
export function cosineDistance(a: Float32Array, b: Float32Array): number {
  let dot = 0;
  for (let i = 0; i < a.length; i++) dot += a[i] * b[i];
  return 1 - dot;
}

/**
 * Run MobileFaceNet inference on a 112×112 aligned face crop.
 * Returns a L2-normalized embedding (256-D for facex_nano; dimension is model-dependent).
 */
export async function embed(
  session: ort.InferenceSession,
  alignedFace: ImageData,
): Promise<Float32Array> {
  const input = preprocess(alignedFace);
  const inputName = session.inputNames[0];
  const tensor = new ort.Tensor('float32', input, [1, 3, 112, 112]);
  const outputs = await session.run({ [inputName]: tensor });
  const outputName = session.outputNames[0];
  const raw = outputs[outputName].data as Float32Array;
  return l2normalize(new Float32Array(raw));
}
