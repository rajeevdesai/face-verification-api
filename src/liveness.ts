/**
 * Passive liveness / anti-spoofing via MiniFASNetV2 (Silent-Face).
 *
 * Crops an expanded region around the face, runs the 80×80 anti-spoofing model,
 * and returns a single liveness probability in [0, 1].
 *
 * LIMITATIONS (see README → Open Risks):
 *   - Detects PRINT attacks only; screen/video replay is scored as live.
 */
import * as ort from 'onnxruntime-web';
import type { NormalizedLandmark } from '@mediapipe/tasks-vision';

const INPUT_SIZE = 80;

/**
 * Silent-Face Anti-Spoofing crop: expanded bounding box around face landmarks.
 *
 * Scale factor 2.7 matches MiniFASNetV2 training crop. The model needs context
 * beyond the tight face region to detect texture/reflection artifacts.
 */
export function cropForLiveness(
  source: ImageData,
  landmarks: NormalizedLandmark[],
  scale = 2.7,
): ImageData {
  const { width, height } = source;

  // Bounding box from normalized landmarks → pixel space
  let minX = Infinity,
    maxX = -Infinity,
    minY = Infinity,
    maxY = -Infinity;
  for (const lm of landmarks) {
    if (lm.x < minX) minX = lm.x;
    if (lm.x > maxX) maxX = lm.x;
    if (lm.y < minY) minY = lm.y;
    if (lm.y > maxY) maxY = lm.y;
  }

  const faceW = (maxX - minX) * width;
  const faceH = (maxY - minY) * height;
  const cx = ((minX + maxX) / 2) * width;
  const cy = ((minY + maxY) / 2) * height;

  // Expand to a square, then scale
  const side = Math.max(faceW, faceH) * scale;
  const halfSide = side / 2;

  const x0 = Math.round(Math.max(0, cx - halfSide));
  const y0 = Math.round(Math.max(0, cy - halfSide));
  const x1 = Math.round(Math.min(width, cx + halfSide));
  const y1 = Math.round(Math.min(height, cy + halfSide));

  const cropW = x1 - x0;
  const cropH = y1 - y0;

  // Crop then resize to INPUT_SIZE×INPUT_SIZE
  const srcCanvas = new OffscreenCanvas(width, height);
  (srcCanvas.getContext('2d') as OffscreenCanvasRenderingContext2D).putImageData(source, 0, 0);

  const out = new OffscreenCanvas(INPUT_SIZE, INPUT_SIZE);
  const ctx = out.getContext('2d') as OffscreenCanvasRenderingContext2D;
  ctx.drawImage(srcCanvas, x0, y0, cropW, cropH, 0, 0, INPUT_SIZE, INPUT_SIZE);

  return ctx.getImageData(0, 0, INPUT_SIZE, INPUT_SIZE);
}

/** NCHW Float32, BGR, values in [0, 1]. garciafido/minifasnet-v2-anti-spoofing-onnx spec. */
function preprocess(crop: ImageData): Float32Array {
  const { data } = crop;
  const size = INPUT_SIZE * INPUT_SIZE;
  const tensor = new Float32Array(3 * size);

  // Aligned channel writes — keep the column layout.
  // prettier-ignore
  for (let i = 0; i < size; i++) {
    tensor[i]          = data[i * 4 + 2] / 255; // B
    tensor[size + i]   = data[i * 4 + 1] / 255; // G
    tensor[size*2 + i] = data[i * 4]     / 255; // R
  }

  return tensor;
}

/**
 * Run MiniFASNetV2 inference on an 80×80 crop.
 *
 * The model emits 3-class logits (confirmed by inspecting the shipped .onnx: the
 * raw output sums far from 1). We softmax them and return the live-class probability.
 *
 * Live is index 2 — determined EMPIRICALLY: a genuine face scores ~0.99 at index 2.
 * The garciafido/minifasnet-v2-anti-spoofing-onnx model card claims index 0 is live;
 * that labeling is wrong for the shipped weights. Indices 0 and 1 are the spoof
 * classes (print / replay); their exact order is not separately verified.
 */
export async function scoreLiveness(
  session: ort.InferenceSession,
  crop: ImageData,
): Promise<number> {
  const input = preprocess(crop);
  const inputName = session.inputNames[0];
  const tensor = new ort.Tensor('float32', input, [1, 3, INPUT_SIZE, INPUT_SIZE]);
  const outputs = await session.run({ [inputName]: tensor });
  const outputName = session.outputNames[0];
  const data = outputs[outputName].data as Float32Array;

  // Logits → softmax; live is class 2 (see header — determined empirically).
  const e = Array.from(data).map(Math.exp);
  const sum = e[0] + e[1] + e[2];
  return e[2] / sum;
}
