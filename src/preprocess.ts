/**
 * Internal image→tensor preprocessing, shared by the recognition and liveness
 * models so both honour the same configurable layout / channel order / mean-std.
 */
import type { TensorLayout, ChannelOrder } from './types.js';

/** Fully-resolved (no optional fields) preprocessing spec. */
export interface ResolvedTensorSpec {
  inputSize: number;
  layout: TensorLayout;
  channelOrder: ChannelOrder;
  mean: [number, number, number];
  std: [number, number, number];
}

/** onnxruntime Tensor dims for a square input of the given layout. */
export function tensorDims(layout: TensorLayout, size: number): number[] {
  return layout === 'NHWC' ? [1, size, size, 3] : [1, 3, size, size];
}

/** Expand a scalar mean/std to a per-channel triplet. */
export function triplet(v: number | [number, number, number]): [number, number, number] {
  return typeof v === 'number' ? [v, v, v] : v;
}

/**
 * Convert an RGBA ImageData (already sized to spec.inputSize²) into a Float32
 * input tensor.
 *
 * Per output channel position p: value = (pixel - mean[p]) / std[p], where the
 * source byte for p is chosen by channelOrder (RGB → R,G,B; BGR → B,G,R), and
 * element placement follows layout (NCHW vs NHWC). mean/std are indexed by the
 * output channel position, i.e. they follow channelOrder.
 */
export function buildInputTensor(img: ImageData, spec: ResolvedTensorSpec): Float32Array {
  const { data } = img; // RGBA, uint8
  const size = spec.inputSize * spec.inputSize;
  const out = new Float32Array(3 * size);
  const srcOffset = spec.channelOrder === 'BGR' ? [2, 1, 0] : [0, 1, 2];
  const nchw = spec.layout !== 'NHWC';
  const { mean, std } = spec;

  for (let i = 0; i < size; i++) {
    for (let p = 0; p < 3; p++) {
      const v = (data[i * 4 + srcOffset[p]] - mean[p]) / std[p];
      out[nchw ? p * size + i : i * 3 + p] = v;
    }
  }
  return out;
}
