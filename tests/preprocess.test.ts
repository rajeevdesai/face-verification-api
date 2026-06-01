import { describe, it, expect } from 'vitest';
import { buildInputTensor, tensorDims, triplet, validateInputSize } from '../src/preprocess.js';
import type { ResolvedTensorSpec } from '../src/preprocess.js';

/** Build an ImageData from per-pixel [r,g,b] triples (alpha forced to 255). */
function img(pixels: [number, number, number][]): ImageData {
  const data = new Uint8ClampedArray(pixels.length * 4);
  pixels.forEach(([r, g, b], i) => {
    data[i * 4] = r;
    data[i * 4 + 1] = g;
    data[i * 4 + 2] = b;
    data[i * 4 + 3] = 255;
  });
  return { width: 1, height: pixels.length, data } as unknown as ImageData;
}

const spec = (o: Partial<ResolvedTensorSpec> = {}): ResolvedTensorSpec => ({
  inputSize: 1,
  layout: 'NCHW',
  channelOrder: 'RGB',
  mean: [0, 0, 0],
  std: [1, 1, 1],
  ...o,
});

describe('triplet', () => {
  it('expands a scalar', () => expect(triplet(5)).toEqual([5, 5, 5]));
  it('passes a triplet through', () => expect(triplet([1, 2, 3])).toEqual([1, 2, 3]));
});

describe('tensorDims', () => {
  it('NCHW', () => expect(tensorDims('NCHW', 80)).toEqual([1, 3, 80, 80]));
  it('NHWC', () => expect(tensorDims('NHWC', 80)).toEqual([1, 80, 80, 3]));
});

describe('validateInputSize', () => {
  it('accepts valid sizes', () => {
    expect(validateInputSize(80)).toBe(80);
    expect(validateInputSize(112)).toBe(112);
  });
  it('rejects non-integers, non-positive, and absurd sizes', () => {
    for (const n of [0, -1, 1.5, 9000, NaN, Infinity]) {
      expect(() => validateInputSize(n)).toThrow(/Invalid model inputSize/);
    }
  });
});

describe('buildInputTensor', () => {
  it('RGB passes channels through in order', () => {
    const t = buildInputTensor(img([[10, 20, 30]]), spec());
    expect(Array.from(t)).toEqual([10, 20, 30]);
  });

  it('BGR swaps channel order', () => {
    const t = buildInputTensor(img([[10, 20, 30]]), spec({ channelOrder: 'BGR' }));
    expect(Array.from(t)).toEqual([30, 20, 10]);
  });

  it('applies (pixel - mean) / std per channel', () => {
    // pixels are integers (Uint8ClampedArray); use 128 to keep the math exact.
    const t = buildInputTensor(
      img([[128, 128, 128]]),
      spec({ mean: [128, 128, 128], std: [64, 64, 64] }),
    );
    expect(Array.from(t)).toEqual([0, 0, 0]);
  });

  it('NCHW is planar (all R, all G, all B)', () => {
    const t = buildInputTensor(
      img([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
        [10, 11, 12],
      ]),
      spec({ inputSize: 2, layout: 'NCHW' }),
    );
    expect(Array.from(t)).toEqual([1, 4, 7, 10, 2, 5, 8, 11, 3, 6, 9, 12]);
  });

  it('NHWC interleaves channels per pixel', () => {
    const t = buildInputTensor(
      img([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
        [10, 11, 12],
      ]),
      spec({ inputSize: 2, layout: 'NHWC' }),
    );
    expect(Array.from(t)).toEqual([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
  });
});
