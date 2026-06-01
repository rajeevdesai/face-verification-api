import { describe, it, expect, vi } from 'vitest';

// embed.ts imports onnxruntime-web at module load; these pure fns don't use it, so stub it out.
vi.mock('onnxruntime-web', () => ({ Tensor: class {} }));

import { l2normalize, cosineDistance } from '../src/embed.js';

function norm(v: Float32Array): number {
  let s = 0;
  for (const x of v) s += x * x;
  return Math.sqrt(s);
}

const unit = (v: number[]) => l2normalize(Float32Array.from(v));

describe('l2normalize', () => {
  it('scales any vector to unit length', () => {
    const out = l2normalize(new Float32Array([3, 4]));
    expect(norm(out)).toBeCloseTo(1, 6);
    expect(out[0]).toBeCloseTo(0.6, 6);
    expect(out[1]).toBeCloseTo(0.8, 6);
  });

  it('mutates in place and returns the same array', () => {
    const v = new Float32Array([1, 2, 2]);
    const out = l2normalize(v);
    expect(out).toBe(v);
    expect(norm(v)).toBeCloseTo(1, 6);
  });

  it('leaves a zero vector untouched (no divide-by-zero)', () => {
    const out = l2normalize(new Float32Array([0, 0, 0]));
    expect(Array.from(out)).toEqual([0, 0, 0]);
    expect(Array.from(out).some(Number.isNaN)).toBe(false);
  });
});

describe('cosineDistance', () => {
  it('is ~0 for identical direction', () => {
    expect(cosineDistance(unit([1, 1, 1]), unit([2, 2, 2]))).toBeCloseTo(0, 6);
  });

  it('is ~1 for orthogonal vectors', () => {
    expect(cosineDistance(unit([1, 0]), unit([0, 1]))).toBeCloseTo(1, 6);
  });

  it('is ~2 for opposite vectors', () => {
    expect(cosineDistance(unit([1, 0]), unit([-1, 0]))).toBeCloseTo(2, 6);
  });
});
