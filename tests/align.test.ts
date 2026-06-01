import { describe, it, expect } from 'vitest';
import { umeyama, extractFivePoints, ARCFACE_DST } from '../src/align.js';
import type { NormalizedLandmark } from '@mediapipe/tasks-vision';

describe('umeyama', () => {
  it('recovers identity transform', () => {
    const pts: [number, number][] = [
      [38.2946, 51.6963],
      [73.5318, 51.5014],
      [56.0252, 71.7366],
      [41.5493, 92.3655],
      [70.7299, 92.2041],
    ];
    const [a, b, tx, d, e, ty] = umeyama(pts, pts);
    expect(a).toBeCloseTo(1, 5);
    expect(b).toBeCloseTo(0, 5);
    expect(tx).toBeCloseTo(0, 4);
    expect(d).toBeCloseTo(0, 5);
    expect(e).toBeCloseTo(1, 5);
    expect(ty).toBeCloseTo(0, 4);
  });

  it('recovers known rotation + scale + translation', () => {
    const src: [number, number][] = [
      [10, 20],
      [30, 20],
      [20, 35],
      [15, 50],
      [25, 50],
    ];
    const angle = Math.PI / 6; // 30°
    const scale = 1.5;
    const tx = 10,
      ty = 20;
    const cos = Math.cos(angle),
      sin = Math.sin(angle);
    const dst: [number, number][] = src.map(([x, y]) => [
      scale * (cos * x - sin * y) + tx,
      scale * (sin * x + cos * y) + ty,
    ]);

    const [a, b, ctxOut, d, e, tyOut] = umeyama(src, dst);
    // a = scale*cos, b = -scale*sin, d = scale*sin, e = scale*cos
    expect(a).toBeCloseTo(scale * cos, 4);
    expect(b).toBeCloseTo(-scale * sin, 4);
    expect(d).toBeCloseTo(scale * sin, 4);
    expect(e).toBeCloseTo(scale * cos, 4);
    expect(ctxOut).toBeCloseTo(tx, 4);
    expect(tyOut).toBeCloseTo(ty, 4);
  });

  it('maps src points to dst points within 1px after transform', () => {
    const src: [number, number][] = [
      [10, 40],
      [50, 42],
      [30, 60],
      [15, 80],
      [45, 80],
    ];
    const angle = -0.3;
    const scale = 0.9;
    const cos = Math.cos(angle),
      sin = Math.sin(angle);
    const dst: [number, number][] = src.map(([x, y]) => [
      scale * (cos * x - sin * y) + 5,
      scale * (sin * x + cos * y) - 3,
    ]);

    const [a, b, tx, d, e, ty] = umeyama(src, dst);
    for (let i = 0; i < src.length; i++) {
      const px = a * src[i][0] + b * src[i][1] + tx;
      const py = d * src[i][0] + e * src[i][1] + ty;
      expect(px).toBeCloseTo(dst[i][0], 3);
      expect(py).toBeCloseTo(dst[i][1], 3);
    }
  });
});

describe('extractFivePoints', () => {
  /** Build a minimal landmark array of length n, all zeros except specified overrides. */
  function makeLandmarks(
    n: number,
    overrides: Record<number, [number, number]>,
  ): NormalizedLandmark[] {
    const lms: NormalizedLandmark[] = Array.from({ length: n }, () => ({ x: 0, y: 0, z: 0 }));
    for (const [idx, [x, y]] of Object.entries(overrides)) {
      lms[Number(idx)] = { x, y, z: 0 };
    }
    return lms;
  }

  it('uses iris indices for 478-landmark model', () => {
    const lms = makeLandmarks(478, {
      468: [0.3, 0.4], // left eye iris center
      473: [0.6, 0.4], // right eye iris center
      1: [0.5, 0.55], // nose
      61: [0.4, 0.7], // left mouth
      291: [0.6, 0.7], // right mouth
    });
    const pts = extractFivePoints(lms, 100, 100);
    expect(pts[0][0]).toBeCloseTo(30, 8); // left eye
    expect(pts[0][1]).toBeCloseTo(40, 8);
    expect(pts[1][0]).toBeCloseTo(60, 8); // right eye
    expect(pts[1][1]).toBeCloseTo(40, 8);
    expect(pts[2][0]).toBeCloseTo(50, 8); // nose
    expect(pts[2][1]).toBeCloseTo(55, 8);
    expect(pts[3][0]).toBeCloseTo(40, 8); // left mouth
    expect(pts[3][1]).toBeCloseTo(70, 8);
    expect(pts[4][0]).toBeCloseTo(60, 8); // right mouth
    expect(pts[4][1]).toBeCloseTo(70, 8);
  });

  it('falls back to eye-corner midpoints for 468-landmark model', () => {
    const lms = makeLandmarks(468, {
      33: [0.25, 0.4], // left eye left corner
      133: [0.35, 0.42], // left eye right corner
      263: [0.6, 0.4], // right eye left corner
      362: [0.7, 0.42], // right eye right corner
      1: [0.5, 0.55],
      61: [0.4, 0.7],
      291: [0.6, 0.7],
    });
    const pts = extractFivePoints(lms, 100, 100);
    // Left eye = midpoint(33, 133) = ([25+35]/2, [40+42]/2)
    expect(pts[0][0]).toBeCloseTo(30, 5);
    expect(pts[0][1]).toBeCloseTo(41, 5);
    // Right eye = midpoint(263, 362)
    expect(pts[1][0]).toBeCloseTo(65, 5);
    expect(pts[1][1]).toBeCloseTo(41, 5);
  });

  it('ARCFACE_DST has 5 points in expected range', () => {
    expect(ARCFACE_DST).toHaveLength(5);
    for (const [x, y] of ARCFACE_DST) {
      expect(x).toBeGreaterThan(0);
      expect(x).toBeLessThan(112);
      expect(y).toBeGreaterThan(0);
      expect(y).toBeLessThan(112);
    }
  });
});
