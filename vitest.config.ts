import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    coverage: {
      provider: 'v8',
      reporter: ['text', 'text-summary', 'html'],
      include: ['src/**/*.ts'],
      // Excluded from the coverage denominator:
      //  - index.ts: re-exports only
      //  - types.ts: types only, no runtime
      //  - detect.ts / models.ts: browser-only orchestration (MediaPipe + model
      //    loading) exercised by the browser integration suite, not Node units
      exclude: ['src/index.ts', 'src/types.ts', 'src/detect.ts', 'src/models.ts'],
      // Floors that gate regressions. Kept a few points below current with margin
      // for browser-only functions (warpFace, cropForLiveness, toImageData, embed
      // run-path) that can't execute under Node and so stay uncovered here.
      thresholds: {
        statements: 80,
        branches: 85,
        functions: 70,
        lines: 80,
      },
    },
  },
});
