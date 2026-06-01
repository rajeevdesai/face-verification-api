import { describe, it, expect } from 'vitest';
import { assertSafeImageSource } from '../src/image.js';

describe('assertSafeImageSource', () => {
  it('allows the supported schemes', () => {
    for (const url of [
      'https://example.com/a.jpg',
      'http://example.com/a.jpg',
      'data:image/png;base64,AAAA',
      'blob:https://example.com/uuid',
    ]) {
      expect(() => assertSafeImageSource(url)).not.toThrow();
    }
  });

  it('allows scheme-less relative / same-origin paths', () => {
    for (const p of ['/models/a.jpg', './a.png', '../b.png', 'a.jpg']) {
      expect(() => assertSafeImageSource(p)).not.toThrow();
    }
  });

  it('rejects dangerous or unexpected schemes', () => {
    for (const url of ['file:///etc/passwd', 'javascript:alert(1)', 'ftp://h/x.png']) {
      expect(() => assertSafeImageSource(url)).toThrow(/Unsupported image source scheme/);
    }
  });
});
