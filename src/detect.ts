/**
 * Face detection + landmarking via MediaPipe FaceLandmarker. Returns every
 * detected face with its 468/478 normalized landmarks, sorted largest-first so
 * callers can default to the most prominent face.
 */
import type { FaceLandmarker, NormalizedLandmark } from '@mediapipe/tasks-vision';
import { imageDataToCanvas } from './image.js';

export interface DetectedFace {
  /** Normalized [0,1] landmarks. 468 or 478 points depending on model variant. */
  landmarks: NormalizedLandmark[];
  /** Bounding box area in normalized units (for picking largest face). */
  boundingBoxArea: number;
}

/** Run FaceLandmarker and return all detected faces sorted largest-first. */
export function detectFaces(faceLandmarker: FaceLandmarker, imageData: ImageData): DetectedFace[] {
  const canvas = imageDataToCanvas(imageData);
  const result = faceLandmarker.detect(canvas);

  return result.faceLandmarks
    .map((landmarks) => ({
      landmarks,
      boundingBoxArea: computeBBoxArea(landmarks),
    }))
    .sort((a, b) => b.boundingBoxArea - a.boundingBoxArea);
}

function computeBBoxArea(landmarks: NormalizedLandmark[]): number {
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
  return (maxX - minX) * (maxY - minY);
}
