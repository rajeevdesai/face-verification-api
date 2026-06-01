/**
 * Model lifecycle. Lazily loads and caches the three models (FaceLandmarker,
 * recognition, liveness) as module singletons, and hands them to the pipeline
 * via getModels(). Call loadModels() once at startup.
 */
import { FaceLandmarker, FilesetResolver } from '@mediapipe/tasks-vision';
import * as ort from 'onnxruntime-web';
import type { ModelConfig } from './types.js';

const DEFAULT_MEDIAPIPE_WASM = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.21/wasm';
const DEFAULT_ORT_WASM = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.0/dist/';

let faceLandmarker: FaceLandmarker | null = null;
let session: ort.InferenceSession | null = null;
let livenessSession: ort.InferenceSession | null = null;

/**
 * Load (or reuse) all three models: FaceLandmarker, MobileFaceNet, and Silent-Face liveness.
 *
 * Must be called before compareFaces(). Safe to call multiple times — subsequent
 * calls are no-ops if models are already loaded.
 *
 * Common failure: onnxruntime-web cannot find its .wasm blobs.
 * Set wasmBasePath to wherever ort .wasm files are served, or rely on the CDN default.
 */
export async function loadModels(config: ModelConfig): Promise<void> {
  const {
    faceLandmarkerPath = 'models/face_landmarker.task',
    recognitionModelPath = 'models/mobilefacenet.onnx',
    livenessModelPath,
    wasmBasePath,
    warmup = true,
  } = config;

  ort.env.wasm.wasmPaths = wasmBasePath ?? DEFAULT_ORT_WASM;
  ort.env.wasm.numThreads = 1; // single-threaded — no COOP/COEP headers required

  if (!faceLandmarker) {
    const filesetResolver = await FilesetResolver.forVisionTasks(
      wasmBasePath ?? DEFAULT_MEDIAPIPE_WASM,
    );
    faceLandmarker = await FaceLandmarker.createFromOptions(filesetResolver, {
      baseOptions: { modelAssetPath: faceLandmarkerPath, delegate: 'GPU' },
      runningMode: 'IMAGE',
      numFaces: 10,
      outputFaceBlendshapes: false,
      outputFacialTransformationMatrixes: false,
    });
  }

  if (!session) {
    session = await ort.InferenceSession.create(recognitionModelPath, {
      executionProviders: ['wasm'],
    });
  }

  if (!livenessSession) {
    livenessSession = await ort.InferenceSession.create(livenessModelPath, {
      executionProviders: ['wasm'],
    });
  }

  if (warmup) {
    const dummy112 = new ort.Tensor('float32', new Float32Array(3 * 112 * 112), [1, 3, 112, 112]);
    await session.run({ [session.inputNames[0]]: dummy112 });

    const dummy80 = new ort.Tensor('float32', new Float32Array(3 * 80 * 80), [1, 3, 80, 80]);
    await livenessSession.run({ [livenessSession.inputNames[0]]: dummy80 });
  }
}

export function getModels(): {
  faceLandmarker: FaceLandmarker;
  session: ort.InferenceSession;
  livenessSession: ort.InferenceSession;
} {
  if (!faceLandmarker || !session || !livenessSession) {
    throw new Error('Models not loaded. Call loadModels() first.');
  }
  return { faceLandmarker, session, livenessSession };
}

/** Reset singletons (useful in tests). */
export function _resetModels(): void {
  faceLandmarker?.close();
  faceLandmarker = null;
  session = null;
  livenessSession = null;
}
