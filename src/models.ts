/**
 * Model lifecycle. Lazily loads and caches the models (FaceLandmarker,
 * recognition, optional liveness) as module singletons, along with their
 * resolved preprocessing configs, and hands them to the pipeline via getModels().
 * Call loadModels() once at startup.
 */
import { FaceLandmarker, FilesetResolver } from '@mediapipe/tasks-vision';
import * as ort from 'onnxruntime-web';
import type { ModelConfig } from './types.js';
import {
  resolveRecognitionConfig,
  RECOGNITION_DEFAULTS,
  type ResolvedRecognitionConfig,
} from './embed.js';
import {
  resolveLivenessConfig,
  LIVENESS_DEFAULTS,
  type ResolvedLivenessConfig,
} from './liveness.js';
import { buildInputTensor, tensorDims } from './preprocess.js';

const DEFAULT_MEDIAPIPE_WASM = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.21/wasm';
const DEFAULT_ORT_WASM = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.0/dist/';

let faceLandmarker: FaceLandmarker | null = null;
let session: ort.InferenceSession | null = null;
let livenessSession: ort.InferenceSession | null = null;
let recognitionConfig: ResolvedRecognitionConfig = RECOGNITION_DEFAULTS;
let livenessConfig: ResolvedLivenessConfig = LIVENESS_DEFAULTS;

/** Build a zeroed dummy crop for warmup at the given square size. */
function dummyCrop(size: number): ImageData {
  return { width: size, height: size, data: new Uint8ClampedArray(size * size * 4) } as ImageData;
}

/**
 * Load (or reuse) the models: FaceLandmarker, recognition, and optionally liveness.
 *
 * Omit livenessModelPath to disable liveness entirely. Must be called before
 * compareFaces(). Safe to call multiple times — subsequent calls are no-ops if
 * models are already loaded.
 *
 * Common failure: onnxruntime-web cannot find its .wasm blobs.
 * Set wasmBasePath to wherever ort .wasm files are served, or rely on the CDN default.
 */
export async function loadModels(config: ModelConfig = {}): Promise<void> {
  const {
    faceLandmarkerPath = 'models/face_landmarker.task',
    recognitionModelPath = 'models/mobilefacenet.onnx',
    livenessModelPath,
    wasmBasePath,
    warmup = true,
    recognition,
    liveness,
  } = config;

  recognitionConfig = resolveRecognitionConfig(recognition);
  livenessConfig = resolveLivenessConfig(liveness);

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

  if (!livenessSession && livenessModelPath) {
    livenessSession = await ort.InferenceSession.create(livenessModelPath, {
      executionProviders: ['wasm'],
    });
  }

  if (warmup) {
    const rc = recognitionConfig;
    const recInput = buildInputTensor(dummyCrop(rc.inputSize), rc);
    await session.run({
      [session.inputNames[0]]: new ort.Tensor(
        'float32',
        recInput,
        tensorDims(rc.layout, rc.inputSize),
      ),
    });

    if (livenessSession) {
      const lc = livenessConfig;
      const livInput = buildInputTensor(dummyCrop(lc.inputSize), lc);
      await livenessSession.run({
        [livenessSession.inputNames[0]]: new ort.Tensor(
          'float32',
          livInput,
          tensorDims(lc.layout, lc.inputSize),
        ),
      });
    }
  }
}

export function getModels(): {
  faceLandmarker: FaceLandmarker;
  session: ort.InferenceSession;
  livenessSession: ort.InferenceSession | null;
  recognitionConfig: ResolvedRecognitionConfig;
  livenessConfig: ResolvedLivenessConfig;
} {
  if (!faceLandmarker || !session) {
    throw new Error('Models not loaded. Call loadModels() first.');
  }
  return { faceLandmarker, session, livenessSession, recognitionConfig, livenessConfig };
}

/** Reset singletons (useful in tests). */
export function _resetModels(): void {
  faceLandmarker?.close();
  faceLandmarker = null;
  session = null;
  livenessSession = null;
  recognitionConfig = RECOGNITION_DEFAULTS;
  livenessConfig = LIVENESS_DEFAULTS;
}
