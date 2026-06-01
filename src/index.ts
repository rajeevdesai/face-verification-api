/** Public entry point — re-exports the supported API surface. */
export { loadModels } from './models.js';
export { compareFaces, DEFAULT_THRESHOLD } from './compare.js';
export type { CompareResult, CompareOptions, ModelConfig, FraudFlag } from './types.js';
