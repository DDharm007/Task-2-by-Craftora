/**
 * Vector store factory.
 *
 * `VECTOR_STORE=auto` (the default) probes Qdrant and silently falls back to
 * the embedded driver when it is unreachable, so a fresh clone runs without
 * Docker. `qdrant` and `embedded` pin the choice explicitly — use `qdrant` in
 * production so a misconfigured URL fails loudly instead of quietly writing to
 * a local file.
 */
import path from 'node:path';
import { config } from '../../config/env.js';
import { logger } from '../../utils/logger.js';
import { errors } from '../../utils/errors.js';
import { EmbeddedVectorStore } from './embedded-store.js';
import { QdrantVectorStore } from './qdrant-store.js';
import type { VectorStore } from './types.js';

let instance: VectorStore | null = null;
let resolving: Promise<VectorStore> | null = null;

async function create(): Promise<VectorStore> {
  const { driver } = config.vectorStore;

  if (driver === 'embedded') {
    return new EmbeddedVectorStore(config.vectorStore.collection, config.vectorStore.embeddedPath);
  }

  const qdrant = new QdrantVectorStore();

  if (driver === 'qdrant') {
    const reachable = await qdrant.ping();
    if (!reachable) {
      throw errors.vectorStore(
        `VECTOR_STORE=qdrant but Qdrant is unreachable at ${config.vectorStore.qdrantUrl}. ` +
          `Start it with \`docker compose up -d qdrant\`, or set VECTOR_STORE=auto to use the embedded store.`,
      );
    }
    logger.info({ url: config.vectorStore.qdrantUrl }, 'Using Qdrant vector store');
    return qdrant;
  }

  // auto
  if (await qdrant.ping()) {
    logger.info({ url: config.vectorStore.qdrantUrl }, 'Qdrant reachable — using Qdrant vector store');
    return qdrant;
  }

  logger.warn(
    { url: config.vectorStore.qdrantUrl, path: config.vectorStore.embeddedPath },
    'Qdrant unreachable — falling back to the embedded vector store (set VECTOR_STORE=qdrant to require Qdrant)',
  );
  return new EmbeddedVectorStore(config.vectorStore.collection, config.vectorStore.embeddedPath);
}

/** Resolve the process-wide vector store, creating it once. */
export function getVectorStore(): Promise<VectorStore> {
  if (instance) return Promise.resolve(instance);
  if (!resolving) {
    resolving = create()
      .then((store) => {
        instance = store;
        return store;
      })
      .catch((error) => {
        resolving = null;
        throw error;
      });
  }
  return resolving;
}

/** Flush and release the store — called on shutdown and by the indexer. */
export async function closeVectorStore(): Promise<void> {
  if (instance) {
    await instance.close();
    instance = null;
    resolving = null;
  }
}

/** Where the BM25 model lives on disk, kept next to the embedded store data. */
export function bm25ModelPath(): string {
  return path.join(config.vectorStore.embeddedPath, `${config.vectorStore.collection}.bm25.json`);
}

export { EmbeddedVectorStore, QdrantVectorStore };
export type { SearchFilter, SearchHit, StoreStats, VectorRecord, VectorStore } from './types.js';
