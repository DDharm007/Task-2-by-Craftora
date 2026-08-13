/**
 * Lifecycle for the process-wide BM25 model.
 *
 * BM25 needs corpus-global statistics (document frequency, average length)
 * that only exist once the whole corpus is known. The indexer fits the model
 * and saves it; the server loads it at query time to build query vectors.
 *
 * If the snapshot is missing — a fresh container mounted against an already
 * populated Qdrant, say — we rebuild it by scrolling the store. That keeps the
 * vector store authoritative and means a lost snapshot degrades startup time
 * rather than breaking keyword search.
 */
import { logger } from '../../utils/logger.js';
import { Bm25Model } from './bm25.js';
import { bm25ModelPath, getVectorStore } from './index.js';

let model: Bm25Model | null = null;
let loading: Promise<Bm25Model> | null = null;

async function loadOrRebuild(): Promise<Bm25Model> {
  const filePath = bm25ModelPath();

  const saved = await Bm25Model.load(filePath);
  if (saved && !saved.isEmpty) {
    logger.info({ terms: saved.size, documents: saved.docCount }, 'Loaded BM25 model from disk');
    return saved;
  }

  logger.warn('BM25 snapshot missing — rebuilding from the vector store');
  const store = await getVectorStore();
  const texts: string[] = [];
  try {
    for await (const batch of store.scrollAll(512)) {
      for (const hit of batch) texts.push(hit.text);
    }
  } catch (error) {
    logger.error({ error: (error as Error).message }, 'Could not scroll the vector store to rebuild BM25');
  }

  const rebuilt = new Bm25Model();
  rebuilt.fit(texts);
  if (!rebuilt.isEmpty) {
    await rebuilt.save(filePath).catch((error) => {
      logger.warn({ error: (error as Error).message }, 'Could not persist rebuilt BM25 model');
    });
    logger.info({ terms: rebuilt.size, documents: rebuilt.docCount }, 'Rebuilt BM25 model');
  }
  return rebuilt;
}

/** The process-wide BM25 model, loaded once. */
export function getBm25Model(): Promise<Bm25Model> {
  if (model) return Promise.resolve(model);
  if (!loading) {
    loading = loadOrRebuild()
      .then((loaded) => {
        model = loaded;
        return loaded;
      })
      .catch((error) => {
        loading = null;
        throw error;
      });
  }
  return loading;
}

/** Replace the cached model — called by the indexer after a fresh fit. */
export function setBm25Model(next: Bm25Model): void {
  model = next;
  loading = Promise.resolve(next);
}

/** Drop the cached model so the next read reloads from disk. */
export function resetBm25Model(): void {
  model = null;
  loading = null;
}
