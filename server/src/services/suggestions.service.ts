/**
 * Console's starter-chip suggestions.
 *
 * Grouping the whole index by question (`buildSuggestionIndex`) means
 * scanning every chunk in the store — cheap once, wasteful if it happened on
 * every page load. This caches that grouped index and only re-scans when it
 * goes stale, while `pickSuggestions` (a plain in-memory sample) still runs
 * fresh on every call, so the chips shown to a user are different each time
 * without the index being rebuilt each time.
 */
import { logger } from '../utils/logger.js';
import { now } from '../utils/async.js';
import { getVectorStore } from '../rag/vector/index.js';
import {
  buildSuggestionIndex,
  pickSuggestions,
  type SuggestionIndex,
} from './dataset.service.js';
import type { QuerySuggestion } from '@goarag/shared';

/** How long the grouped index stays valid before a re-scan. The index only
    changes when someone re-runs the indexer, so this is generous. */
const CACHE_TTL_MS = 10 * 60_000;

let cached: SuggestionIndex | null = null;
let cachedAt = 0;

async function getSuggestionIndex(): Promise<SuggestionIndex> {
  if (cached && now() - cachedAt < CACHE_TTL_MS) return cached;

  const store = await getVectorStore();
  const chunks: Array<{ metadata: import('@goarag/shared').ChunkMetadata }> = [];
  for await (const batch of store.scrollAll(1_000)) {
    for (const hit of batch) chunks.push({ metadata: hit.metadata });
  }

  cached = buildSuggestionIndex(chunks);
  cachedAt = now();
  logger.debug({ questions: cached.size }, 'Suggestion index rebuilt');
  return cached;
}

export async function getSuggestions(count: number): Promise<QuerySuggestion[]> {
  const index = await getSuggestionIndex();
  return pickSuggestions(index, count);
}

/** Force a rebuild on next call — used after a fresh index run. */
export function invalidateSuggestionCache(): void {
  cached = null;
}
