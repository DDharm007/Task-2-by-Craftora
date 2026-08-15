/**
 * The chunking pipeline.
 *
 * Composes the six strategies in `strategies.ts` into a single pass over a
 * document, producing retrieval-ready chunks with complete metadata.
 *
 * How the strategies combine, and why:
 *
 *   1. `semantic`       — primary splitter. Cuts at topic shifts.
 *   2. `recursive`      — used for short passages (no split needed) and to
 *                         break up any semantic chunk that came out oversized.
 *   3. `overlap`        — every child is widened into its neighbours so facts
 *                         near a boundary are never truncated.
 *   4. `sliding_window` — extra overlapping views, only for passages long
 *                         enough that boundary effects actually matter.
 *   5. `parent`         — coarse spans grouping ≥2 children, used to hand the
 *                         LLM wider context than the matched child.
 *   6. `metadata`       — a topic/language header prefixed to the embedded
 *                         text (stored separately from the display text).
 *
 * MS MARCO passages are short, so most produce exactly one child and no
 * parent. Longer passages fan out into all of the above.
 */
import { createHash } from 'node:crypto';
import type { Chunk, ChunkMetadata, ChunkStrategy, SourceDocument } from '@goarag/shared';
import { estimateTokens, normalizeWhitespace, splitSentences, truncate } from '../../utils/text.js';
import {
  applyOverlap,
  buildMetadataHeader,
  parentChildChunk,
  recursiveChunk,
  semanticChunk,
  slidingWindowChunk,
  withMetadataHeader,
  type TextSpan,
} from './strategies.js';

export interface ChunkingOptions {
  targetTokens: number;
  minTokens: number;
  overlapTokens: number;
  parentTokens: number;
  semanticPercentile: number;
  windowTokens: number;
  windowStride: number;
}

/** A chunk plus the exact string that should be embedded for it. */
export interface PreparedChunk extends Chunk {
  /** Chunk text prefixed with the metadata header — this is what gets embedded. */
  embedText: string;
  /** Stable human-readable key, useful for debugging and idempotent re-indexing. */
  chunkKey: string;
  /** True for coarse parent spans, which are excluded from search by default. */
  isParent: boolean;
}

/**
 * Deterministic UUID (v5-style, SHA-1 based) derived from a key.
 *
 * Qdrant point ids must be a uint64 or a UUID, and we need them stable so
 * re-running the indexer upserts in place instead of duplicating.
 */
export function deterministicUuid(key: string): string {
  const hash = createHash('sha1').update(key).digest();
  const bytes = Buffer.from(hash.subarray(0, 16));
  bytes[6] = ((bytes[6] as number) & 0x0f) | 0x50; // version 5
  bytes[8] = ((bytes[8] as number) & 0x3f) | 0x80; // RFC 4122 variant
  const hex = bytes.toString('hex');
  return `${hex.slice(0, 8)}-${hex.slice(8, 12)}-${hex.slice(12, 16)}-${hex.slice(16, 20)}-${hex.slice(20)}`;
}

/** Derive a short topic label from the document's query or leading text. */
function deriveTopic(doc: SourceDocument): string {
  const candidate = (doc.queryText ?? doc.topic ?? '').trim();
  if (candidate) return truncate(candidate.replace(/^[.\s]+/, ''), 90);
  return truncate(doc.text.trim(), 60);
}

/**
 * Chunk one source document.
 *
 * Returns children first, then parents; both carry full metadata.
 */
export function chunkDocument(doc: SourceDocument, options: ChunkingOptions): PreparedChunk[] {
  const text = normalizeWhitespace(doc.text);
  if (!text) return [];

  const totalTokens = estimateTokens(text);
  const topic = deriveTopic(doc);
  const header = buildMetadataHeader({ topic, language: doc.language, source: doc.source });

  // ── children ──────────────────────────────────────────────────────────────
  let childSpans: TextSpan[];
  let baseStrategy: ChunkStrategy;

  // Route on sentence count, not token count. MS MARCO passages usually sit
  // under the token budget yet still cover two unrelated ideas — splitting
  // only when oversized would leave those glued together and let an
  // irrelevant half drag the chunk's embedding off-topic. Semantic chunking
  // cuts at the topic shift regardless of length; recursive is reserved for
  // passages with too few sentences for a breakpoint to be meaningful.
  const sentenceCount = splitSentences(text).length;

  if (sentenceCount < 3) {
    childSpans = recursiveChunk(text, options.targetTokens, options.minTokens);
    baseStrategy = 'recursive';
  } else {
    childSpans = semanticChunk(text, options.targetTokens, options.minTokens, options.semanticPercentile);
    baseStrategy = 'semantic';

    // Any semantic span that is still oversized gets broken structurally.
    const refined: TextSpan[] = [];
    for (const span of childSpans) {
      if (span.tokenCount > options.targetTokens * 1.5) {
        const sub = recursiveChunk(span.text, options.targetTokens, options.minTokens);
        // Re-anchor sub-span offsets onto the parent document.
        for (const s of sub) {
          refined.push({
            ...s,
            charStart: span.charStart + s.charStart,
            charEnd: Math.min(span.charEnd, span.charStart + s.charEnd),
          });
        }
      } else {
        refined.push(span);
      }
    }
    childSpans = refined;
  }

  if (childSpans.length === 0) return [];

  // ── overlap ───────────────────────────────────────────────────────────────
  const overlapped = applyOverlap(childSpans, text, options.overlapTokens);
  const overlapApplied = options.overlapTokens > 0 && childSpans.length > 1;

  // ── parents ───────────────────────────────────────────────────────────────
  const { parents, childToParent } = parentChildChunk(text, overlapped, options.parentTokens);
  // A parent that wraps a single child adds nothing, so only keep real groupings.
  const parentChildCounts = new Map<number, number>();
  for (const p of childToParent) parentChildCounts.set(p, (parentChildCounts.get(p) ?? 0) + 1);
  const keptParents = new Set<number>(
    [...parentChildCounts.entries()].filter(([, count]) => count >= 2).map(([index]) => index),
  );

  const chunks: PreparedChunk[] = [];
  const parentIds = new Map<number, string>();

  for (const parentIndex of keptParents) {
    const span = parents[parentIndex];
    if (!span) continue;
    const chunkKey = `${doc.documentId}::${doc.passageId}::parent::${parentIndex}`;
    const id = deterministicUuid(chunkKey);
    parentIds.set(parentIndex, id);
    chunks.push(
      buildChunk({
        id,
        chunkKey,
        span,
        doc,
        topic,
        header,
        strategy: 'parent',
        chunkIndex: parentIndex,
        parentChunk: null,
        isParent: true,
      }),
    );
  }

  overlapped.forEach((span, index) => {
    const parentIndex = childToParent[index];
    const parentId = parentIndex !== undefined ? (parentIds.get(parentIndex) ?? null) : null;
    // Record `overlap` as the strategy when overlap actually widened this span.
    const strategy: ChunkStrategy = overlapApplied ? 'overlap' : baseStrategy;
    const chunkKey = `${doc.documentId}::${doc.passageId}::${strategy}::${index}`;
    chunks.push(
      buildChunk({
        id: deterministicUuid(chunkKey),
        chunkKey,
        span,
        doc,
        topic,
        header,
        strategy,
        chunkIndex: index,
        parentChunk: parentId,
        isParent: false,
      }),
    );
  });

  // ── sliding windows ───────────────────────────────────────────────────────
  // Only worth the extra vectors when the passage is long enough for a fact to
  // straddle a boundary; otherwise the windows duplicate the children exactly.
  if (totalTokens > options.windowTokens * 2) {
    const windows = slidingWindowChunk(text, options.windowTokens, options.windowStride);
    const existing = new Set(chunks.map((c) => c.text));
    windows.forEach((span, index) => {
      if (existing.has(span.text)) return;
      const chunkKey = `${doc.documentId}::${doc.passageId}::sliding_window::${index}`;
      chunks.push(
        buildChunk({
          id: deterministicUuid(chunkKey),
          chunkKey,
          span,
          doc,
          topic,
          header,
          strategy: 'sliding_window',
          chunkIndex: index,
          parentChunk: null,
          isParent: false,
        }),
      );
    });
  }

  return chunks;
}

function buildChunk(input: {
  id: string;
  chunkKey: string;
  span: TextSpan;
  doc: SourceDocument;
  topic: string;
  header: string;
  strategy: ChunkStrategy;
  chunkIndex: number;
  parentChunk: string | null;
  isParent: boolean;
}): PreparedChunk {
  const { id, chunkKey, span, doc, topic, header, strategy, chunkIndex, parentChunk, isParent } = input;

  const metadata: ChunkMetadata = {
    documentId: doc.documentId,
    source: doc.source,
    language: doc.language,
    passageId: doc.passageId,
    chunkIndex,
    parentChunk,
    topic,
    strategy,
    tokenCount: span.tokenCount,
    charStart: span.charStart,
    charEnd: span.charEnd,
    isSelected: doc.isSelected,
    queryId: doc.queryId,
    queryText: doc.queryText,
    indexedAt: new Date().toISOString(),
  };

  return {
    id,
    chunkKey,
    text: span.text,
    // The metadata header is embedded but never displayed — it lifts
    // cross-lingual recall without polluting the answer context.
    embedText: withMetadataHeader(header, span.text),
    metadata,
    isParent,
  };
}

/** Chunk a batch of documents. */
export function chunkDocuments(docs: readonly SourceDocument[], options: ChunkingOptions): PreparedChunk[] {
  const out: PreparedChunk[] = [];
  const seen = new Set<string>();
  for (const doc of docs) {
    for (const chunk of chunkDocument(doc, options)) {
      // Deterministic ids mean a repeated passage collapses instead of duplicating.
      if (seen.has(chunk.id)) continue;
      seen.add(chunk.id);
      out.push(chunk);
    }
  }
  return out;
}

/** Aggregate statistics over a chunk set, surfaced on the dashboard. */
export function summarizeChunks(chunks: readonly PreparedChunk[]) {
  const byStrategy = new Map<ChunkStrategy, number>();
  const byLanguage = new Map<string, number>();
  let totalChars = 0;
  let totalTokens = 0;

  for (const chunk of chunks) {
    byStrategy.set(chunk.metadata.strategy, (byStrategy.get(chunk.metadata.strategy) ?? 0) + 1);
    byLanguage.set(chunk.metadata.language, (byLanguage.get(chunk.metadata.language) ?? 0) + 1);
    totalChars += chunk.text.length;
    totalTokens += chunk.metadata.tokenCount;
  }

  return {
    count: chunks.length,
    averageChars: chunks.length ? Math.round(totalChars / chunks.length) : 0,
    averageTokens: chunks.length ? Math.round(totalTokens / chunks.length) : 0,
    byStrategy: [...byStrategy.entries()].map(([strategy, count]) => ({ strategy, count })),
    byLanguage: [...byLanguage.entries()]
      .map(([language, count]) => ({ language, count }))
      .sort((a, b) => b.count - a.count),
  };
}
