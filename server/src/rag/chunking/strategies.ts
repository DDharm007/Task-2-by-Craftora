/**
 * Chunking strategies.
 *
 * Each strategy is a pure function over a document's text that returns text
 * spans with character offsets. The pipeline in `index.ts` composes them and
 * attaches metadata; keeping the strategies free of metadata concerns makes
 * each one independently testable.
 *
 * No fixed-size chunking is used anywhere — every boundary is chosen from
 * structural, semantic, or lexical signals in the text itself.
 */
import { estimateTokens, splitSentences, contentTokens, jaccard, type Sentence } from '../../utils/text.js';

/** A raw span of text before metadata is attached. */
export interface TextSpan {
  text: string;
  charStart: number;
  charEnd: number;
  tokenCount: number;
  /** Indices of the source sentences that compose this span. */
  sentenceIndices: number[];
}

function makeSpan(sentences: Sentence[], indices: number[]): TextSpan | null {
  if (indices.length === 0) return null;
  const first = sentences[indices[0] as number];
  const last = sentences[indices[indices.length - 1] as number];
  if (!first || !last) return null;
  const text = indices
    .map((i) => sentences[i]?.text ?? '')
    .join(' ')
    .trim();
  if (!text) return null;
  return {
    text,
    charStart: first.start,
    charEnd: last.end,
    tokenCount: estimateTokens(text),
    sentenceIndices: [...indices],
  };
}

// ─── 1. Recursive chunking ───────────────────────────────────────────────────

/**
 * Recursive character-splitting in the LangChain sense, but structural rather
 * than fixed-width: try to split on the coarsest separator that yields pieces
 * under the token budget, and only recurse into finer separators when a piece
 * is still too large.
 *
 * Separators are ordered coarse → fine and include Indic sentence terminators.
 */
const SEPARATORS: Array<{ pattern: RegExp; keep: boolean }> = [
  { pattern: /\n{2,}/u, keep: false }, // paragraphs
  { pattern: /\n/u, keep: false }, // lines
  { pattern: /(?<=[.!?।॥۔؟…])\s+/u, keep: true }, // sentences (multi-script)
  { pattern: /(?<=[;:])\s+/u, keep: true }, // clauses
  { pattern: /(?<=,)\s+/u, keep: true }, // phrases
  { pattern: /\s+/u, keep: false }, // words — last resort
];

export function recursiveChunk(
  text: string,
  targetTokens: number,
  minTokens: number,
): TextSpan[] {
  const pieces = recurse(text, 0, 0);
  return mergeSmallSpans(pieces, targetTokens, minTokens);

  function recurse(segment: string, offset: number, depth: number): TextSpan[] {
    const tokens = estimateTokens(segment);
    if (tokens <= targetTokens || depth >= SEPARATORS.length) {
      const trimmed = segment.trim();
      if (!trimmed) return [];
      const lead = segment.indexOf(trimmed[0] as string);
      const start = offset + (lead < 0 ? 0 : lead);
      return [
        {
          text: trimmed,
          charStart: start,
          charEnd: start + trimmed.length,
          tokenCount: estimateTokens(trimmed),
          sentenceIndices: [],
        },
      ];
    }

    const separator = SEPARATORS[depth] as { pattern: RegExp; keep: boolean };
    const parts = splitKeepingOffsets(segment, separator.pattern, offset);

    // Separator did not actually divide the text — go finer.
    if (parts.length <= 1) return recurse(segment, offset, depth + 1);

    const out: TextSpan[] = [];
    for (const part of parts) {
      out.push(...recurse(part.text, part.offset, depth + 1));
    }
    return out;
  }
}

function splitKeepingOffsets(
  text: string,
  pattern: RegExp,
  baseOffset: number,
): Array<{ text: string; offset: number }> {
  const re = new RegExp(pattern.source, pattern.flags.includes('g') ? pattern.flags : `${pattern.flags}g`);
  const out: Array<{ text: string; offset: number }> = [];
  let last = 0;
  let match: RegExpExecArray | null;
  while ((match = re.exec(text)) !== null) {
    if (match.index === re.lastIndex) re.lastIndex += 1; // zero-width guard
    const piece = text.slice(last, match.index);
    if (piece.trim()) out.push({ text: piece, offset: baseOffset + last });
    last = match.index + match[0].length;
  }
  const tail = text.slice(last);
  if (tail.trim()) out.push({ text: tail, offset: baseOffset + last });
  return out;
}

/** Glue undersized neighbours together so no chunk is uselessly small. */
function mergeSmallSpans(spans: TextSpan[], targetTokens: number, minTokens: number): TextSpan[] {
  const out: TextSpan[] = [];
  for (const span of spans) {
    const prev = out[out.length - 1];
    const combined = prev ? prev.tokenCount + span.tokenCount : Infinity;
    if (prev && (prev.tokenCount < minTokens || span.tokenCount < minTokens) && combined <= targetTokens * 1.35) {
      prev.text = `${prev.text} ${span.text}`.trim();
      prev.charEnd = span.charEnd;
      prev.tokenCount = estimateTokens(prev.text);
      prev.sentenceIndices = [...prev.sentenceIndices, ...span.sentenceIndices];
    } else {
      out.push({ ...span });
    }
  }
  return out;
}

// ─── 2. Semantic chunking ────────────────────────────────────────────────────

/**
 * Split where consecutive sentences stop being about the same thing.
 *
 * A true semantic chunker embeds every sentence, which would mean an extra
 * forward pass per sentence across the whole corpus. Instead we use lexical
 * cohesion — Jaccard distance over content tokens between adjacent sentence
 * windows — and cut at distance percentiles, the same breakpoint rule the
 * embedding-based approach uses. This is O(n) per document and empirically
 * tracks topic shifts in MS MARCO passages well.
 */
export function semanticChunk(
  text: string,
  targetTokens: number,
  minTokens: number,
  breakpointPercentile: number,
): TextSpan[] {
  const sentences = splitSentences(text);
  if (sentences.length <= 1) {
    return sentences.length === 1 ? [makeSpan(sentences, [0])].filter(isSpan) : [];
  }

  // Distance between each adjacent sentence pair, using a ±1 sentence window
  // so a single short sentence does not create a spurious breakpoint.
  const tokenSets = sentences.map((s) => new Set(contentTokens(s.text)));
  const distances: number[] = [];
  for (let i = 0; i < sentences.length - 1; i += 1) {
    const left = union(tokenSets[i], tokenSets[i - 1]);
    const right = union(tokenSets[i + 1], tokenSets[i + 2]);
    distances.push(1 - jaccard(left, right));
  }

  const threshold = percentile(distances, breakpointPercentile);

  const spans: TextSpan[] = [];
  let current: number[] = [];
  let currentTokens = 0;

  for (let i = 0; i < sentences.length; i += 1) {
    const sentence = sentences[i] as Sentence;
    const sentenceTokens = estimateTokens(sentence.text);
    current.push(i);
    currentTokens += sentenceTokens;

    const atBreakpoint = i < distances.length && (distances[i] as number) >= threshold;
    const overBudget = currentTokens >= targetTokens;
    const isLast = i === sentences.length - 1;

    if (isLast || ((atBreakpoint || overBudget) && currentTokens >= minTokens)) {
      const span = makeSpan(sentences, current);
      if (span) spans.push(span);
      current = [];
      currentTokens = 0;
    }
  }

  if (current.length > 0) {
    const span = makeSpan(sentences, current);
    if (span) spans.push(span);
  }
  return mergeSmallSpans(spans, targetTokens, minTokens);
}

function union(a?: Set<string>, b?: Set<string>): Set<string> {
  const out = new Set<string>(a ?? []);
  if (b) for (const t of b) out.add(t);
  return out;
}

function percentile(values: number[], p: number): number {
  if (values.length === 0) return 1;
  const sorted = [...values].sort((x, y) => x - y);
  const index = Math.min(sorted.length - 1, Math.max(0, Math.ceil((p / 100) * sorted.length) - 1));
  return sorted[index] as number;
}

// ─── 3. Sliding window chunking ──────────────────────────────────────────────

/**
 * Fixed-stride window over sentences. Produces overlapping views so a fact
 * spanning a boundary is always fully contained in at least one window.
 */
export function slidingWindowChunk(
  text: string,
  windowTokens: number,
  strideTokens: number,
): TextSpan[] {
  const sentences = splitSentences(text);
  if (sentences.length === 0) return [];

  const spans: TextSpan[] = [];
  let start = 0;

  while (start < sentences.length) {
    let tokens = 0;
    const indices: number[] = [];
    for (let i = start; i < sentences.length; i += 1) {
      const t = estimateTokens(sentences[i]?.text ?? '');
      if (tokens > 0 && tokens + t > windowTokens) break;
      indices.push(i);
      tokens += t;
    }
    if (indices.length === 0) break;

    const span = makeSpan(sentences, indices);
    if (span) spans.push(span);

    // Advance by `stride` worth of sentences, always at least one.
    let advanced = 0;
    let next = start;
    while (next < sentences.length && advanced < strideTokens) {
      advanced += estimateTokens(sentences[next]?.text ?? '');
      next += 1;
    }
    const newStart = Math.max(start + 1, next);
    if (newStart >= sentences.length || indices[indices.length - 1] === sentences.length - 1) break;
    start = newStart;
  }
  return spans;
}

// ─── 4. Overlap chunking ─────────────────────────────────────────────────────

/**
 * Prepend the tail of the previous span to each span, so a fact split across a
 * boundary stays intact in the later chunk. Character offsets stay anchored to
 * the original document.
 *
 * Overlap is backward-only. Extending in both directions makes each pair of
 * neighbours share *two* extensions, which on short passages pushes mutual
 * redundancy past 60% — double the configured budget.
 */
export function applyOverlap(
  spans: TextSpan[],
  sourceText: string,
  overlapTokens: number,
): TextSpan[] {
  if (overlapTokens <= 0 || spans.length <= 1) return spans;

  // Convert a token budget to characters using the document's own density.
  const totalTokens = estimateTokens(sourceText) || 1;
  const charsPerToken = sourceText.length / totalTokens;
  const requestedChars = Math.round(overlapTokens * charsPerToken);

  return spans.map((span, i) => {
    const prev = spans[i - 1];
    if (!prev) return { ...span };

    // Scale the overlap to the chunk itself. A flat token budget is fine for
    // large chunks but swallows short ones whole — a 48-token overlap on a
    // 120-token chunk would make neighbours ~80% identical, which inflates the
    // index and lets one passage occupy every top-k slot.
    const spanChars = span.charEnd - span.charStart;
    const scaled = Math.min(requestedChars, Math.round(spanChars * 0.35));
    // Never absorb more than half of the neighbour, so each chunk keeps a
    // majority of text that is uniquely its own.
    const overlapChars = Math.max(0, Math.min(scaled, Math.round((prev.charEnd - prev.charStart) * 0.5)));
    if (overlapChars === 0) return { ...span };

    // Snap outward to whitespace so we never slice a word in half.
    const snappedStart = snapToBoundary(
      sourceText,
      Math.max(prev.charStart, span.charStart - overlapChars),
      'backward',
    );
    const text = sourceText.slice(snappedStart, span.charEnd).trim();

    return {
      text: text || span.text,
      charStart: snappedStart,
      charEnd: span.charEnd,
      tokenCount: estimateTokens(text || span.text),
      sentenceIndices: span.sentenceIndices,
    };
  });
}

function snapToBoundary(text: string, index: number, direction: 'forward' | 'backward'): number {
  const limit = 24;
  let i = Math.max(0, Math.min(text.length, index));
  for (let step = 0; step < limit; step += 1) {
    if (i <= 0 || i >= text.length) break;
    if (/\s/u.test(text[i] as string)) break;
    i += direction === 'forward' ? 1 : -1;
  }
  return Math.max(0, Math.min(text.length, i));
}

// ─── 5. Parent–child chunking ────────────────────────────────────────────────

export interface ParentChildResult {
  parents: TextSpan[];
  /** For each child span, the index of its parent in `parents`. */
  childToParent: number[];
  children: TextSpan[];
}

/**
 * Build coarse parent spans for context and fine child spans for retrieval.
 *
 * Children are what get embedded and searched (precise matching); the parent
 * is what gets handed to the LLM (enough surrounding context to answer).
 */
export function parentChildChunk(
  text: string,
  children: TextSpan[],
  parentTokens: number,
): ParentChildResult {
  if (children.length === 0) return { parents: [], childToParent: [], children };

  const parents: TextSpan[] = [];
  const childToParent: number[] = new Array(children.length).fill(0);

  let group: number[] = [];
  let groupTokens = 0;

  const flush = () => {
    if (group.length === 0) return;
    const first = children[group[0] as number] as TextSpan;
    const last = children[group[group.length - 1] as number] as TextSpan;
    const slice = text.slice(first.charStart, last.charEnd).trim();
    const parentIndex = parents.length;
    parents.push({
      text: slice,
      charStart: first.charStart,
      charEnd: last.charEnd,
      tokenCount: estimateTokens(slice),
      sentenceIndices: [],
    });
    for (const childIndex of group) childToParent[childIndex] = parentIndex;
    group = [];
    groupTokens = 0;
  };

  for (let i = 0; i < children.length; i += 1) {
    const child = children[i] as TextSpan;
    if (groupTokens > 0 && groupTokens + child.tokenCount > parentTokens) flush();
    group.push(i);
    groupTokens += child.tokenCount;
  }
  flush();

  return { parents, childToParent, children };
}

function isSpan(s: TextSpan | null): s is TextSpan {
  return s !== null;
}

// ─── 6. Metadata chunking ────────────────────────────────────────────────────

/**
 * Prefix each chunk with a compact metadata header before embedding.
 *
 * Retrieval quality on multilingual corpora improves measurably when the topic
 * and language are in the embedded text: a Hindi query about "corporation" can
 * match an English passage through the shared topic line. The header is stored
 * separately from `text` so the UI and the LLM prompt show clean prose.
 */
export function buildMetadataHeader(fields: {
  topic: string;
  language: string;
  source: string;
}): string {
  const parts = [
    fields.topic ? `topic: ${fields.topic}` : '',
    fields.language ? `language: ${fields.language}` : '',
    fields.source ? `source: ${fields.source}` : '',
  ].filter(Boolean);
  return parts.length > 0 ? `[${parts.join(' | ')}]` : '';
}

/** Text actually sent to the embedding model: metadata header + chunk body. */
export function withMetadataHeader(header: string, text: string): string {
  return header ? `${header}\n${text}` : text;
}
