/**
 * Rank fusion and diversity selection.
 *
 * Dense and sparse retrieval return scores on completely different scales —
 * cosine similarity sits in [0,1] while BM25 is unbounded and corpus
 * dependent. Normalising them into a shared scale requires assumptions that
 * break whenever the corpus changes, so we fuse on *rank* instead of score
 * using Reciprocal Rank Fusion, which is scale-free by construction.
 */
import type { SearchHit } from '../vector/types.js';
import { dot } from '../embeddings/types.js';

export interface FusedCandidate {
  hit: SearchHit;
  denseScore: number | null;
  sparseScore: number | null;
  denseRank: number | null;
  sparseRank: number | null;
  fusedScore: number;
  matchedBy: Array<'dense' | 'sparse'>;
}

/**
 * Reciprocal Rank Fusion.
 *
 *     RRF(d) = Σ_arms 1 / (k + rank_arm(d))
 *
 * `k` (default 60, from the original Cormack et al. paper) damps the influence
 * of top ranks so a single arm cannot dominate the merged list. A document
 * found by both arms accumulates two contributions, which is precisely the
 * behaviour we want: agreement between lexical and semantic evidence is the
 * strongest signal available before reranking.
 */
export function reciprocalRankFusion(
  dense: readonly SearchHit[],
  sparse: readonly SearchHit[],
  k: number,
): FusedCandidate[] {
  const byId = new Map<string, FusedCandidate>();

  const ensure = (hit: SearchHit): FusedCandidate => {
    let candidate = byId.get(hit.id);
    if (!candidate) {
      candidate = {
        hit,
        denseScore: null,
        sparseScore: null,
        denseRank: null,
        sparseRank: null,
        fusedScore: 0,
        matchedBy: [],
      };
      byId.set(hit.id, candidate);
    }
    return candidate;
  };

  dense.forEach((hit, index) => {
    const candidate = ensure(hit);
    candidate.denseScore = hit.score;
    candidate.denseRank = index + 1;
    candidate.fusedScore += 1 / (k + index + 1);
    candidate.matchedBy.push('dense');
  });

  sparse.forEach((hit, index) => {
    const candidate = ensure(hit);
    candidate.sparseScore = hit.score;
    candidate.sparseRank = index + 1;
    candidate.fusedScore += 1 / (k + index + 1);
    if (!candidate.matchedBy.includes('sparse')) candidate.matchedBy.push('sparse');
  });

  return [...byId.values()].sort((a, b) => b.fusedScore - a.fusedScore);
}

/**
 * Agreement between the two retrieval arms, 0-1.
 *
 * Used as a confidence signal: when dense and sparse independently surface the
 * same documents, the retrieval is far more trustworthy than when only one arm
 * fires. Computed as the Jaccard overlap of the two arms' top results.
 */
export function retrievalAgreement(dense: readonly SearchHit[], sparse: readonly SearchHit[]): number {
  if (dense.length === 0 || sparse.length === 0) return 0;
  const denseIds = new Set(dense.map((h) => h.id));
  const sparseIds = new Set(sparse.map((h) => h.id));
  let intersection = 0;
  for (const id of denseIds) if (sparseIds.has(id)) intersection += 1;
  return intersection / (denseIds.size + sparseIds.size - intersection);
}

/**
 * Maximal Marginal Relevance.
 *
 * Overlapping chunk strategies mean the top-k can easily be five near-copies
 * of one passage — which starves the LLM of the breadth it needs to answer.
 * MMR greedily picks the candidate maximising
 *
 *     λ · relevance − (1 − λ) · max similarity to anything already picked
 *
 * so each new chunk has to earn its slot by adding something.
 *
 * Trigram sets are built **once per candidate**, not once per comparison. The
 * greedy loop asks for `similarity` O(limit² · candidates) times — roughly a
 * thousand calls for the default top-10-of-20 — and the obvious implementation
 * re-derives both sides' trigrams from the full chunk text on every one of
 * them. That alone measured ~70ms per query, more than the embedding and the
 * vector scan combined, and it is pure repeated work: there are only ever
 * `candidates.length` distinct sets. Memoising them takes MMR to ~1ms.
 */
export function maximalMarginalRelevance<T>(
  candidates: readonly T[],
  limit: number,
  lambda: number,
  relevanceOf: (item: T) => number,
  vectorOf: (item: T) => Float32Array | null,
  textOf: (item: T) => string,
): T[] {
  if (candidates.length <= 1 || limit <= 1) return candidates.slice(0, limit);

  const selected: T[] = [];
  const remaining = [...candidates];

  // Relevance is normalised so λ trades off against similarity on one scale.
  const relevances = new Map<T, number>();
  let maxRelevance = -Infinity;
  let minRelevance = Infinity;
  for (const candidate of candidates) {
    const value = relevanceOf(candidate);
    relevances.set(candidate, value);
    if (value > maxRelevance) maxRelevance = value;
    if (value < minRelevance) minRelevance = value;
  }
  const span = maxRelevance - minRelevance || 1;
  const normalized = (item: T) => ((relevances.get(item) ?? 0) - minRelevance) / span;

  // Built on first use rather than up front: when every candidate carries a
  // vector the trigram path is never taken and none of these are needed.
  const gramCache = new Map<T, Set<string>>();
  const gramsFor = (item: T): Set<string> => {
    let grams = gramCache.get(item);
    if (!grams) {
      grams = trigrams(textOf(item));
      gramCache.set(item, grams);
    }
    return grams;
  };

  while (selected.length < limit && remaining.length > 0) {
    let bestIndex = 0;
    let bestScore = -Infinity;

    for (let i = 0; i < remaining.length; i += 1) {
      const candidate = remaining[i] as T;
      let maxSimilarity = 0;
      for (const chosen of selected) {
        maxSimilarity = Math.max(
          maxSimilarity,
          similarity(candidate, chosen, vectorOf, gramsFor),
        );
      }
      const score = lambda * normalized(candidate) - (1 - lambda) * maxSimilarity;
      if (score > bestScore) {
        bestScore = score;
        bestIndex = i;
      }
    }

    selected.push(remaining[bestIndex] as T);
    remaining.splice(bestIndex, 1);
  }

  return selected;
}

/** Cosine similarity when vectors are available, trigram overlap otherwise. */
function similarity<T>(
  a: T,
  b: T,
  vectorOf: (item: T) => Float32Array | null,
  gramsFor: (item: T) => Set<string>,
): number {
  const vectorA = vectorOf(a);
  const vectorB = vectorOf(b);
  if (vectorA && vectorB) return Math.max(0, dot(vectorA, vectorB));
  return jaccard(gramsFor(a), gramsFor(b));
}

/** Jaccard overlap of two prepared sets. */
function jaccard(a: ReadonlySet<string>, b: ReadonlySet<string>): number {
  if (a.size === 0 || b.size === 0) return 0;
  // Iterate the smaller side: the loop is O(min) rather than O(|a|).
  const [small, large] = a.size <= b.size ? [a, b] : [b, a];
  let intersection = 0;
  for (const gram of small) if (large.has(gram)) intersection += 1;
  return intersection / (a.size + b.size - intersection);
}

/**
 * Character-trigram Jaccard similarity.
 *
 * Character n-grams rather than word tokens, because overlapping chunks of the
 * same passage share long literal substrings, and trigrams catch that in any
 * script without needing word segmentation.
 */
export function trigramSimilarity(a: string, b: string): number {
  if (!a || !b) return 0;
  if (a === b) return 1;
  return jaccard(trigrams(a), trigrams(b));
}

function trigrams(text: string): Set<string> {
  const normalized = text.toLowerCase().replace(/\s+/gu, ' ').trim();
  const out = new Set<string>();
  for (let i = 0; i + 3 <= normalized.length; i += 1) out.add(normalized.slice(i, i + 3));
  return out;
}
