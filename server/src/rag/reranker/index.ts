/**
 * Reranking: top-10 candidates → top-5 final context.
 *
 * Bi-encoder retrieval embeds the query and the document independently, so it
 * can only ever compare two summaries. A cross-encoder reads the query and
 * document *together* in one forward pass, letting attention align specific
 * query terms against specific document spans. That is far more accurate, and
 * far too slow to run over a whole corpus — which is exactly why it belongs in
 * a second stage over a shortlist.
 *
 * Two providers:
 *   • `local`     — BAAI bge-reranker-base (XLM-RoBERTa cross-encoder, ONNX).
 *                   Multilingual, so a Hindi query scores English passages
 *                   correctly. ~266MB, downloaded once.
 *   • `heuristic` — no download. Combines fused rank, lexical coverage and
 *                   term proximity. Weaker than the cross-encoder but honest
 *                   about it, and keeps the pipeline working offline.
 */
import { config } from '../../config/env.js';
import { logger } from '../../utils/logger.js';
import { chunkArray, once } from '../../utils/async.js';
import { contentTokens, tokenize } from '../../utils/text.js';

export interface RerankCandidate {
  id: string;
  text: string;
  /** Score carried over from fusion, used by the heuristic provider. */
  priorScore: number;
}

export interface RerankResult {
  id: string;
  /** Relevance in [0,1]. Comparable across queries. */
  score: number;
}

export interface Reranker {
  readonly name: string;
  readonly model: string;
  rerank(query: string, candidates: readonly RerankCandidate[]): Promise<RerankResult[]>;
  warmup(): Promise<void>;
  healthCheck(): Promise<{ ok: boolean; detail: string }>;
}

// ─── Local cross-encoder ─────────────────────────────────────────────────────

/**
 * Tokenizer + model handles.
 *
 * We deliberately do NOT use the `text-classification` pipeline here. BGE
 * rerankers have a *single* output logit (a regression head, `id2label` is just
 * `{0: "LABEL_0"}`), and that pipeline applies softmax across the label axis —
 * softmax over one class returns exactly 1.0 for every input, destroying the
 * score. Reading the raw logit and squashing it with a sigmoid is both the
 * documented usage and the only thing that discriminates: on a relevant pair
 * the logit is ~+8, on an irrelevant one ~-10.
 */
interface RerankerHandles {
  tokenizer: (
    texts: string[],
    options: { text_pair: string[]; padding: boolean; truncation: boolean; max_length: number },
  ) => Record<string, unknown>;
  model: (inputs: Record<string, unknown>) => Promise<{ logits: { dims: number[]; data: ArrayLike<number> } }>;
}

class LocalCrossEncoderReranker implements Reranker {
  readonly name = 'local:cross-encoder';
  readonly model: string;
  private readonly getHandles: () => Promise<RerankerHandles>;

  constructor() {
    this.model = config.reranker.model;
    this.getHandles = once(async () => {
      const mod = await import('@huggingface/transformers');
      mod.env.cacheDir = config.embedding.cacheDir;
      mod.env.allowLocalModels = true;

      logger.info(
        { model: this.model, quantization: config.reranker.quantization },
        'Loading cross-encoder reranker (first run downloads weights)',
      );
      const started = Date.now();

      const dtype = config.reranker.quantization as
        | 'fp32' | 'fp16' | 'q8' | 'int8' | 'uint8' | 'q4' | 'q4f16';

      const [tokenizer, model] = await Promise.all([
        mod.AutoTokenizer.from_pretrained(this.model),
        mod.AutoModelForSequenceClassification.from_pretrained(this.model, { dtype, device: 'cpu' }),
      ]);

      logger.info({ model: this.model, ms: Date.now() - started }, 'Reranker ready');
      return {
        tokenizer: tokenizer as unknown as RerankerHandles['tokenizer'],
        model: model as unknown as RerankerHandles['model'],
      };
    });
  }

  async warmup(): Promise<void> {
    await this.rerank('warmup', [{ id: 'w', text: 'warmup passage', priorScore: 0 }]);
  }

  async rerank(query: string, candidates: readonly RerankCandidate[]): Promise<RerankResult[]> {
    if (candidates.length === 0) return [];
    const { tokenizer, model } = await this.getHandles();
    const out: RerankResult[] = [];

    // Batched so a top-10 rerank is a few forward passes, not ten.
    for (const batch of chunkArray([...candidates], config.reranker.batchSize)) {
      const queries = batch.map(() => query);
      // Cap the document side: the model truncates at 512 tokens anyway, and
      // trimming first avoids tokenising text that will be discarded.
      const documents = batch.map((candidate) => candidate.text.slice(0, 2_000));

      const inputs = tokenizer(queries, {
        text_pair: documents,
        padding: true,
        truncation: true,
        max_length: 512,
      });

      const { logits } = await model(inputs);
      batch.forEach((candidate, i) => {
        const logit = Number(logits.data[i] ?? 0);
        out.push({ id: candidate.id, score: sigmoid(logit) });
      });
    }

    return out.sort((a, b) => b.score - a.score);
  }

  async healthCheck(): Promise<{ ok: boolean; detail: string }> {
    try {
      // Probe with a clearly relevant and a clearly irrelevant pair — a
      // reranker that returns identical scores for both is broken even though
      // it is technically "responding".
      const results = await this.rerank('What is a corporation?', [
        { id: 'relevant', text: 'A corporation is a company recognised in law as a single entity.', priorScore: 0 },
        { id: 'irrelevant', text: 'Hurricanes form over warm ocean water.', priorScore: 0 },
      ]);
      const relevant = results.find((r) => r.id === 'relevant')?.score ?? 0;
      const irrelevant = results.find((r) => r.id === 'irrelevant')?.score ?? 0;

      if (relevant <= irrelevant) {
        return { ok: false, detail: `reranker is not discriminating (${relevant.toFixed(3)} vs ${irrelevant.toFixed(3)})` };
      }
      return {
        ok: true,
        detail: `${this.model} (${config.reranker.quantization}) · separation ${(relevant - irrelevant).toFixed(3)}`,
      };
    } catch (error) {
      return { ok: false, detail: (error as Error).message };
    }
  }
}

/** Squash an unbounded logit into a [0,1] relevance. */
function sigmoid(x: number): number {
  if (x >= 0) return 1 / (1 + Math.exp(-x));
  // Numerically stable form for large negative logits.
  const z = Math.exp(x);
  return z / (1 + z);
}

// ─── Heuristic reranker ──────────────────────────────────────────────────────

/**
 * Lexical reranker used when no model is available.
 *
 * Blends three signals the cross-encoder would otherwise capture implicitly:
 *   • coverage  — how many of the query's content words the passage contains
 *   • proximity — how tightly those matches cluster (a passage mentioning both
 *                 terms in one sentence beats one mentioning them 200 words apart)
 *   • prior     — the fused retrieval rank, so semantic matches are not lost
 */
class HeuristicReranker implements Reranker {
  readonly name = 'heuristic';
  readonly model = 'lexical-coverage+proximity';

  async warmup(): Promise<void> {
    // Nothing to load.
  }

  async rerank(query: string, candidates: readonly RerankCandidate[]): Promise<RerankResult[]> {
    const queryTerms = contentTokens(query);
    if (queryTerms.length === 0) {
      return candidates
        .map((candidate) => ({ id: candidate.id, score: candidate.priorScore }))
        .sort((a, b) => b.score - a.score);
    }

    const querySet = new Set(queryTerms);
    const maxPrior = Math.max(...candidates.map((c) => c.priorScore), 1e-9);

    return candidates
      .map((candidate) => {
        const tokens = tokenize(candidate.text);
        const positions = new Map<string, number[]>();
        tokens.forEach((token, i) => {
          if (!querySet.has(token)) return;
          const list = positions.get(token) ?? [];
          list.push(i);
          positions.set(token, list);
        });

        const coverage = positions.size / querySet.size;
        const proximity = computeProximity(positions, tokens.length);
        const prior = candidate.priorScore / maxPrior;

        const score = 0.5 * coverage + 0.2 * proximity + 0.3 * prior;
        return { id: candidate.id, score: Math.max(0, Math.min(1, score)) };
      })
      .sort((a, b) => b.score - a.score);
  }

  async healthCheck(): Promise<{ ok: boolean; detail: string }> {
    return { ok: true, detail: 'heuristic reranker (no model)' };
  }
}

/** 1 when all matched terms are adjacent, decaying to 0 as they spread apart. */
function computeProximity(positions: Map<string, number[]>, documentLength: number): number {
  const firsts = [...positions.values()].map((list) => list[0] as number).sort((a, b) => a - b);
  if (firsts.length <= 1 || documentLength === 0) return firsts.length === 1 ? 0.5 : 0;
  const span = (firsts[firsts.length - 1] as number) - (firsts[0] as number);
  const ideal = firsts.length - 1;
  if (span <= ideal) return 1;
  return Math.max(0, 1 - (span - ideal) / documentLength);
}

// ─── Factory ─────────────────────────────────────────────────────────────────

let instance: Reranker | null = null;

export function getReranker(): Reranker {
  if (!instance) {
    instance =
      config.reranker.provider === 'heuristic' ? new HeuristicReranker() : new LocalCrossEncoderReranker();
    logger.info({ provider: instance.name, model: instance.model }, 'Reranker selected');
  }
  return instance;
}

/**
 * Rerank with an automatic fall back to the heuristic provider.
 *
 * A reranker failure should degrade result ordering, never fail the request —
 * we still hold a perfectly usable fused candidate list.
 */
export async function rerankWithFallback(
  query: string,
  candidates: readonly RerankCandidate[],
): Promise<{ results: RerankResult[]; provider: string; degraded: boolean }> {
  const reranker = getReranker();
  try {
    const results = await reranker.rerank(query, candidates);
    return { results, provider: reranker.name, degraded: false };
  } catch (error) {
    logger.error(
      { error: (error as Error).message, provider: reranker.name },
      'Reranker failed — falling back to heuristic ordering',
    );
    const fallback = new HeuristicReranker();
    const results = await fallback.rerank(query, candidates);
    return { results, provider: `${fallback.name} (fallback)`, degraded: true };
  }
}

export { HeuristicReranker, LocalCrossEncoderReranker };
