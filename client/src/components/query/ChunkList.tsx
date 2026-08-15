/**
 * Retrieval visualisation.
 *
 * Each retrieved chunk shows its dense, sparse, fused and rerank scores side
 * by side, plus how reranking moved it. Seeing rank 7 promoted to 1 by the
 * cross-encoder is the clearest demonstration that the second stage is doing
 * real work.
 */
import { ArrowDown, ArrowUp, FileText, Minus, Search } from 'lucide-react';
import type { RetrievedChunk } from '@goarag/shared';
import { languageName } from '@goarag/shared';
import { cn, formatPercent, scoreTone, truncateWords } from '@/lib/utils';
import { Card, CardHeader, CardContent } from '@/components/ui/card';
import { Badge, EmptyState, Meter, Tooltip } from '@/components/ui/primitives';
import { useSession } from '@/store/session';

/** Arrow showing how far reranking moved a chunk. */
function RankDelta({ chunk }: { chunk: RetrievedChunk }) {
  if (chunk.rankAfterRerank === null) return null;
  const delta = chunk.rankBeforeRerank - chunk.rankAfterRerank;

  if (delta === 0) {
    return (
      <Tooltip content="Rerank kept this chunk at the same rank">
        <span className="flex items-center gap-0.5 font-mono text-2xs text-ink-tertiary">
          <Minus className="size-3" />0
        </span>
      </Tooltip>
    );
  }

  const promoted = delta > 0;
  return (
    <Tooltip
      content={`Rerank moved this from #${chunk.rankBeforeRerank} to #${chunk.rankAfterRerank}`}
    >
      <span
        className={cn(
          'flex items-center gap-0.5 font-mono text-2xs',
          promoted ? 'text-success' : 'text-warning',
        )}
      >
        {promoted ? <ArrowUp className="size-3" /> : <ArrowDown className="size-3" />}
        {Math.abs(delta)}
      </span>
    </Tooltip>
  );
}

/** One score column. */
function Score({ label, value, hint }: { label: string; value: number | null; hint: string }) {
  return (
    <Tooltip content={hint}>
      <div className="min-w-0">
        <div className="label mb-0.5 !text-[10px]">{label}</div>
        <div className="font-mono text-xs text-ink">{value === null ? '—' : value.toFixed(3)}</div>
      </div>
    </Tooltip>
  );
}

function ChunkRow({ chunk, index }: { chunk: RetrievedChunk; index: number }) {
  const inspected = useSession((state) => state.inspectedChunkId);
  const inspectChunk = useSession((state) => state.inspectChunk);
  const isOpen = inspected === chunk.id;
  const tone = scoreTone(chunk.score);

  return (
    <li
      id={`chunk-${chunk.id}`}
      className={cn(
        'scroll-mt-20 rounded border transition-colors',
        isOpen ? 'border-ink bg-subtle' : 'border-border bg-card hover:border-border-strong',
      )}
    >
      <button
        type="button"
        onClick={() => inspectChunk(isOpen ? null : chunk.id)}
        className="w-full px-3 py-2.5 text-left"
        aria-expanded={isOpen}
      >
        <div className="flex items-start gap-2.5">
          <span className="mt-0.5 flex size-5 shrink-0 items-center justify-center rounded border border-border bg-subtle font-mono text-[10px] font-medium text-ink-secondary">
            {index + 1}
          </span>

          <div className="min-w-0 flex-1">
            <div className="mb-1 flex flex-wrap items-center gap-1.5">
              <Badge tone="neutral">{chunk.metadata.strategy}</Badge>
              <Badge tone="neutral">{languageName(chunk.metadata.language)}</Badge>
              {chunk.matchedBy.length === 2 ? (
                <Tooltip content="Found by both dense and keyword search — the strongest pre-rerank signal">
                  <Badge tone="success">hybrid</Badge>
                </Tooltip>
              ) : (
                <Badge tone="neutral">{chunk.matchedBy[0] ?? 'n/a'}</Badge>
              )}
              {chunk.parentText ? (
                <Tooltip content="A wider parent span was supplied to the model as context">
                  <Badge tone="neutral">+parent</Badge>
                </Tooltip>
              ) : null}
              <span className="ml-auto flex items-center gap-2">
                <RankDelta chunk={chunk} />
                <span
                  className={cn(
                    'font-mono text-xs font-medium',
                    // Explicit classes — Tailwind cannot see interpolated names.
                    tone === 'success' && 'text-success',
                    tone === 'warning' && 'text-warning',
                    tone === 'danger' && 'text-danger',
                  )}
                >
                  {formatPercent(chunk.score, 1)}
                </span>
              </span>
            </div>

            <p className="text-xs leading-relaxed text-ink-secondary">
              {isOpen ? chunk.text : truncateWords(chunk.text, 34)}
            </p>

            {/* The score text above is already tinted by verdict, so the bar
                carries magnitude — letting two chunks be ranked by eye. */}
            <Meter value={chunk.score} className="mt-2" />
          </div>
        </div>
      </button>

      {isOpen ? (
        <div className="border-t border-border px-3 py-2.5">
          <div className="grid grid-cols-4 gap-3">
            <Score
              label="Dense"
              value={chunk.denseScore}
              hint="Cosine similarity between the query and chunk embeddings (BGE-M3)"
            />
            <Score
              label="Sparse"
              value={chunk.sparseScore}
              hint="BM25 keyword relevance. Unbounded — higher is better"
            />
            <Score
              label="Fused"
              value={chunk.fusedScore}
              hint="Reciprocal Rank Fusion of the dense and sparse rankings"
            />
            <Score
              label="Rerank"
              value={chunk.rerankScore}
              hint="Cross-encoder relevance — the score used for the final ordering"
            />
          </div>

          {chunk.parentText ? (
            <div className="mt-3 border-t border-border pt-2.5">
              <p className="label mb-1">Parent context sent to the model</p>
              <p className="max-h-40 overflow-y-auto text-xs leading-relaxed text-ink-secondary">
                {chunk.parentText}
              </p>
            </div>
          ) : null}

          <dl className="mt-3 grid grid-cols-2 gap-x-4 border-t border-border pt-2.5 text-2xs sm:grid-cols-3">
            {[
              ['Document', chunk.metadata.documentId],
              ['Passage', chunk.metadata.passageId],
              ['Chunk index', String(chunk.metadata.chunkIndex)],
              ['Tokens', String(chunk.metadata.tokenCount)],
              ['Chars', `${chunk.metadata.charStart}–${chunk.metadata.charEnd}`],
              ['Source', chunk.metadata.source],
              ['Topic', chunk.metadata.topic],
              ['Parent', chunk.metadata.parentChunk ? `${chunk.metadata.parentChunk.slice(0, 8)}…` : 'none'],
              ['Ground truth', chunk.metadata.isSelected ? 'relevant' : 'not labelled'],
            ].map(([label, value]) => (
              <div key={label} className="flex flex-col py-0.5">
                <dt className="text-ink-tertiary">{label}</dt>
                <dd className="truncate font-mono text-ink-secondary" title={value}>
                  {value}
                </dd>
              </div>
            ))}
          </dl>
        </div>
      ) : null}
    </li>
  );
}

export function ChunkList() {
  const chunks = useSession((state) => state.chunks);
  const isStreaming = useSession((state) => state.isStreaming);

  return (
    <Card>
      <CardHeader
        title="Retrieved context"
        icon={FileText}
        description={
          chunks.length > 0
            ? `Top ${chunks.length} after reranking · click a chunk for its scores`
            : 'Hybrid dense + keyword retrieval, then cross-encoder rerank'
        }
        action={chunks.length > 0 ? <Badge tone="neutral">{chunks.length}</Badge> : null}
      />
      <CardContent className={chunks.length === 0 ? 'p-0' : ''}>
        {chunks.length === 0 ? (
          <EmptyState
            icon={Search}
            title={isStreaming ? 'Retrieving…' : 'No chunks retrieved yet'}
            description={
              isStreaming
                ? 'Running dense and keyword search in parallel.'
                : 'Ask a question and the chunks that grounded the answer will appear here, with their scores.'
            }
          />
        ) : (
          <ul className="space-y-2">
            {chunks.map((chunk, index) => (
              <ChunkRow key={chunk.id} chunk={chunk} index={index} />
            ))}
          </ul>
        )}
      </CardContent>
    </Card>
  );
}
