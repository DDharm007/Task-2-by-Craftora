/**
 * Answer rendering: streamed Markdown, citations, copy/export and playback.
 *
 * Citation markers like `[2]` in the model output are turned into buttons that
 * scroll to and highlight the matching chunk, so a claim can be traced back to
 * its source in one click.
 */
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import {
  Check,
  Copy,
  Download,
  ShieldAlert,
  Sparkles,
  Volume2,
  Square,
  Loader2,
  ChevronDown,
  ChevronRight,
} from 'lucide-react';
import type { AnswerSource, AnswerStatus, Citation } from '@goarag/shared';
import { copyToClipboard, downloadFile, cn } from '@/lib/utils';
import { Card, CardHeader, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/primitives';
import { useSession } from '@/store/session';
import { resolveBcp47 } from '@/hooks/useVoiceTTS';
import { useTTS } from '@/components/voice/TTSProvider';

function statusBadge(status: AnswerStatus) {
  switch (status) {
    case 'answered':
      return { tone: 'success' as const, label: 'Answered' };
    case 'insufficient_context':
      return { tone: 'warning' as const, label: 'No evidence' };
    case 'low_confidence':
      return { tone: 'warning' as const, label: 'Low confidence' };
    case 'unverified':
      // Warning, not success: the answer exists but nothing in the indexed
      // corpus backs it.
      return { tone: 'warning' as const, label: 'Not from corpus' };
    case 'blocked':
      return { tone: 'danger' as const, label: 'Blocked' };
  }
}

/** Plain-language note on where an ungrounded answer actually came from. */
function sourceNotice(result: { status: AnswerStatus; answerSource: AnswerSource }): string | null {
  if (result.status !== 'unverified') return null;
  return result.answerSource === 'web'
    ? 'Retrieval found nothing relevant in the indexed corpus. This answer comes from a web search and is not grounded in your documents — treat it as unverified.'
    : 'Retrieval found nothing relevant in the indexed corpus. This answer comes from the model’s own knowledge, with no supporting sources — treat it as unverified.';
}

/**
 * Split text on `[n]` markers so each becomes an interactive element.
 * Applied to text nodes only, so it never breaks Markdown structure.
 */
function renderWithCitations(
  text: string,
  citations: Citation[],
  onCite: (chunkId: string) => void,
): React.ReactNode {
  if (citations.length === 0 || !text.includes('[')) return text;

  const parts = text.split(/(\[\d+\])/g);
  return parts.map((part, index) => {
    const match = /^\[(\d+)\]$/.exec(part);
    if (!match) return part;

    const number = Number(match[1]);
    const citation = citations.find((item) => item.index === number);
    if (!citation) return part;

    return (
      <button
        key={`${number}-${index}`}
        type="button"
        onClick={() => onCite(citation.chunkId)}
        title={`${citation.source} — ${citation.topic}`}
        className="mx-0.5 inline-flex h-4 min-w-4 items-center justify-center rounded border border-border bg-subtle px-1 align-super font-mono text-[10px] font-medium text-ink-secondary transition-colors hover:border-ink hover:text-ink"
      >
        {number}
      </button>
    );
  });
}

export function AnswerPanel() {
  const answer = useSession((state) => state.streamedAnswer);
  const reasoning = useSession((state) => state.streamedReasoning);
  const isStreaming = useSession((state) => state.isStreaming);
  const result = useSession((state) => state.result);
  const inspectChunk = useSession((state) => state.inspectChunk);

  const [copied, setCopied] = useState(false);
  const [reasoningOpen, setReasoningOpen] = useState(false);
  const tts = useTTS();

  const citations = result?.citations ?? [];
  const badge = result ? statusBadge(result.status) : null;

  const markdownComponents = useMemo(
    () => ({
      // Intercept text nodes to inject citation buttons.
      p: ({ children }: { children?: React.ReactNode }) => (
        <p className="mb-3 last:mb-0">{mapChildren(children)}</p>
      ),
      li: ({ children }: { children?: React.ReactNode }) => (
        <li className="mb-1">{mapChildren(children)}</li>
      ),
      ul: ({ children }: { children?: React.ReactNode }) => (
        <ul className="mb-3 list-disc space-y-0.5 pl-5 last:mb-0">{children}</ul>
      ),
      ol: ({ children }: { children?: React.ReactNode }) => (
        <ol className="mb-3 list-decimal space-y-0.5 pl-5 last:mb-0">{children}</ol>
      ),
      code: ({ children }: { children?: React.ReactNode }) => (
        <code className="rounded bg-subtle px-1 py-0.5 font-mono text-[0.85em]">{children}</code>
      ),
      pre: ({ children }: { children?: React.ReactNode }) => (
        <pre className="mb-3 overflow-x-auto rounded border border-border bg-subtle p-3 font-mono text-xs">
          {children}
        </pre>
      ),
      a: ({ children, href }: { children?: React.ReactNode; href?: string }) => (
        <a href={href} target="_blank" rel="noreferrer noopener" className="underline underline-offset-2">
          {children}
        </a>
      ),
      table: ({ children }: { children?: React.ReactNode }) => (
        <div className="mb-3 overflow-x-auto">
          <table className="w-full border-collapse text-xs">{children}</table>
        </div>
      ),
      th: ({ children }: { children?: React.ReactNode }) => (
        <th className="border border-border bg-subtle px-2 py-1 text-left font-medium">{children}</th>
      ),
      td: ({ children }: { children?: React.ReactNode }) => (
        <td className="border border-border px-2 py-1">{children}</td>
      ),
      h1: ({ children }: { children?: React.ReactNode }) => (
        <h3 className="mb-2 mt-4 text-sm font-semibold first:mt-0">{children}</h3>
      ),
      h2: ({ children }: { children?: React.ReactNode }) => (
        <h3 className="mb-2 mt-4 text-sm font-semibold first:mt-0">{children}</h3>
      ),
      h3: ({ children }: { children?: React.ReactNode }) => (
        <h4 className="mb-1.5 mt-3 text-xs font-semibold first:mt-0">{children}</h4>
      ),
    }),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [citations],
  );

  function mapChildren(children: React.ReactNode): React.ReactNode {
    if (typeof children === 'string') return renderWithCitations(children, citations, inspectChunk);
    if (Array.isArray(children)) {
      return children.map((child, index) =>
        typeof child === 'string' ? (
          <span key={index}>{renderWithCitations(child, citations, inspectChunk)}</span>
        ) : (
          child
        ),
      );
    }
    return children;
  }

  const handleCopy = async () => {
    if (await copyToClipboard(answer)) {
      setCopied(true);
      setTimeout(() => setCopied(false), 1600);
    }
  };

  const handleExport = () => {
    if (!result) return;
    const lines = [
      `# GoaRAG answer`,
      '',
      `**Question:** ${result.query}`,
      `**Status:** ${result.status}`,
      `**Confidence:** ${(result.confidence.overall * 100).toFixed(1)}%`,
      `**Model:** ${result.model}`,
      `**Latency:** ${Math.round(result.latency.total)}ms`,
      '',
      '## Answer',
      '',
      result.answer,
      '',
      '## Sources',
      '',
      ...result.citations.map(
        (citation) =>
          `${citation.index}. **${citation.topic}** — ${citation.source} (${citation.language}, score ${citation.score.toFixed(3)})\n   > ${citation.snippet}`,
      ),
      '',
      '## Retrieval',
      '',
      ...result.chunks.map(
        (chunk, index) =>
          `${index + 1}. [${chunk.metadata.strategy}] rerank=${chunk.rerankScore?.toFixed(3) ?? 'n/a'} dense=${chunk.denseScore?.toFixed(3) ?? 'n/a'} sparse=${chunk.sparseScore?.toFixed(3) ?? 'n/a'} — ${chunk.metadata.documentId}`,
      ),
    ];
    downloadFile(`goarag-${result.requestId.slice(0, 8)}.md`, lines.join('\n'), 'text/markdown');
  };

  /**
   * Read the answer aloud.
   *
   * Server-side neural TTS (Sarvam, then ElevenLabs) with the browser's own
   * voice as the fallback — see `useVoiceTTS`.
   */
  const handleSpeak = useCallback(() => {
    // Also cancels a synthesis request that has not returned yet, so a slow
    // clip cannot start playing after the user has pressed stop.
    if (tts.isSpeaking || tts.isLoading) {
      tts.stop();
      return;
    }

    const spoken = answer
      .replace(/\[\d+\]/g, '')
      .replace(/[*_`#>]/g, '')
      .trim();
    if (!spoken) return;

    // Map the STT language code to a BCP-47 tag the TTS engine understands.
    const detectedLang = result?.transcription?.languageCode ?? undefined;
    const bcp47 = resolveBcp47(detectedLang);
    tts.speak(spoken, bcp47);
  }, [tts, answer, result]);

  // Stop playback when the component unmounts.
  //
  // Deliberately empty deps with a ref: listing `tts` here re-ran the cleanup on
  // every render, cancelling playback the instant it started and re-rendering
  // into the same cancel again.
  const ttsRef = useRef(tts);
  ttsRef.current = tts;
  useEffect(() => () => ttsRef.current.stop(), []);

  const canSpeak = tts.isSupported;
  const isRefusal = result?.status === 'insufficient_context' || result?.status === 'low_confidence';
  const notice = result && !isStreaming ? sourceNotice(result) : null;

  return (
    <Card>
      <CardHeader
        title="Answer"
        icon={result?.status === 'blocked' ? ShieldAlert : Sparkles}
        description={result ? `${result.model.split('/').pop()}` : 'Grounded in retrieved context only'}
        action={
          <>
            {badge ? <Badge tone={badge.tone}>{badge.label}</Badge> : null}
            {canSpeak && answer && !isStreaming ? (
              <Button
                variant="ghost"
                size="icon-sm"
                onClick={handleSpeak}
                aria-label={
                  tts.isLoading
                    ? 'Preparing audio — click to cancel'
                    : tts.isSpeaking
                      ? 'Stop playback'
                      : 'Read answer aloud'
                }
                aria-busy={tts.isLoading}
                title={
                  tts.isLoading
                    ? 'Preparing audio…'
                    : tts.isSpeaking
                      ? 'Stop'
                      : 'Read aloud — all languages'
                }
              >
                {tts.isLoading ? (
                  <Loader2 className="animate-spin" />
                ) : tts.isSpeaking ? (
                  <Square className="fill-current" />
                ) : (
                  <Volume2 />
                )}
              </Button>
            ) : null}
            {answer ? (
              <Button variant="ghost" size="icon-sm" onClick={handleCopy} aria-label="Copy answer">
                {copied ? <Check className="text-success" /> : <Copy />}
              </Button>
            ) : null}
            {result ? (
              <Button variant="ghost" size="icon-sm" onClick={handleExport} aria-label="Export as Markdown">
                <Download />
              </Button>
            ) : null}
          </>
        }
      />

      <CardContent>
        {reasoning ? (
          <div className="mb-3 rounded border border-border bg-subtle">
            <button
              type="button"
              onClick={() => setReasoningOpen((open) => !open)}
              className="flex w-full items-center gap-1.5 px-3 py-2 text-left text-xs font-medium text-ink-secondary hover:text-ink"
            >
              {reasoningOpen ? <ChevronDown className="size-3.5" /> : <ChevronRight className="size-3.5" />}
              Model reasoning
              <span className="ml-auto font-mono text-2xs text-ink-tertiary">
                {reasoning.length.toLocaleString()} chars
              </span>
            </button>
            {reasoningOpen ? (
              <p className="whitespace-pre-wrap border-t border-border px-3 py-2 font-mono text-2xs leading-relaxed text-ink-secondary">
                {reasoning}
              </p>
            ) : null}
          </div>
        ) : null}

        {notice ? (
          <div className="mb-3 flex items-start gap-2 rounded border border-warning-border bg-warning-subtle p-2.5 text-xs text-warning">
            <ShieldAlert className="mt-px size-3.5 shrink-0" />
            <span>{notice}</span>
          </div>
        ) : null}

        {answer ? (
          <div
            lang={/[\u0900-\u097F]/.test(answer) ? 'hi' : undefined}
            className={cn(
              'text-sm leading-relaxed text-ink',
              isRefusal && 'text-ink-secondary',
            )}
          >
            <ReactMarkdown remarkPlugins={[remarkGfm]} components={markdownComponents}>
              {answer}
            </ReactMarkdown>
            {isStreaming ? (
              <span className="ml-0.5 inline-block h-4 w-1.5 translate-y-0.5 bg-ink" aria-hidden />
            ) : null}
          </div>
        ) : isStreaming ? (
          <p className="text-sm text-ink-tertiary">Waiting for the model…</p>
        ) : (
          <p className="text-sm text-ink-tertiary">
            Ask a question to see a grounded answer with citations.
          </p>
        )}

        {citations.length > 0 && !isStreaming ? (
          <div className="mt-4 border-t border-border pt-3">
            <p className="label mb-2">Sources</p>
            <ol className="space-y-1.5">
              {citations.map((citation) => (
                <li key={citation.chunkId}>
                  <button
                    type="button"
                    onClick={() => inspectChunk(citation.chunkId)}
                    className="group flex w-full items-baseline gap-2 text-left"
                  >
                    <span className="flex size-4 shrink-0 items-center justify-center rounded border border-border bg-subtle font-mono text-[10px] text-ink-secondary">
                      {citation.index}
                    </span>
                    <span className="min-w-0 flex-1">
                      <span className="block truncate text-xs text-ink group-hover:underline">
                        {citation.topic}
                      </span>
                      <span className="block truncate font-mono text-2xs text-ink-tertiary">
                        {citation.source}
                      </span>
                    </span>
                    <span className="shrink-0 font-mono text-2xs text-ink-secondary">
                      {citation.score.toFixed(3)}
                    </span>
                  </button>
                </li>
              ))}
            </ol>
          </div>
        ) : null}
      </CardContent>
    </Card>
  );
}
