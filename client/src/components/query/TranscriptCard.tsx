/**
 * Transcript display.
 *
 * Surfaces STT confidence and detected language alongside the text. A low
 * transcript confidence is the most common root cause of a surprising answer,
 * so it is shown explicitly rather than buried.
 */
import { Languages, Mic } from 'lucide-react';
import type { TranscriptionResult } from '@goarag/shared';
import { languageName } from '@goarag/shared';
import { formatMs, formatPercent, scoreTone } from '@/lib/utils';
import { Card, CardHeader, CardContent } from '@/components/ui/card';
import { Badge, Meter } from '@/components/ui/primitives';

export function TranscriptCard({ transcription }: { transcription: TranscriptionResult }) {
  const tone = scoreTone(transcription.confidence);
  // ElevenLabs reports ISO-639-3 ("hin"); the corpus uses FLORES tags.
  const language = transcription.languageCode
    ? languageName(transcription.languageCode) === transcription.languageCode
      ? transcription.languageCode.toUpperCase()
      : languageName(transcription.languageCode)
    : 'auto';

  return (
    <Card>
      <CardHeader
        title="Transcript"
        icon={Mic}
        description={`${transcription.provider} · ${transcription.model}`}
        action={
          <Badge tone={tone}>
            {formatPercent(transcription.confidence)} confident
          </Badge>
        }
      />
      <CardContent className="space-y-3">
        <p
          lang={
            transcription.languageCode?.toLowerCase().startsWith('hin') || /[\u0900-\u097F]/.test(transcription.text)
              ? 'hi'
              : undefined
          }
          className="text-sm leading-relaxed text-ink"
        >
          {transcription.text || <span className="text-ink-tertiary">No speech detected</span>}
        </p>

        <Meter value={transcription.confidence} />

        <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-2xs text-ink-secondary">
          <span className="flex items-center gap-1">
            <Languages className="size-3" />
            {language}
            {transcription.languageProbability !== null
              ? ` (${formatPercent(transcription.languageProbability)})`
              : ''}
          </span>
          {transcription.durationSeconds !== null ? (
            <span>{transcription.durationSeconds.toFixed(1)}s audio</span>
          ) : null}
          <span>{transcription.words.filter((w) => w.type === 'word').length} words</span>
          <span>{formatMs(transcription.latencyMs)} to transcribe</span>
        </div>
      </CardContent>
    </Card>
  );
}
