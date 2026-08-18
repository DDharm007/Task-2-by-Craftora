/**
 * VoiceInput — Scribe STT control (ElevenLabs Scribe via backend).
 *
 * Records audio using MediaRecorder and uploads to /api/transcribe which
 * calls ElevenLabs Scribe for high-accuracy speech-to-text.
 */
import { useCallback, useEffect, useMemo } from 'react';
import { AlertCircle, ArrowUp, Loader2, Podcast, Radio, X } from 'lucide-react';
import { cn, formatDuration, themeColorHex } from '@/lib/utils';
import { Button } from '@/components/ui/button';
import { SpecularButton } from '@/components/ui/SpecularButton';
import { useAudioRecorder } from '@/hooks/useAudioRecorder';
import { useTheme } from '@/hooks/useTheme';
import { Waveform } from '@/components/voice/Waveform';

interface VoiceInputProps {
  /** Called with the final text if live STT is used (deprecated). */
  onTranscript?: (text: string) => void;
  /** Called with recorded audio blob for ElevenLabs Scribe transcription. */
  onRecorded: (audio: Blob) => void;
  disabled?: boolean;
  autoResumeListen?: number;
}

// ── Main component ────────────────────────────────────────────────────────────

export function VoiceInput({ onRecorded, disabled = false }: VoiceInputProps) {
  // ── Upload STT (MediaRecorder → ElevenLabs Scribe backend)
  const recorder = useAudioRecorder();
  // The mic button's shader colours are read from CSS variables, which a GL
  // uniform can't subscribe to on its own — recompute them whenever the
  // theme changes so the highlight doesn't stay stuck in Light forever.
  const { theme } = useTheme();
  const shaderColors = useMemo(
    () => ({
      idleBase: themeColorHex('border-strong'),
      idleLine: themeColorHex('ink'),
      idleText: themeColorHex('ink'),
      activeBase: themeColorHex('danger-border'),
      activeLine: themeColorHex('danger'),
      activeText: themeColorHex('danger'),
    }),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [theme],
  );

  // ── Handlers for upload mode
  const handleUploadToggle = useCallback(async () => {
    if (recorder.status === 'recording') {
      const blob = await recorder.stop();
      if (blob) onRecorded(blob);
    } else {
      await recorder.start();
    }
  }, [recorder, onRecorded]);

  // ── Keyboard shortcuts
  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement | null;
      if (
        target?.tagName === 'INPUT' ||
        target?.tagName === 'TEXTAREA' ||
        target?.isContentEditable ||
        disabled
      )
        return;

      if (e.code === 'Space') {
        e.preventDefault();
        void handleUploadToggle();
      } else if (e.code === 'Escape') {
        e.preventDefault();
        recorder.cancel();
      }
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  });

  if (recorder.isSupported === false) {
    return (
      <div className="flex items-start gap-2 rounded-md border border-warning-border bg-warning-subtle p-3 text-xs text-warning">
        <AlertCircle className="mt-px size-4 shrink-0" />
        <span>
          Microphone access is not supported in this browser. Use Chrome, Edge, or Safari, or type
          your question below.
        </span>
      </div>
    );
  }

  const isRecording = recorder.status === 'recording';
  const isUploading = recorder.status === 'requesting';
  const anyActive = isRecording;
  const anyError = recorder.error ?? null;

  return (
    <div className="space-y-3">
      {/* ── Mode indicator */}
      <div className="flex items-center justify-between gap-2">
        <div className="flex items-center gap-1.5 rounded-full bg-subtle px-2.5 py-1 text-xs font-medium text-ink">
          <Radio className="size-3 text-brand" />
          <span>Scribe</span>
        </div>
      </div>

      {/* ── Controls row */}
      <div className="flex items-center gap-3">
        {/* Idle: a circular mic button — transparent, so the specular ring is
            the whole affordance rather than competing with a filled disc
            behind it. Recording: a horizontal pill labelled "Send" rather
            than a small stop-icon circle, because the click does two things
            at once (stop the recording *and* hand it to the pipeline) and a
            bare square icon undersold the second half of that. `SpecularButton`
            clamps its corner radius to half the shorter side on its own, so a
            short wide button becomes a stadium shape with no extra styling. */}
        {anyActive ? (
          <SpecularButton
            onClick={handleUploadToggle}
            disabled={disabled || isUploading}
            aria-label="Stop recording and send"
            className="h-10 shrink-0 gap-2 border border-danger-border px-4 text-sm font-medium text-danger"
            baseColor={shaderColors.activeBase}
            lineColor={shaderColors.activeLine}
            textColor={shaderColors.activeText}
            autoAnimate
            shineSize={14}
            shineFade={50}
          >
            <ArrowUp className="size-4" />
            Send
          </SpecularButton>
        ) : (
          <SpecularButton
            onClick={handleUploadToggle}
            disabled={disabled || isUploading || !recorder.isSupported}
            loading={isUploading}
            aria-label="Start voice input"
            className="size-10 shrink-0 border border-border text-ink"
            baseColor={shaderColors.idleBase}
            lineColor={shaderColors.idleLine}
            textColor={shaderColors.idleText}
            shineSize={14}
            shineFade={50}
          >
            {isUploading ? (
              <Loader2 className="size-4 animate-spin" />
            ) : (
              <Podcast className="size-4" />
            )}
          </SpecularButton>
        )}

        {/* Waveform */}
        <div className="min-w-0 flex-1">
          <Waveform data={recorder.waveform} active={isRecording} />
        </div>

        {/* Timer / stop */}
        <div className="flex shrink-0 items-center gap-2">
          <span
            className={cn(
              'font-mono text-sm tabular-nums',
              anyActive ? 'text-ink' : 'text-ink-tertiary',
            )}
          >
            {formatDuration(recorder.elapsed)}
          </span>
          {anyActive && (
            <Button
              variant="ghost"
              size="icon-sm"
              onClick={() => recorder.cancel()}
              aria-label="Discard"
            >
              <X />
            </Button>
          )}
        </div>
      </div>

      {/* ── Hint text */}
      <p className="text-2xs text-ink-secondary">
        {isRecording ? (
          <>
            Recording — <kbd className="font-mono">Space</kbd> or Send to finish,{' '}
            <kbd className="font-mono">Esc</kbd> to discard.
          </>
        ) : (
          <>
            <kbd className="font-mono">Space</kbd> or click mic to record. Audio goes to ElevenLabs
            Scribe for higher accuracy.
          </>
        )}
      </p>

      {/* ── Error */}
      {anyError ? (
        <div className="flex items-start gap-2 rounded-md border border-danger-border bg-danger-subtle p-2.5 text-xs text-danger">
          <AlertCircle className="mt-px size-3.5 shrink-0" />
          <span>{anyError}</span>
        </div>
      ) : null}
    </div>
  );
}
