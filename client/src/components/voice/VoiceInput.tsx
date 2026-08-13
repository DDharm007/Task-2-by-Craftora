/**
 * VoiceInput — Scribe STT control (ElevenLabs Scribe via backend).
 *
 * Records audio using MediaRecorder and uploads to /api/transcribe which
 * calls ElevenLabs Scribe for high-accuracy speech-to-text.
 */
import { useCallback, useEffect, useRef } from 'react';
import { AlertCircle, Mic, Radio, Square, X } from 'lucide-react';
import { cn, formatDuration, themeColor } from '@/lib/utils';
import { Button } from '@/components/ui/button';
import { useAudioRecorder } from '@/hooks/useAudioRecorder';
import { useTheme } from '@/hooks/useTheme';

interface VoiceInputProps {
  /** Called with the final text if live STT is used (deprecated). */
  onTranscript?: (text: string) => void;
  /** Called with recorded audio blob for ElevenLabs Scribe transcription. */
  onRecorded: (audio: Blob) => void;
  disabled?: boolean;
  autoResumeListen?: number;
}

// ── Waveform canvas ──────────────────────────────────────────────────────────

function Waveform({
  data,
  active,
  color,
  inactiveColor,
}: {
  data: number[];
  active: boolean;
  color: string;
  inactiveColor: string;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const ratio = window.devicePixelRatio || 1;
    const { width, height } = canvas.getBoundingClientRect();
    canvas.width = width * ratio;
    canvas.height = height * ratio;
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    ctx.clearRect(0, 0, width, height);

    const bars = data.length;
    const gap = 2;
    const barW = Math.max(1.5, (width - gap * (bars - 1)) / bars);
    const cy = height / 2;

    ctx.fillStyle = active ? color : inactiveColor;

    data.forEach((amp, i) => {
      const bh = Math.max(2, amp * (height - 4));
      const x = i * (barW + gap);
      const y = cy - bh / 2;
      const r = Math.min(barW / 2, 1.5);
      ctx.beginPath();
      ctx.roundRect(x, y, barW, bh, r);
      ctx.fill();
    });
  }, [data, active, color, inactiveColor]);

  return <canvas ref={canvasRef} className="h-10 w-full" aria-hidden />;
}

// ── Main component ────────────────────────────────────────────────────────────

export function VoiceInput({ onRecorded, disabled = false }: VoiceInputProps) {
  // ── Upload STT (MediaRecorder → ElevenLabs Scribe backend)
  const recorder = useAudioRecorder();
  // Re-render on theme change so the canvas below re-reads the CSS variables
  // it can't respond to on its own — `theme` itself is unused, subscribing is
  // the point.
  useTheme();

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
      <div className="flex items-start gap-2 rounded border border-warning-border bg-warning-subtle p-3 text-xs text-warning">
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
        <div className="flex items-center gap-1.5 rounded border border-border bg-subtle px-2.5 py-1 text-xs font-medium text-ink">
          <Radio className="size-3 text-brand" />
          <span>Scribe</span>
        </div>
      </div>

      {/* ── Controls row */}
      <div className="flex items-center gap-3">
        {/* Mic button */}
        <Button
          variant={anyActive ? 'danger' : 'primary'}
          size="icon"
          onClick={handleUploadToggle}
          disabled={disabled || isUploading || !recorder.isSupported}
          loading={isUploading}
          aria-label={anyActive ? 'Stop' : 'Start voice input'}
          className="size-10 shrink-0 rounded-full"
        >
          {isUploading ? null : anyActive ? <Square className="fill-current" /> : <Mic />}
        </Button>

        {/* Waveform */}
        <div className="min-w-0 flex-1">
          <Waveform
            data={recorder.waveform}
            active={isRecording}
            color={themeColor('waveform')}
            inactiveColor={themeColor('border-strong')}
          />
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
            Recording — <kbd className="font-mono">Space</kbd> to stop,{' '}
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
        <div className="flex items-start gap-2 rounded border border-danger-border bg-danger-subtle p-2.5 text-xs text-danger">
          <AlertCircle className="mt-px size-3.5 shrink-0" />
          <span>{anyError}</span>
        </div>
      ) : null}
    </div>
  );
}
