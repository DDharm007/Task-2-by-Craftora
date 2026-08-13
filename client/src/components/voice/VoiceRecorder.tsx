/**
 * Voice capture control.
 *
 * A single mic button, a live waveform, and an elapsed timer. The waveform is
 * driven by real microphone amplitude (see useAudioRecorder) rather than a
 * decorative animation — when it is flat, the mic genuinely is not picking
 * anything up, which makes "why did it not hear me?" self-diagnosing.
 */
import { useEffect, useRef } from 'react';
import { Mic, Square, X, AlertCircle } from 'lucide-react';
import { cn, formatDuration } from '@/lib/utils';
import { Button } from '@/components/ui/button';
import { useAudioRecorder } from '@/hooks/useAudioRecorder';

interface VoiceRecorderProps {
  onRecorded: (audio: Blob) => void;
  disabled?: boolean;
}

/** Bar-graph waveform. Canvas, so a 60fps update does not thrash the DOM. */
function Waveform({ data, active }: { data: number[]; active: boolean }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const context = canvas.getContext('2d');
    if (!context) return;

    const ratio = window.devicePixelRatio || 1;
    const { width, height } = canvas.getBoundingClientRect();
    canvas.width = width * ratio;
    canvas.height = height * ratio;
    context.setTransform(ratio, 0, 0, ratio, 0, 0);
    context.clearRect(0, 0, width, height);

    const bars = data.length;
    const gap = 2;
    const barWidth = Math.max(1.5, (width - gap * (bars - 1)) / bars);
    const centre = height / 2;

    // Solid ink at full opacity when live, muted when idle.
    context.fillStyle = active ? '#111827' : '#D1D5DB';

    data.forEach((amplitude, index) => {
      const barHeight = Math.max(2, amplitude * (height - 4));
      const x = index * (barWidth + gap);
      const y = centre - barHeight / 2;
      // Rounded caps read as a waveform rather than a bar chart.
      const radius = Math.min(barWidth / 2, 1.5);
      context.beginPath();
      context.roundRect(x, y, barWidth, barHeight, radius);
      context.fill();
    });
  }, [data, active]);

  return <canvas ref={canvasRef} className="h-12 w-full" aria-hidden />;
}

export function VoiceRecorder({ onRecorded, disabled = false }: VoiceRecorderProps) {
  const recorder = useAudioRecorder();
  const isRecording = recorder.status === 'recording';
  const isBusy = recorder.status === 'requesting';

  const handleToggle = async () => {
    if (isRecording) {
      const audio = await recorder.stop();
      if (audio) onRecorded(audio);
      return;
    }
    await recorder.start();
  };

  // Space toggles recording, Escape cancels — but never while the user is
  // typing into the composer.
  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement | null;
      const typing =
        target?.tagName === 'INPUT' ||
        target?.tagName === 'TEXTAREA' ||
        target?.isContentEditable;
      if (typing || disabled) return;

      if (event.code === 'Space') {
        event.preventDefault();
        void handleToggle();
      } else if (event.code === 'Escape' && isRecording) {
        event.preventDefault();
        recorder.cancel();
      }
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  });

  if (!recorder.isSupported) {
    return (
      <div className="flex items-start gap-2 rounded border border-warning-border bg-warning-subtle p-3 text-xs text-warning">
        <AlertCircle className="mt-px size-4 shrink-0" />
        <span>
          This browser cannot record audio. Use Chrome, Edge or Safari — or type your question below.
        </span>
      </div>
    );
  }

  return (
    <div className="space-y-3">
      <div className="flex items-center gap-3">
        <Button
          variant={isRecording ? 'danger' : 'primary'}
          size="icon"
          onClick={handleToggle}
          disabled={disabled || isBusy}
          loading={isBusy}
          aria-label={isRecording ? 'Stop recording' : 'Start recording'}
          className="size-10 shrink-0 rounded-full"
        >
          {isBusy ? null : isRecording ? <Square className="fill-current" /> : <Mic />}
        </Button>

        <div className="min-w-0 flex-1">
          <Waveform data={recorder.waveform} active={isRecording} />
        </div>

        <div className="flex shrink-0 items-center gap-2">
          <span
            className={cn(
              'font-mono text-sm tabular-nums',
              isRecording ? 'text-ink' : 'text-ink-tertiary',
            )}
          >
            {formatDuration(recorder.elapsed)}
          </span>
          {isRecording ? (
            <Button
              variant="ghost"
              size="icon-sm"
              onClick={recorder.cancel}
              aria-label="Discard recording"
            >
              <X />
            </Button>
          ) : null}
        </div>
      </div>

      <p className="text-2xs text-ink-secondary">
        {isRecording ? (
          <>
            Recording — press <kbd className="font-mono">Space</kbd> to stop,{' '}
            <kbd className="font-mono">Esc</kbd> to discard.
          </>
        ) : (
          <>
            Press <kbd className="font-mono">Space</kbd> or click the microphone to ask a question
            aloud. Hindi and English both work.
          </>
        )}
      </p>

      {recorder.error ? (
        <div className="flex items-start gap-2 rounded border border-danger-border bg-danger-subtle p-2.5 text-xs text-danger">
          <AlertCircle className="mt-px size-3.5 shrink-0" />
          <span>{recorder.error}</span>
        </div>
      ) : null}
    </div>
  );
}
