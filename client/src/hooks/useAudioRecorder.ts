/**
 * Microphone capture with live amplitude analysis.
 *
 * Returns both the recorded audio and a rolling amplitude buffer that drives
 * the waveform. Amplitude is sampled via requestAnimationFrame from a Web
 * Audio AnalyserNode rather than from the MediaRecorder chunks, so the
 * visualisation reflects what the mic hears right now instead of lagging
 * behind the encoder.
 */
import { useCallback, useEffect, useRef, useState } from 'react';

export type RecorderStatus = 'idle' | 'requesting' | 'recording' | 'stopped' | 'denied' | 'unsupported';

/** Number of amplitude samples kept for the waveform. */
const WAVEFORM_BARS = 56;

export interface UseAudioRecorder {
  status: RecorderStatus;
  /** Seconds elapsed in the current recording. */
  elapsed: number;
  /** Rolling amplitudes in [0,1], oldest first. */
  waveform: number[];
  /** Instantaneous level in [0,1]. */
  level: number;
  error: string | null;
  isSupported: boolean;
  start: () => Promise<void>;
  stop: () => Promise<Blob | null>;
  cancel: () => void;
}

/** Pick a container the browser can actually produce. Safari differs from Chrome. */
function pickMimeType(): string {
  if (typeof MediaRecorder === 'undefined') return '';
  const candidates = [
    'audio/webm;codecs=opus',
    'audio/webm',
    'audio/ogg;codecs=opus',
    'audio/mp4',
  ];
  return candidates.find((type) => MediaRecorder.isTypeSupported(type)) ?? '';
}

export function useAudioRecorder(): UseAudioRecorder {
  const [status, setStatus] = useState<RecorderStatus>('idle');
  const [elapsed, setElapsed] = useState(0);
  const [waveform, setWaveform] = useState<number[]>(() => new Array(WAVEFORM_BARS).fill(0));
  const [level, setLevel] = useState(0);
  const [error, setError] = useState<string | null>(null);

  const recorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const streamRef = useRef<MediaStream | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const frameRef = useRef<number | null>(null);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const startedAtRef = useRef(0);
  /** Loudest sample observed during the take, used to detect a dead mic. */
  const peakRef = useRef(0);

  const isSupported =
    typeof navigator !== 'undefined' &&
    typeof navigator.mediaDevices?.getUserMedia === 'function' &&
    typeof MediaRecorder !== 'undefined';

  /** Release the mic, audio graph and timers. */
  const teardown = useCallback(() => {
    if (frameRef.current !== null) {
      cancelAnimationFrame(frameRef.current);
      frameRef.current = null;
    }
    if (timerRef.current !== null) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
    // Stopping the tracks is what actually turns off the browser's mic
    // indicator — closing the AudioContext alone does not.
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    if (audioContextRef.current && audioContextRef.current.state !== 'closed') {
      void audioContextRef.current.close();
    }
    audioContextRef.current = null;
    analyserRef.current = null;
    recorderRef.current = null;
  }, []);

  useEffect(() => teardown, [teardown]);

  const sampleAmplitude = useCallback(() => {
    const analyser = analyserRef.current;
    if (!analyser) return;

    const buffer = new Uint8Array(analyser.fftSize);
    analyser.getByteTimeDomainData(buffer);

    // RMS over the window, centred on 128 (silence in unsigned 8-bit PCM).
    let sumSquares = 0;
    for (let i = 0; i < buffer.length; i += 1) {
      const deviation = ((buffer[i] as number) - 128) / 128;
      sumSquares += deviation * deviation;
    }
    const rms = Math.sqrt(sumSquares / buffer.length);
    // Speech RMS rarely exceeds ~0.3, so scale it into a usable visual range.
    const normalized = Math.min(1, rms * 3.2);

    if (normalized > peakRef.current) peakRef.current = normalized;
    setLevel(normalized);
    setWaveform((previous) => [...previous.slice(1), normalized]);

    frameRef.current = requestAnimationFrame(sampleAmplitude);
  }, []);

  const start = useCallback(async () => {
    if (!isSupported) {
      setStatus('unsupported');
      setError('This browser cannot record audio. Try Chrome, Edge or Safari.');
      return;
    }

    setError(null);
    setStatus('requesting');

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
          channelCount: 1,
        },
      });
      streamRef.current = stream;

      const AudioContextCtor =
        window.AudioContext ?? (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
      const audioContext = new AudioContextCtor();
      audioContextRef.current = audioContext;

      const analyser = audioContext.createAnalyser();
      analyser.fftSize = 1024;
      analyser.smoothingTimeConstant = 0.7;
      audioContext.createMediaStreamSource(stream).connect(analyser);
      analyserRef.current = analyser;

      const mimeType = pickMimeType();
      const recorder = new MediaRecorder(stream, mimeType ? { mimeType } : undefined);
      chunksRef.current = [];
      recorder.ondataavailable = (event) => {
        if (event.data.size > 0) chunksRef.current.push(event.data);
      };
      // A timeslice keeps data flowing, so a long recording is not lost if the
      // tab is backgrounded and the recorder is throttled.
      recorder.start(250);
      recorderRef.current = recorder;

      startedAtRef.current = Date.now();
      peakRef.current = 0;
      setElapsed(0);
      setWaveform(new Array(WAVEFORM_BARS).fill(0));
      setStatus('recording');

      timerRef.current = setInterval(() => {
        setElapsed((Date.now() - startedAtRef.current) / 1000);
      }, 100);
      frameRef.current = requestAnimationFrame(sampleAmplitude);
    } catch (err) {
      teardown();
      const name = (err as Error).name;
      if (name === 'NotAllowedError' || name === 'SecurityError') {
        setStatus('denied');
        setError('Microphone access was denied. Allow it in your browser settings and try again.');
      } else if (name === 'NotFoundError') {
        setStatus('unsupported');
        setError('No microphone was found.');
      } else {
        setStatus('idle');
        setError((err as Error).message || 'Could not start recording.');
      }
    }
  }, [isSupported, sampleAmplitude, teardown]);

  const stop = useCallback(async (): Promise<Blob | null> => {
    const recorder = recorderRef.current;
    if (!recorder || recorder.state === 'inactive') {
      teardown();
      setStatus('idle');
      return null;
    }

    // `requestData()` flushes whatever the encoder is holding before we stop,
    // so the tail of the recording is never dropped.
    if (recorder.state === 'recording') {
      try {
        recorder.requestData();
      } catch {
        // Not supported everywhere; the timeslice already flushes periodically.
      }
    }

    const blob = await new Promise<Blob>((resolve) => {
      recorder.onstop = () => {
        resolve(new Blob(chunksRef.current, { type: recorder.mimeType || 'audio/webm' }));
      };
      recorder.stop();
    });

    const durationSeconds = (Date.now() - startedAtRef.current) / 1000;
    teardown();
    setStatus('stopped');
    setLevel(0);

    // Reject recordings that cannot contain speech, with a reason the user can
    // act on — otherwise this surfaces later as a confusing empty transcript.
    if (blob.size === 0) {
      setError('Nothing was recorded. Check that the right microphone is selected.');
      return null;
    }
    if (durationSeconds < 0.4) {
      setError('That recording was too short. Hold the button and speak for at least a second.');
      return null;
    }
    // An opus-encoded webm of pure silence is still a few hundred bytes, so
    // size alone is not conclusive — the observed peak level is.
    if (peakRef.current < 0.02) {
      setError(
        'No sound reached the microphone. Check that the correct input device is selected and unmuted.',
      );
      return null;
    }
    if (blob.size < 1_200) {
      setError('The recording was too quiet to transcribe. Move closer to the microphone and try again.');
      return null;
    }

    return blob;
  }, [teardown]);

  const cancel = useCallback(() => {
    const recorder = recorderRef.current;
    if (recorder && recorder.state !== 'inactive') {
      recorder.onstop = null;
      recorder.stop();
    }
    chunksRef.current = [];
    teardown();
    setStatus('idle');
    setElapsed(0);
    setLevel(0);
    setWaveform(new Array(WAVEFORM_BARS).fill(0));
  }, [teardown]);

  return { status, elapsed, waveform, level, error, isSupported, start, stop, cancel };
}

export { WAVEFORM_BARS };
