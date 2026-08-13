/**
 * Own Speech-to-Text engine — Web Speech API (SpeechRecognition).
 *
 * Provides real-time, continuous transcription directly in the browser using
 * the platform's built-in recognition engine (Google on Chrome/Edge, Apple
 * on Safari). No API key required. Supports 100+ languages automatically.
 *
 * This complements the ElevenLabs Scribe backend path in useAudioRecorder:
 *   • useVoiceSTT  → instant live captions while speaking (this hook)
 *   • useAudioRecorder → records a Blob → server sends to ElevenLabs Scribe
 *
 * The hook degrades gracefully: if SpeechRecognition is not available the
 * `isSupported` flag is false and the UI falls back to the recorded-blob path.
 */
import { useCallback, useEffect, useRef, useState } from 'react';

export type STTStatus =
  | 'idle'
  | 'requesting'
  | 'listening'
  | 'processing'
  | 'stopped'
  | 'denied'
  | 'unsupported'
  | 'error';

export interface VoiceSTTState {
  status: STTStatus;
  /** Live interim transcript (not yet finalised). */
  interim: string;
  /** Finalised transcript words (stable). */
  transcript: string;
  /** Detected language, if the engine reports it. */
  language: string | null;
  error: string | null;
  isSupported: boolean;
  isListening: boolean;
  /** Start listening for the given BCP-47 language. */
  start: (language?: string) => void;
  /** Stop and return the finalised transcript. */
  stop: () => string;
  /** Discard the current recognition session. */
  cancel: () => void;
  /** Clear the stored transcript. */
  clear: () => void;
}

/** Prefix the webkit variant for Safari. */
const SpeechRecognitionCtor: (new () => SpeechRecognition) | undefined =
  typeof window !== 'undefined'
    ? ((
        window as { SpeechRecognition?: new () => SpeechRecognition }
      ).SpeechRecognition ??
      (
        window as {
          webkitSpeechRecognition?: new () => SpeechRecognition;
        }
      ).webkitSpeechRecognition)
    : undefined;

export function useVoiceSTT(): VoiceSTTState {
  const isSupported = Boolean(SpeechRecognitionCtor);

  const [status, setStatus] = useState<STTStatus>(isSupported ? 'idle' : 'unsupported');
  const [interim, setInterim] = useState('');
  const [transcript, setTranscript] = useState('');
  const [language, setLanguage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const recognizerRef = useRef<SpeechRecognition | null>(null);
  const finalRef = useRef('');

  const teardown = useCallback(() => {
    const rec = recognizerRef.current;
    if (!rec) return;
    rec.onresult = null;
    rec.onerror = null;
    rec.onend = null;
    rec.onstart = null;
    try { rec.abort(); } catch { /* ignore */ }
    recognizerRef.current = null;
  }, []);

  useEffect(() => teardown, [teardown]);

  const start = useCallback(
    (lang = 'en-US') => {
      if (!isSupported || !SpeechRecognitionCtor) {
        setStatus('unsupported');
        setError('Speech recognition is not supported in this browser. Try Chrome or Edge.');
        return;
      }

      teardown();
      finalRef.current = '';
      setInterim('');
      setTranscript('');
      setError(null);
      setStatus('requesting');

      const rec = new SpeechRecognitionCtor();
      // Continuous so it keeps listening without a click-per-sentence
      rec.continuous = true;
      // Interim results give the live "what you're saying right now" effect
      rec.interimResults = true;
      rec.lang = lang;
      // The engine sometimes reports multiple alternatives; take the best.
      rec.maxAlternatives = 1;

      rec.onstart = () => setStatus('listening');

      rec.onresult = (event: SpeechRecognitionEvent) => {
        let final = '';
        let live = '';
        for (let i = 0; i < event.results.length; i++) {
          const result = event.results[i];
          if (!result) continue;
          const text = result[0]?.transcript ?? '';
          if (result.isFinal) {
            final += text + ' ';
          } else {
            live += text;
          }
        }
        if (final) {
          finalRef.current += final;
          setTranscript(finalRef.current.trim());
        }
        setInterim(live.trim());
      };

      rec.onerror = (event: SpeechRecognitionErrorEvent) => {
        switch (event.error) {
          case 'not-allowed':
          case 'service-not-allowed':
            setStatus('denied');
            setError('Microphone permission denied. Allow access in browser settings.');
            break;
          case 'no-speech':
            // Common and not an error — just keep listening.
            setInterim('');
            break;
          case 'network':
            setStatus('error');
            setError('Network error — check your internet connection and try again.');
            break;
          case 'audio-capture':
            setStatus('error');
            setError('No microphone found. Plug one in and try again.');
            break;
          case 'aborted':
            // We caused this ourselves; not a user-facing error.
            break;
          default:
            setStatus('error');
            setError(`Recognition error: ${event.error}`);
        }
      };

      rec.onend = () => {
        setInterim('');
        // Only update status if we were still listening (not already stopped by user).
        setStatus((prev) => (prev === 'listening' ? 'stopped' : prev));
      };

      recognizerRef.current = rec;
      try {
        rec.start();
      } catch (err) {
        teardown();
        setStatus('error');
        setError((err as Error).message || 'Could not start recognition.');
      }
    },
    [isSupported, teardown],
  );

  const stop = useCallback((): string => {
    const rec = recognizerRef.current;
    if (rec) {
      try { rec.stop(); } catch { /* ignore */ }
    }
    teardown();
    setStatus('stopped');
    setInterim('');
    const result = finalRef.current.trim();
    if (result) setLanguage(null); // Reset detection (browser doesn't expose it)
    return result;
  }, [teardown]);

  const cancel = useCallback(() => {
    teardown();
    finalRef.current = '';
    setStatus('idle');
    setInterim('');
    setTranscript('');
    setError(null);
  }, [teardown]);

  const clear = useCallback(() => {
    finalRef.current = '';
    setTranscript('');
    setInterim('');
  }, []);

  return {
    status,
    interim,
    transcript,
    language,
    error,
    isSupported,
    isListening: status === 'listening',
    start,
    stop,
    cancel,
    clear,
  };
}

/** All BCP-47 tags the Web Speech API reliably handles in Chrome/Edge. */
export const SPEECH_LANGUAGES: Array<{ code: string; label: string; flag: string }> = [
  { code: 'hi-IN', label: 'हिन्दी', flag: '🇮🇳' },
  { code: 'en-US', label: 'English (US)', flag: '🇺🇸' },
  { code: 'en-IN', label: 'English (India)', flag: '🇮🇳' },
  { code: 'en-GB', label: 'English (UK)', flag: '🇬🇧' },
  { code: 'bn-IN', label: 'বাংলা', flag: '🇮🇳' },
  { code: 'ta-IN', label: 'தமிழ்', flag: '🇮🇳' },
  { code: 'te-IN', label: 'తెలుగు', flag: '🇮🇳' },
  { code: 'mr-IN', label: 'मराठी', flag: '🇮🇳' },
  { code: 'gu-IN', label: 'ગુજરાતી', flag: '🇮🇳' },
  { code: 'kn-IN', label: 'ಕನ್ನಡ', flag: '🇮🇳' },
  { code: 'ml-IN', label: 'മലയാളം', flag: '🇮🇳' },
  { code: 'pa-IN', label: 'ਪੰਜਾਬੀ', flag: '🇮🇳' },
  { code: 'ur-PK', label: 'اردو', flag: '🇵🇰' },
  { code: 'ar-SA', label: 'العربية', flag: '🇸🇦' },
  { code: 'zh-CN', label: '普通话', flag: '🇨🇳' },
  { code: 'zh-TW', label: '繁體中文', flag: '🇹🇼' },
  { code: 'ja-JP', label: '日本語', flag: '🇯🇵' },
  { code: 'ko-KR', label: '한국어', flag: '🇰🇷' },
  { code: 'es-ES', label: 'Español', flag: '🇪🇸' },
  { code: 'fr-FR', label: 'Français', flag: '🇫🇷' },
  { code: 'de-DE', label: 'Deutsch', flag: '🇩🇪' },
  { code: 'it-IT', label: 'Italiano', flag: '🇮🇹' },
  { code: 'pt-BR', label: 'Português', flag: '🇧🇷' },
  { code: 'ru-RU', label: 'Русский', flag: '🇷🇺' },
  { code: 'tr-TR', label: 'Türkçe', flag: '🇹🇷' },
  { code: 'nl-NL', label: 'Nederlands', flag: '🇳🇱' },
  { code: 'pl-PL', label: 'Polski', flag: '🇵🇱' },
  { code: 'vi-VN', label: 'Tiếng Việt', flag: '🇻🇳' },
  { code: 'th-TH', label: 'ภาษาไทย', flag: '🇹🇭' },
  { code: 'id-ID', label: 'Bahasa Indonesia', flag: '🇮🇩' },
  { code: 'ms-MY', label: 'Bahasa Melayu', flag: '🇲🇾' },
  { code: 'sw-KE', label: 'Kiswahili', flag: '🇰🇪' },
];
