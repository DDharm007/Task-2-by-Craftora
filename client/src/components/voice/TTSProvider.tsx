/**
 * One shared text-to-speech engine for the app.
 *
 * `useVoiceTTS` owns an `Audio` element and playback state, so calling it in two
 * components produces two independent players: the answer panel's stop button
 * would not stop audio the console started, and its icon would not reflect that
 * anything was playing. Everything reads the same instance through this context.
 */
import { createContext, useContext, type ReactNode } from 'react';
import { useVoiceTTS, type VoiceTTSState } from '@/hooks/useVoiceTTS';

const TTSContext = createContext<VoiceTTSState | null>(null);

export function TTSProvider({ children }: { children: ReactNode }) {
  const tts = useVoiceTTS();
  return <TTSContext.Provider value={tts}>{children}</TTSContext.Provider>;
}

/** The shared engine. Throws if used outside the provider. */
export function useTTS(): VoiceTTSState {
  const tts = useContext(TTSContext);
  if (!tts) throw new Error('useTTS must be used within <TTSProvider>');
  return tts;
}
