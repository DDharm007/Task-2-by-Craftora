/**
 * Theme selection: Light, Dark, and "HHGoa'26 Exclusive".
 *
 * The whole mechanism is one HTML attribute — `data-theme` on `<html>` — read
 * by the CSS custom properties in styles/index.css. React only owns the
 * selection and its persistence; the actual repaint is CSS doing what CSS is
 * already good at, so switching themes touches zero component render trees.
 *
 * A blocking inline script in index.html sets `data-theme` before first paint
 * (see the head there), so a returning visitor who picked Dark or Exclusive
 * never sees a flash of Light first. This hook picks up from whatever that
 * script already applied rather than re-deciding on mount.
 */
import { createContext, useCallback, useContext, useEffect, useState, type ReactNode } from 'react';

export type Theme = 'light' | 'dark' | 'exclusive';

export const THEME_STORAGE_KEY = 'goarag:theme';

const THEMES: readonly Theme[] = ['light', 'dark', 'exclusive'];

/** Mirrors the `colors` map in index.html's pre-paint bootstrap script. */
const THEME_COLOR: Record<Theme, string> = {
  light: '#FFFFFF',
  dark: '#0B0D10',
  exclusive: '#036735',
};

function isTheme(value: string | null): value is Theme {
  return value !== null && (THEMES as readonly string[]).includes(value);
}

/** What the bootstrap script in index.html already applied, if anything. */
function initialTheme(): Theme {
  if (typeof document === 'undefined') return 'light';
  const attr = document.documentElement.getAttribute('data-theme');
  if (isTheme(attr)) return attr;
  const stored = window.localStorage.getItem(THEME_STORAGE_KEY);
  return isTheme(stored) ? stored : 'light';
}

interface ThemeContextValue {
  theme: Theme;
  setTheme: (theme: Theme) => void;
}

const ThemeContext = createContext<ThemeContextValue | null>(null);

export function ThemeProvider({ children }: { children: ReactNode }) {
  const [theme, setThemeState] = useState<Theme>(initialTheme);

  useEffect(() => {
    // Light has no attribute at all — it's the bare `:root` in the
    // stylesheet, so removing the attribute (rather than writing
    // `data-theme="light"`) keeps that the true zero-config default.
    if (theme === 'light') {
      document.documentElement.removeAttribute('data-theme');
    } else {
      document.documentElement.setAttribute('data-theme', theme);
    }
    window.localStorage.setItem(THEME_STORAGE_KEY, theme);

    document.querySelector('#theme-color-meta')?.setAttribute('content', THEME_COLOR[theme]);
  }, [theme]);

  const setTheme = useCallback((next: Theme) => setThemeState(next), []);

  return <ThemeContext.Provider value={{ theme, setTheme }}>{children}</ThemeContext.Provider>;
}

export function useTheme(): ThemeContextValue {
  const ctx = useContext(ThemeContext);
  if (!ctx) throw new Error('useTheme must be used within <ThemeProvider>');
  return ctx;
}
