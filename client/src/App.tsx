import { Navigate, Route, Routes } from 'react-router-dom';
import { TooltipProvider } from '@/components/ui/primitives';
import { TTSProvider } from '@/components/voice/TTSProvider';
import { ThemeProvider } from '@/hooks/useTheme';
import { AppShell } from '@/components/layout/AppShell';
import { ConsolePage } from '@/pages/Console';
import { AnalyticsPage } from '@/pages/Analytics';
import { DashboardPage } from '@/pages/Dashboard';

export default function App() {
  return (
    <ThemeProvider>
      <TooltipProvider delayDuration={200} skipDelayDuration={300}>
        <TTSProvider>
          <AppShell>
            <Routes>
              <Route path="/" element={<ConsolePage />} />
              <Route path="/analytics" element={<AnalyticsPage />} />
              <Route path="/dashboard" element={<DashboardPage />} />
              <Route path="*" element={<Navigate to="/" replace />} />
            </Routes>
          </AppShell>
        </TTSProvider>
      </TooltipProvider>
    </ThemeProvider>
  );
}
