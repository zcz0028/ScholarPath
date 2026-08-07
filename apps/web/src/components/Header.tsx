import { CircleHelp, Clock3, Coins, DatabaseZap, Globe2 } from "lucide-react";
import type { Language, SearchMode, SearchResponse } from "../types/api";
import { useCopy } from "../i18n/translations";

interface HeaderProps {
  language: Language;
  onLanguageChange: (language: Language) => void;
  mode: SearchMode;
  onModeChange: (mode: SearchMode) => void;
  response: SearchResponse | null;
}

function formatLatency(ms?: number) {
  if (ms == null) return "—";
  return ms >= 1000 ? `${(ms / 1000).toFixed(2)}s` : `${ms}ms`;
}

export function Header({ language, onLanguageChange, mode, onModeChange, response }: HeaderProps) {
  const copy = useCopy(language);
  const cost = response?.cost;
  return (
    <header className="topbar">
      <div className="brand-wrap">
        <div className="brand-mark" aria-hidden="true"><span /></div>
        <div>
          <div className="brand-name">ScholarPath</div>
          <div className="brand-subtitle">{copy.subtitle}</div>
        </div>
      </div>

      <div className="mode-switch" aria-label="Search mode">
        <button className={mode === "benchmark" ? "active" : ""} onClick={() => onModeChange("benchmark")} title={copy.benchmarkHint}>{copy.benchmark}</button>
        <button className={mode === "live" ? "active" : ""} onClick={() => onModeChange("live")} title={copy.liveHint}>{copy.live}</button>
      </div>

      <div className="top-actions">
        <div className="language-switch"><Globe2 size={14} /><button className={language === "zh" ? "active" : ""} onClick={() => onLanguageChange("zh")}>中</button><button className={language === "en" ? "active" : ""} onClick={() => onLanguageChange("en")}>EN</button></div>
        <Metric label={copy.apiCalls} value={cost ? String(cost.api_calls) : "—"} icon={<DatabaseZap size={15} />} />
        <Metric label={copy.cacheHits} value={cost ? String(cost.cache_hits) : "—"} icon={<DatabaseZap size={15} />} />
        <Metric label={copy.latency} value={formatLatency(response?.latency_ms)} icon={<Clock3 size={15} />} />
        <Metric label={copy.estCost} value={cost ? `$${cost.estimated_cost_usd.toFixed(4)}` : "—"} icon={<Coins size={15} />} />
        <CircleHelp size={19} className="help-icon" />
      </div>
    </header>
  );
}

function Metric({ label, value, icon }: { label: string; value: string; icon: React.ReactNode }) {
  return <div className="top-metric"><span className="metric-label">{label}</span><span className="metric-value">{icon}{value}</span></div>;
}
