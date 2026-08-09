import { Search, Sparkles } from "lucide-react";
import type { Language, QueryListItem, SearchMode } from "../types/api";
import { useCopy } from "../i18n/translations";

interface SearchHeroProps {
  language: Language;
  mode: SearchMode;
  query: string;
  onQueryChange: (query: string) => void;
  onSearch: () => void;
  benchmarkQueries: QueryListItem[];
  selectedQid: string | null;
  onSelectBenchmark: (item: QueryListItem) => void;
  loading: boolean;
  compact?: boolean;
}

export function SearchHero(props: SearchHeroProps) {
  const { language, mode, query, onQueryChange, onSearch, benchmarkQueries, selectedQid, onSelectBenchmark, loading, compact = false } = props;
  const copy = useCopy(language);
  const suggestions = mode === "benchmark" && query.trim() && !selectedQid
    ? benchmarkQueries.filter((item) => item.question.toLowerCase().includes(query.toLowerCase())).slice(0, 6)
    : [];
  const examples = [...benchmarkQueries]
    .sort((a, b) => Number(b.day5_citation_triggered) - Number(a.day5_citation_triggered) || Number(b.day4_rescue_triggered) - Number(a.day4_rescue_triggered))
    .slice(0, 4);

  return <section className={`search-section ${compact ? "compact" : "landing"}`}>
    <div className="search-heading">
      <Sparkles size={compact ? 18 : 22} />
      <div>
        <h1>{copy.ask}</h1>
        {!compact && <p>{copy.askSubline}</p>}
      </div>
    </div>
    <div className="query-box-wrap">
      <input
        className="query-input"
        value={query}
        onChange={(event) => onQueryChange(event.target.value)}
        onKeyDown={(event) => { if (event.key === "Enter") onSearch(); }}
        placeholder={mode === "benchmark" ? copy.placeholderBenchmark : copy.placeholderLive}
      />
      <button className="search-button" onClick={onSearch} disabled={loading || query.trim().length < 3} aria-label={copy.search}><Search size={compact ? 25 : 29} /></button>
      {suggestions.length > 0 && <div className="query-suggestions">
        {suggestions.map((item) => <button key={item.qid} onClick={() => onSelectBenchmark(item)}><strong>{item.qid}</strong><span>{item.question}</span></button>)}
      </div>}
    </div>
    {mode === "benchmark" && examples.length > 0 && <div className="example-row">
      <span>{copy.examples}</span>
      <div className="example-chips">{examples.map((item) => <button key={item.qid} className={selectedQid === item.qid ? "selected" : ""} onClick={() => onSelectBenchmark(item)} title={item.question}>{truncate(item.question, compact ? 42 : 54)}</button>)}</div>
    </div>}
    {mode === "live" && !compact && <div className="live-search-hint">{copy.liveInputHint}</div>}
  </section>;
}

function truncate(value: string, length: number) {
  return value.length <= length ? value : `${value.slice(0, length - 1)}…`;
}
