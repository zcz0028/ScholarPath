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
}

export function SearchHero(props: SearchHeroProps) {
  const { language, mode, query, onQueryChange, onSearch, benchmarkQueries, selectedQid, onSelectBenchmark, loading } = props;
  const copy = useCopy(language);
  const suggestions = mode === "benchmark" && query.trim() && !selectedQid
    ? benchmarkQueries.filter((item) => item.question.toLowerCase().includes(query.toLowerCase())).slice(0, 6)
    : [];
  const examples = [...benchmarkQueries]
    .sort((a, b) => Number(b.day5_citation_triggered) - Number(a.day5_citation_triggered) || Number(b.day4_rescue_triggered) - Number(a.day4_rescue_triggered))
    .slice(0, 4);

  return <section className="search-section">
    <div className="search-heading"><Sparkles size={20} /><h1>{copy.ask}</h1></div>
    <div className="query-box-wrap">
      <input
        className="query-input"
        value={query}
        onChange={(event) => onQueryChange(event.target.value)}
        onKeyDown={(event) => { if (event.key === "Enter") onSearch(); }}
        placeholder={mode === "benchmark" ? copy.placeholderBenchmark : copy.placeholderLive}
      />
      <button className="search-button" onClick={onSearch} disabled={loading || query.trim().length < 3} aria-label={copy.search}><Search size={27} /></button>
      {suggestions.length > 0 && <div className="query-suggestions">
        {suggestions.map((item) => <button key={item.qid} onClick={() => onSelectBenchmark(item)}><strong>{item.qid}</strong><span>{item.question}</span></button>)}
      </div>}
    </div>
    {mode === "benchmark" && examples.length > 0 && <div className="example-row"><span>{copy.examples}</span><div className="example-chips">{examples.map((item) => <button key={item.qid} className={selectedQid === item.qid ? "selected" : ""} onClick={() => onSelectBenchmark(item)} title={item.question}>{truncate(item.question, 48)}</button>)}</div></div>}
  </section>;
}

function truncate(value: string, length: number) {
  return value.length <= length ? value : `${value.slice(0, length - 1)}…`;
}
