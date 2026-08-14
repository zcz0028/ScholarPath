import { AlertCircle, LoaderCircle } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { getBenchmarkQueries, searchScholarPath } from "./api/scholarpath";
import { Header } from "./components/Header";
import { PaperDetails } from "./components/PaperDetails";
import { PaperList } from "./components/PaperList";
import { PipelineBar } from "./components/PipelineBar";
import { SearchHero } from "./components/SearchHero";
import { SearchReasoning } from "./components/SearchReasoning";
import { useCopy } from "./i18n/translations";
import type { Language, PaperResult, QueryListItem, SearchMode, SearchResponse } from "./types/api";

export default function App() {
  const [language, setLanguage] = useState<Language>("zh");
  const [mode, setMode] = useState<SearchMode>("benchmark");
  const [query, setQuery] = useState("");
  const [selectedQid, setSelectedQid] = useState<string | null>(null);
  const [benchmarkQueries, setBenchmarkQueries] = useState<QueryListItem[]>([]);
  const [response, setResponse] = useState<SearchResponse | null>(null);
  const [selectedPaper, setSelectedPaper] = useState<PaperResult | null>(null);
  const [reasoningExpanded, setReasoningExpanded] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const copy = useCopy(language);
  const hasRun = Boolean(response || loading || error);

  useEffect(() => {
    getBenchmarkQueries().then((result) => setBenchmarkQueries(result.items)).catch((err: Error) => setError(err.message));
  }, []);

  useEffect(() => setSelectedPaper(null), [response?.run_id]);

  const selectedBenchmark = useMemo(
    () => benchmarkQueries.find((item) => item.qid === selectedQid) || null,
    [benchmarkQueries, selectedQid],
  );

  function changeMode(nextMode: SearchMode) {
    setMode(nextMode);
    setResponse(null);
    setSelectedPaper(null);
    setError(null);
    setReasoningExpanded(false);
    setSelectedQid(null);
    setQuery("");
  }

  function updateQuery(next: string) {
    setQuery(next);
    if (mode === "benchmark" && selectedBenchmark?.question !== next) setSelectedQid(null);
  }

  function selectBenchmark(item: QueryListItem) {
    setSelectedQid(item.qid);
    setQuery(item.question);
    setError(null);
  }

  async function runSearch() {
    const clean = query.trim();
    if (clean.length < 3) return;
    let qid = selectedQid;
    if (mode === "benchmark" && !qid) {
      const exact = benchmarkQueries.find((item) => item.question.trim() === clean);
      qid = exact?.qid || null;
      if (!qid) { setError(copy.benchmarkSelectError); return; }
    }
    setLoading(true); setError(null); setSelectedPaper(null);
    try {
      const result = await searchScholarPath({
        query: clean,
        qid: mode === "benchmark" ? qid : null,
        mode,
        top_k: 20,
        enable_citation: mode === "benchmark",
      });
      setResponse(result);
      setReasoningExpanded(false);
    } catch (err) {
      setResponse(null);
      setError(err instanceof Error ? err.message : copy.error);
    } finally {
      setLoading(false);
    }
  }

  return <div className={`app-shell ${hasRun ? "workspace-state" : "landing-state"}`}>
    <Header language={language} onLanguageChange={setLanguage} mode={mode} onModeChange={changeMode} response={response} />
    <main className="page-content">
      <SearchHero
        language={language}
        mode={mode}
        query={query}
        onQueryChange={updateQuery}
        onSearch={runSearch}
        benchmarkQueries={benchmarkQueries}
        selectedQid={selectedQid}
        onSelectBenchmark={selectBenchmark}
        loading={loading}
        compact={hasRun}
      />

      {!hasRun && <LandingModeNote language={language} mode={mode} />}

      {error && <div className="status-message error"><AlertCircle size={18} /><div><strong>{copy.error}</strong><span>{error}</span></div></div>}
      {loading && <div className="status-message loading"><LoaderCircle size={19} className="spin" /><span>{copy.loading}</span></div>}

      {response && <>
        <SearchReasoning language={language} response={response} expanded={reasoningExpanded} onToggle={() => setReasoningExpanded((value) => !value)} />
        <div className={`workspace-grid ${selectedPaper ? "paper-selected" : "paper-unselected"}`}>
          <div className="result-column">
            <PaperList language={language} papers={response.results} selectedPaper={selectedPaper} onSelectPaper={setSelectedPaper} />
          </div>
          <PaperDetails language={language} paper={selectedPaper} />
        </div>
        {response.warnings.length > 0 && <div className="warning-box"><AlertCircle size={17} /><div><strong>{copy.warning}</strong>{response.warnings.map((warning, index) => <p key={index}>{warning}</p>)}</div></div>}
      </>}
    </main>
    {hasRun && <PipelineBar language={language} response={response} />}
  </div>;
}

function LandingModeNote({ language, mode }: { language: Language; mode: SearchMode }) {
  const copy = useCopy(language);
  return <div className="landing-mode-note">
    <span className={`mode-dot ${mode}`} />
    <strong>{mode === "benchmark" ? copy.benchmark : copy.live}</strong>
    <span>{mode === "benchmark" ? copy.benchmarkLandingNote : copy.liveLandingNote}</span>
  </div>;
}
