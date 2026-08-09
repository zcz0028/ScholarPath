import { ExternalLink } from "lucide-react";
import type { Language, PaperResult } from "../types/api";
import { useCopy } from "../i18n/translations";

interface PaperListProps {
  language: Language;
  papers: PaperResult[];
  selectedPaper: PaperResult | null;
  onSelectPaper: (paper: PaperResult) => void;
}

export function PaperList({ language, papers, selectedPaper, onSelectPaper }: PaperListProps) {
  const copy = useCopy(language);
  return <section className="paper-list-card">
    <div className="list-toolbar"><span><strong>{papers.length}</strong> {copy.results}</span><span>{copy.sort}: <strong>{copy.relevance}</strong></span></div>
    <div className="paper-list">
      {papers.length === 0 && <div className="paper-empty">{copy.noResults}</div>}
      {papers.map((paper) => <article key={`${paper.rank}-${paper.openalex_id || paper.doi || paper.title}`} className={`paper-row ${selectedPaper?.rank === paper.rank ? "selected" : ""}`} onClick={() => onSelectPaper(paper)}>
        <div className="rank-badge">{paper.rank}</div>
        <div className="paper-main">
          <h2>{paper.title}</h2>
          <p className="paper-authors">{paper.authors.length ? paper.authors.slice(0, 6).join(", ") + (paper.authors.length > 6 ? ", et al." : "") : "—"}</p>
          <p className="paper-meta">{[paper.year, paper.venue].filter(Boolean).join(" · ")}</p>
          <div className="paper-tags">{paper.reason_tags.slice(0, 5).map((tag) => <span key={tag}>{formatTag(tag, language)}</span>)}{paper.openalex_id && <span>OpenAlex</span>}{paper.arxiv_id && <span>arXiv</span>}{paper.doi && <span>DOI</span>}</div>
        </div>
        <div className="paper-side">
          <div className="score-box"><strong>{paper.score == null ? "—" : paper.score.toFixed(3)}</strong><span>{copy.score}</span></div>
          {paper.url && <a href={paper.url} target="_blank" rel="noreferrer" onClick={(event) => event.stopPropagation()} aria-label={copy.viewPaper}><ExternalLink size={19} /></a>}
        </div>
      </article>)}
    </div>
  </section>;
}

function formatTag(tag: string, language: Language) {
  const labels: Record<string, [string, string]> = {
    anchor_rescue_source: ["救援召回", "Rescue Retrieval"],
    multi_plan_supported: ["多计划支持", "Multi-plan Support"],
    high_rank_rescue_hit: ["高位救援命中", "High-rank Rescue"],
    strong_identifier_available: ["强标识符", "Strong ID"],
    multi_source_supported: ["多来源支持", "Multi-source Support"],
    high_rank_in_at_least_one_source: ["来源高排名", "High Rank in Source"],
  };
  const mapped = labels[tag];
  return mapped ? mapped[language === "zh" ? 0 : 1] : tag.replace(/_/g, " ");
}
