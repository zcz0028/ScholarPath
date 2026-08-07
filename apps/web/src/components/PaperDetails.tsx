import { Anchor, ArrowRight, ExternalLink, FileSearch2, GitBranch, SearchCheck } from "lucide-react";
import { useState } from "react";
import type { Language, PaperResult } from "../types/api";
import { useCopy } from "../i18n/translations";

type DetailTab = "why" | "evidence" | "citation" | "sources";

export function PaperDetails({ language, paper }: { language: Language; paper: PaperResult | null }) {
  const copy = useCopy(language);
  const [tab, setTab] = useState<DetailTab>("why");
  if (!paper) return <EmptyPaperDetails language={language} />;

  return <aside className="details-card">
    <div className="details-head">
      <div><span className="eyebrow">{copy.paperDetails}</span><h2>{paper.title}</h2><p>{paper.authors.slice(0, 6).join(", ")}{paper.authors.length > 6 ? ", et al." : ""}</p><p className="paper-meta">{[paper.year, paper.venue].filter(Boolean).join(" · ")}</p></div>
      <div className="detail-score"><strong>{paper.score == null ? "—" : paper.score.toFixed(3)}</strong><span>{copy.score}</span></div>
    </div>
    <div className="paper-tags detail-tags">{paper.reason_tags.slice(0, 6).map((tag) => <span key={tag}>{tag.replace(/_/g, " ")}</span>)}</div>
    <div className="detail-tabs">
      <TabButton active={tab === "why"} onClick={() => setTab("why")}>{copy.whyRecommended}</TabButton>
      <TabButton active={tab === "evidence"} onClick={() => setTab("evidence")}>{copy.matchEvidence}</TabButton>
      <TabButton active={tab === "citation"} onClick={() => setTab("citation")}>{copy.citationPath}</TabButton>
      <TabButton active={tab === "sources"} onClick={() => setTab("sources")}>{copy.retrievalSources}</TabButton>
    </div>
    <div className="detail-tab-body">
      {tab === "why" && <WhyTab paper={paper} fallback={copy.reasonUnavailable} />}
      {tab === "evidence" && <EvidenceTab paper={paper} fallback={copy.evidenceUnavailable} />}
      {tab === "citation" && <CitationTab paper={paper} fallback={copy.citationUnavailable} />}
      {tab === "sources" && <SourcesTab paper={paper} fallback={copy.sourceUnavailable} />}
    </div>
    <div className="key-info"><h3>{copy.keyInfo}</h3><div className="key-grid"><Info label={copy.openalexId} value={paper.openalex_id} /><Info label={copy.doi} value={paper.doi} /><Info label={copy.arxivId} value={paper.arxiv_id} /><Info label={copy.year} value={paper.year ? String(paper.year) : null} /><Info wide label={copy.venue} value={paper.venue} /></div></div>
    <div className="detail-actions">
      <a className={`action-button secondary ${!paper.url ? "disabled" : ""}`} href={paper.url || undefined} target="_blank" rel="noreferrer"><ExternalLink size={17} />{copy.viewPaper}</a>
      <button className="action-button primary" disabled={!paper.citation_path} onClick={() => setTab("citation")}><GitBranch size={17} />{copy.viewCitation}</button>
    </div>
  </aside>;
}

function EmptyPaperDetails({ language }: { language: Language }) {
  const copy = useCopy(language);
  return <aside className="details-card empty-details"><span className="eyebrow">{copy.paperDetails}</span><div className="empty-illustration"><FileSearch2 size={74} /></div><h2>{copy.selectPaper}</h2><p>{copy.selectPaperHelp}</p><div className="empty-menu"><EmptyMenuItem icon={<SearchCheck size={19} />} title={copy.whyRecommended} /><EmptyMenuItem icon={<Anchor size={19} />} title={copy.matchEvidence} /><EmptyMenuItem icon={<GitBranch size={19} />} title={copy.citationPath} /><EmptyMenuItem icon={<FileSearch2 size={19} />} title={copy.retrievalSources} /></div></aside>;
}
function EmptyMenuItem({ icon, title }: { icon: React.ReactNode; title: string }) { return <div>{icon}<span>{title}</span><ArrowRight size={17} /></div>; }
function TabButton({ active, onClick, children }: { active: boolean; onClick: () => void; children: React.ReactNode }) { return <button className={active ? "active" : ""} onClick={onClick}>{children}</button>; }
function WhyTab({ paper, fallback }: { paper: PaperResult; fallback: string }) { return <div className="content-panel"><h3>{paper.reason_text ? "Reason" : ""}</h3><p>{paper.reason_text || fallback}</p>{paper.reason_tags.length > 0 && <div className="evidence-bullets">{paper.reason_tags.map((tag) => <div key={tag}><SearchCheck size={15} />{tag.replace(/_/g, " ")}</div>)}</div>}</div>; }
function EvidenceTab({ paper, fallback }: { paper: PaperResult; fallback: string }) { return <div className="content-panel">{paper.constraint_evidence.length ? <pre>{JSON.stringify(paper.constraint_evidence, null, 2)}</pre> : <p>{fallback}</p>}<div className="single-score"><span>Score</span><strong>{paper.score == null ? "—" : paper.score.toFixed(6)}</strong></div></div>; }
function CitationTab({ paper, fallback }: { paper: PaperResult; fallback: string }) { const path = paper.citation_path; if (!path) return <div className="content-panel"><p>{fallback}</p></div>; return <div className="content-panel citation-flow"><div><small>Seed</small><strong>{path.seed_title || path.seed_openalex_id || "—"}</strong></div><ArrowRight /><div><small>{path.edge_type || "citation"} · hop {path.hop ?? "—"}</small><strong>{path.expanded_title || paper.title}</strong></div></div>; }
function SourcesTab({ paper, fallback }: { paper: PaperResult; fallback: string }) { return <div className="content-panel source-list">{paper.retrieval_sources.length ? paper.retrieval_sources.map((source) => <div key={source}><SearchCheck size={16} /><span>{source.replace(/_/g, " ")}</span></div>) : <p>{fallback}</p>}</div>; }
function Info({ label, value, wide }: { label: string; value?: string | null; wide?: boolean }) { return <div className={wide ? "wide" : ""}><small>{label}</small><strong>{value || "—"}</strong></div>; }
