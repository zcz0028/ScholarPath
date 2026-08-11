import {
  Anchor, ArrowRight, CheckCircle2, Circle, ExternalLink, FileSearch2, GitBranch, SearchCheck,
} from "lucide-react";
import { useState } from "react";
import type { ConstraintEvidenceItem, Language, PaperResult } from "../types/api";
import { useCopy } from "../i18n/translations";
import "./PaperDetails.day8.css";

type DetailTab = "why" | "evidence" | "citation" | "sources";

export function PaperDetails({ language, paper }: { language: Language; paper: PaperResult | null }) {
  const copy = useCopy(language);
  const [tab, setTab] = useState<DetailTab>("why");
  if (!paper) return <EmptyPaperDetails language={language} />;

  return <aside className="details-card">
    <div className="details-head">
      <div>
        <span className="eyebrow">{copy.paperDetails}</span><h2>{paper.title}</h2>
        <p>{paper.authors.slice(0, 6).join(", ")}{paper.authors.length > 6 ? ", et al." : ""}</p>
        <p className="paper-meta">{[paper.year, paper.venue].filter(Boolean).join(" · ")}</p>
      </div>
      <div className="detail-score"><strong>{paper.score == null ? "—" : paper.score.toFixed(3)}</strong><span>{copy.score}</span></div>
    </div>

    <div className="paper-tags detail-tags">{paper.reason_tags.slice(0, 6).map((tag) => <span key={tag}>{formatEvidenceLabel(tag, language)}</span>)}</div>

    <div className="detail-tabs">
      <TabButton active={tab === "why"} onClick={() => setTab("why")}>{copy.whyRecommended}</TabButton>
      <TabButton active={tab === "evidence"} onClick={() => setTab("evidence")}>{copy.matchEvidence}</TabButton>
      <TabButton active={tab === "citation"} onClick={() => setTab("citation")}>{copy.citationPath}</TabButton>
      <TabButton active={tab === "sources"} onClick={() => setTab("sources")}>{copy.retrievalSources}</TabButton>
    </div>

    <div className="detail-tab-body">
      {tab === "why" && <WhyTab language={language} paper={paper} fallback={copy.reasonUnavailable} />}
      {tab === "evidence" && <EvidenceTab language={language} paper={paper} fallback={copy.evidenceUnavailable} />}
      {tab === "citation" && <CitationTab language={language} paper={paper} fallback={copy.citationUnavailable} />}
      {tab === "sources" && <SourcesTab language={language} paper={paper} fallback={copy.sourceUnavailable} />}
    </div>

    <div className="key-info"><h3>{copy.keyInfo}</h3><div className="key-grid">
      <Info label={copy.openalexId} value={paper.openalex_id} /><Info label={copy.doi} value={paper.doi} />
      <Info label={copy.arxivId} value={paper.arxiv_id} /><Info label={copy.year} value={paper.year ? String(paper.year) : null} />
      <Info wide label={copy.venue} value={paper.venue} />
    </div></div>

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

function WhyTab({ language, paper, fallback }: { language: Language; paper: PaperResult; fallback: string }) {
  const hasReason = Boolean(paper.reason_text);
  const hasMatchSummary = paper.constraint_count > 0;
  return <div className="content-panel day8-why-panel">
    {hasReason && <><h3>{language === "zh" ? "推荐依据" : "Recommendation basis"}</h3><p className="day8-reason-text">{paper.reason_text}</p></>}
    {hasMatchSummary && <div className="day8-match-summary">
      <div className="day8-match-summary-head"><span>{language === "zh" ? "约束匹配" : "Constraint match"}</span><strong>{paper.matched_count} / {paper.constraint_count}</strong></div>
      {paper.matched_constraints.length > 0 && <div className="day8-constraint-group"><small>{language === "zh" ? "已匹配" : "Matched"}</small><div className="day8-chip-row">{paper.matched_constraints.map((v) => <span className="day8-chip matched" key={v}><CheckCircle2 size={13}/>{v}</span>)}</div></div>}
      {paper.unmatched_constraints.length > 0 && <div className="day8-constraint-group"><small>{language === "zh" ? "尚未匹配" : "Not matched"}</small><div className="day8-chip-row">{paper.unmatched_constraints.map((v) => <span className="day8-chip unmatched" key={v}><Circle size={13}/>{v}</span>)}</div></div>}
    </div>}
    {!hasReason && !hasMatchSummary && <p>{fallback}</p>}
    {paper.reason_tags.length > 0 && <div className="evidence-bullets">{paper.reason_tags.map((tag) => <div key={tag}><SearchCheck size={15}/>{formatEvidenceLabel(tag, language)}</div>)}</div>}
  </div>;
}

function EvidenceTab({ language, paper, fallback }: { language: Language; paper: PaperResult; fallback: string }) {
  const evidence = paper.constraint_evidence || [];
  if (!evidence.length) return <div className="content-panel"><p>{fallback}</p></div>;
  return <div className="content-panel day8-evidence-panel">
    <div className="day8-evidence-head"><div><h3>{language === "zh" ? "约束证据" : "Constraint evidence"}</h3><p>{language === "zh" ? "证据由后端结构化匹配器生成，不调用大模型。" : "Evidence is generated by the deterministic backend matcher without an LLM."}</p></div><span className="day8-count-badge">{paper.matched_count} / {paper.constraint_count} {language === "zh" ? "已匹配" : "matched"}</span></div>
    <div className="day8-evidence-list">{evidence.map((item) => <EvidenceItem key={item.constraint_id} item={item} language={language}/>)}</div>
  </div>;
}

function EvidenceItem({ item, language }: { item: ConstraintEvidenceItem; language: Language }) {
  return <div className={`day8-evidence-item ${item.matched ? "matched" : "unmatched"}`}>
    <div className="day8-evidence-icon">{item.matched ? <CheckCircle2 size={18}/> : <Circle size={18}/>}</div>
    <div className="day8-evidence-main">
      <div className="day8-evidence-title-row"><strong>{item.constraint_text}</strong><span>{formatConstraintType(item.constraint_type, language)}</span></div>
      {item.matched ? <><div className="day8-evidence-meta"><span>{formatEvidenceField(item.evidence_field, language)}</span><span>·</span><span>{formatMatchType(item.match_type, language)}</span></div>
        {item.evidence_text && <blockquote className="day8-evidence-quote">{item.evidence_text}</blockquote>}
        <div className="day8-evidence-metrics"><span>{language === "zh" ? "证据置信度" : "Evidence confidence"} <strong>{formatMetric(item.confidence)}</strong></span><span>{language === "zh" ? "词元覆盖" : "Token coverage"} <strong>{formatPercent(item.token_coverage)}</strong></span></div>
      </> : <p className="day8-unmatched-note">{language === "zh" ? "当前论文元数据中暂无可验证的直接匹配证据。" : "No verifiable direct evidence was found in the current paper metadata."}</p>}
    </div>
  </div>;
}

function CitationTab({ language, paper, fallback }: { language: Language; paper: PaperResult; fallback: string }) {
  const path = paper.citation_path;
  if (!path) return <div className="content-panel day8-citation-empty"><GitBranch size={26}/><div><h3>{language === "zh" ? "暂无可验证引用路径" : "No verified citation path"}</h3><p>{fallback}</p></div></div>;
  return <div className="content-panel citation-flow"><div><small>{language === "zh" ? "种子论文" : "Seed"}</small><strong>{path.seed_title || path.seed_openalex_id || "—"}</strong></div><ArrowRight/><div><small>{formatCitationEdge(path.edge_type, language)} · {language === "zh" ? "跳数" : "hop"} {path.hop ?? "—"}</small><strong>{path.expanded_title || paper.title}</strong></div></div>;
}

function SourcesTab({ language, paper, fallback }: { language: Language; paper: PaperResult; fallback: string }) {
  return <div className="content-panel source-list day8-source-list">{paper.retrieval_sources.length ? paper.retrieval_sources.map((source) => <div key={source}><SearchCheck size={16}/><span>{formatSourceLabel(source, language)}</span></div>) : <p>{fallback}</p>}</div>;
}

function formatEvidenceLabel(value: string, language: Language) {
  const labels: Record<string, [string, string]> = {
    model_or_entity_match:["模型 / 实体匹配","Model / Entity Match"], task_or_modality_match:["任务 / 模态匹配","Task / Modality Match"],
    method_or_property_match:["方法 / 属性匹配","Method / Property Match"], topic_match:["主题匹配","Topic Match"],
    title_evidence:["标题证据","Title Evidence"], abstract_evidence:["摘要证据","Abstract Evidence"], concept_evidence:["概念证据","Concept Evidence"],
    anchor_rescue_source:["救援召回","Rescue Retrieval"], multi_plan_supported:["多计划支持","Multi-plan Support"],
    high_rank_rescue_hit:["高位救援命中","High-rank Rescue"], strong_identifier_available:["强标识符可用","Strong Identifier"],
    multi_source_supported:["多来源支持","Multi-source Support"], high_rank_in_at_least_one_source:["至少一个来源中高排名","High Rank in Source"],
  };
  const mapped = labels[value]; return mapped ? mapped[language==="zh"?0:1] : value.replace(/_/g," ");
}
function formatSourceLabel(value: string, language: Language) {
  const labels: Record<string,[string,string]> = {b4_base:["B4 基础召回","B4 Base Retrieval"],day4_rescue:["Day4 定向救援召回","Day4 Rescue Retrieval"],openalex:["OpenAlex","OpenAlex"],openalex_live:["OpenAlex 实时检索","OpenAlex Live Retrieval"]};
  const mapped=labels[value]; return mapped ? mapped[language==="zh"?0:1] : value.replace(/_/g," ");
}
function formatConstraintType(value:string,language:Language){const labels:Record<string,[string,string]>={model_or_entity:["模型 / 实体","Model / Entity"],task_or_modality:["任务 / 模态","Task / Modality"],method_or_property:["方法 / 属性","Method / Property"],topic:["主题","Topic"]};const m=labels[value];return m?m[language==="zh"?0:1]:value.replace(/_/g," ");}
function formatEvidenceField(value:ConstraintEvidenceItem["evidence_field"],language:Language){const labels:Record<string,[string,string]>={title:["论文标题","Paper title"],abstract:["论文摘要","Paper abstract"],concept:["OpenAlex 概念","OpenAlex concept"]};if(!value)return language==="zh"?"无":"None";const m=labels[value];return m?m[language==="zh"?0:1]:value;}
function formatMatchType(value:string,language:Language){const labels:Record<string,[string,string]>={title_phrase_match:["标题精确短语匹配","Title phrase match"],title_alias_match:["标题别名匹配","Title alias match"],title_token_match:["标题词元覆盖匹配","Title token match"],abstract_phrase_match:["摘要精确短语匹配","Abstract phrase match"],abstract_alias_match:["摘要别名匹配","Abstract alias match"],abstract_token_match:["摘要词元覆盖匹配","Abstract token match"],concept_phrase_match:["概念精确短语匹配","Concept phrase match"],concept_alias_match:["概念别名匹配","Concept alias match"],concept_token_match:["概念词元覆盖匹配","Concept token match"],none:["未匹配","Not matched"]};const m=labels[value];return m?m[language==="zh"?0:1]:value.replace(/_/g," ");}
function formatCitationEdge(value:string|null|undefined,language:Language){if(!value)return language==="zh"?"引用关系":"Citation";const labels:Record<string,[string,string]>={references:["参考文献","References"],cited_by:["被引用","Cited by"],citation:["引用关系","Citation"]};const m=labels[value];return m?m[language==="zh"?0:1]:value.replace(/_/g," ");}
function formatMetric(value:number){return Number.isFinite(value)?value.toFixed(2):"—";}
function formatPercent(value:number){return Number.isFinite(value)?`${Math.round(value*100)}%`:"—";}
function Info({label,value,wide}:{label:string;value?:string|null;wide?:boolean}){return <div className={wide?"wide":""}><small>{label}</small><strong>{value||"—"}</strong></div>;}
