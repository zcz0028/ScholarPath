import { Anchor, ChevronDown, CircleCheck, Network, SearchCheck, Sparkles } from "lucide-react";
import type { Language, SearchResponse } from "../types/api";
import { useCopy } from "../i18n/translations";

interface SearchReasoningProps {
  language: Language;
  response: SearchResponse;
  expanded: boolean;
  onToggle: () => void;
}

export function SearchReasoning({ language, response, expanded, onToggle }: SearchReasoningProps) {
  const copy = useCopy(language);
  const reasoning = response.reasoning ?? null;

  const constraints = reasoning?.constraints ?? response.parsed_constraints;
  const candidateSubqueries = reasoning?.candidate_subqueries ?? [];
  const plans = reasoning?.selected_plans ?? response.query_plan;
  const anchors = reasoning?.academic_anchors ?? response.academic_anchors;
  const aliases = reasoning?.derived_aliases ?? [];
  const execution = reasoning?.execution ?? response.pipeline.stages;

  const seedStage = execution.find((stage) => String(stage.name) === "day5_citation")
    ?? response.pipeline.stages.find((stage) => String(stage.name) === "day5_citation");
  const planCount = plans.length;
  const seedCount = numeric(seedStage?.seed_count);

  return <section className={`reasoning-card ${expanded ? "expanded" : ""}`}>
    <button className="reasoning-summary" onClick={onToggle}>
      <span className="reasoning-title"><Sparkles size={18} />{copy.reasoning}</span>
      <span className="reason-step"><CircleCheck size={17} />{copy.understood}</span>
      <span className="reason-arrow">→</span>
      <span className="reason-step"><span className="count-badge">{planCount}</span>{copy.searchPlans}</span>
      {seedCount != null && <><span className="reason-arrow">→</span><span className="reason-step"><Network size={17} />{seedCount} {copy.seedPapers}</span></>}
      <ChevronDown size={19} className="reason-chevron" />
    </button>

    {expanded && <div className="reasoning-grid">
      <ReasonPanel icon={<SearchCheck size={18} />} title={copy.researchUnderstanding}>
        {reasoning?.cleaned_query && reasoning.cleaned_query !== reasoning.original_query && (
          <div className="plan-row">
            <span className="plan-number">Q</span>
            <div>
              <strong>{reasoning.cleaned_query}</strong>
              <small>{copy.cleanedQuery}</small>
            </div>
          </div>
        )}

        {constraints.length
          ? constraints.map((item, index) => (
              <div className="constraint-row" key={`constraint-${index}`}>
                <span>{stringValue(item.text) || stringValue(item.id) || `Constraint ${index + 1}`}</span>
                <small>{stringValue(item.constraint_type)}</small>
              </div>
            ))
          : <EmptyLine />}

        {candidateSubqueries.length > 0 && <>
          <small>{copy.candidateSubqueries}</small>
          {candidateSubqueries.map((item, index) => (
            <div className="plan-row" key={`subquery-${index}`}>
              <span className="plan-number">{index + 1}</span>
              <div>
                <strong>{stringValue(item.text) || `Subquery ${index + 1}`}</strong>
                <small>{[stringValue(item.subquery_type), stringValue(item.reason)].filter(Boolean).join(" · ")}</small>
              </div>
            </div>
          ))}
        </>}
      </ReasonPanel>

      <ReasonPanel icon={<Network size={18} />} title={copy.searchPlans}>
        {plans.length
          ? plans.map((item, index) => (
              <div className="plan-row" key={`plan-${index}`}>
                <span className="plan-number">{index + 1}</span>
                <div>
                  <strong>{stringValue(item.text) || `Plan ${index + 1}`}</strong>
                  <small>{[stringValue(item.plan_type), priorityText(item)].filter(Boolean).join(" · ")}</small>
                  {stringValue(item.reason) && <small>{stringValue(item.reason)}</small>}
                </div>
              </div>
            ))
          : <EmptyLine />}
      </ReasonPanel>

      <ReasonPanel icon={<Anchor size={18} />} title={copy.anchors}>
        {anchors.length
          ? anchors.map((item, index) => (
              <div className="anchor-row" key={`anchor-${index}`}>
                {stringValue(item.text) || stringValue(item.title) || JSON.stringify(item)}
              </div>
            ))
          : <div className="empty-anchor"><Anchor size={30} /><strong>{copy.noAnchors}</strong><span>{copy.noAnchorsHelp}</span></div>}

        {aliases.length > 0 && <>
          <small>{copy.derivedAliases}</small>
          {aliases.map((alias, index) => <div className="anchor-row" key={`alias-${index}`}>{alias}</div>)}
        </>}
      </ReasonPanel>

      <ReasonPanel icon={<Sparkles size={18} />} title={copy.enhancements}>
        <div className="enhancement-list">
          {execution.length
            ? execution.map((stage, index) => (
                <div key={`execution-${index}`}>
                  <CircleCheck size={15} />
                  <span>{prettyStage(String(stage.name || `stage_${index + 1}`))}</span>
                  <small>{String(stage.status || "")}</small>
                </div>
              ))
            : <EmptyLine />}
        </div>
      </ReasonPanel>
    </div>}
  </section>;
}

function ReasonPanel({ icon, title, children }: { icon: React.ReactNode; title: string; children: React.ReactNode }) {
  return <div className="reason-panel"><h3>{icon}{title}</h3><div className="reason-panel-body">{children}</div></div>;
}

function EmptyLine() { return <div className="muted-empty">—</div>; }
function stringValue(value: unknown) { return typeof value === "string" ? value : ""; }
function numeric(value: unknown) { return typeof value === "number" ? value : null; }
function priorityText(item: Record<string, unknown>) {
  const p = item.priority;
  const c = item.confidence;
  return typeof p === "number" ? `Priority ${p}${typeof c === "number" ? ` · ${c.toFixed(2)}` : ""}` : "";
}
function prettyStage(name: string) {
  return name.replace(/_/g, " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
}
