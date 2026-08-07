import { ArrowRight, CheckCircle2, Clock3 } from "lucide-react";
import type { Language, SearchResponse } from "../types/api";
import { useCopy } from "../i18n/translations";

export function PipelineBar({ language, response }: { language: Language; response: SearchResponse | null }) {
  const copy = useCopy(language);
  return <section className="pipeline-bar">
    <div className="pipeline-title"><CheckCircle2 size={17} />{copy.pipeline}</div>
    <div className="pipeline-stages">{response?.pipeline.stages.length ? response.pipeline.stages.map((stage, index) => <div className="stage-wrap" key={index}><div className="stage"><span>{index + 1}</span><div><strong>{pretty(String(stage.name || `stage_${index + 1}`))}</strong><small>{String(stage.status || "")}</small></div></div>{index < response.pipeline.stages.length - 1 && <ArrowRight size={15} />}</div>) : <span className="pipeline-placeholder">—</span>}</div>
    <div className="pipeline-time"><Clock3 size={18} /><div><small>{copy.totalTime}</small><strong>{response ? format(response.latency_ms) : "—"}</strong></div></div>
  </section>;
}
function pretty(value: string) { return value.replace(/_/g, " ").replace(/\b\w/g, (letter) => letter.toUpperCase()); }
function format(ms: number) { return ms >= 1000 ? `${(ms / 1000).toFixed(2)}s` : `${ms}ms`; }
