export type SearchMode = "benchmark" | "live";
export type Language = "zh" | "en";

export interface CostSummary {
  api_calls: number;
  cache_hits: number;
  retries: number;
  estimated_cost_usd: number;
}

export interface CitationPathView {
  seed_openalex_id?: string | null;
  seed_title?: string | null;
  expanded_openalex_id?: string | null;
  expanded_title?: string | null;
  edge_type?: string | null;
  hop?: number | null;
}

export type ConstraintEvidenceField = "title" | "abstract" | "concept" | null;

export interface ConstraintEvidenceItem {
  constraint_id: string;
  constraint_text: string;
  constraint_type: string;
  matched: boolean;
  match_type: string;
  evidence_field: ConstraintEvidenceField;
  evidence_text?: string | null;
  confidence: number;
  token_coverage: number;
}

export interface PaperResult {
  rank: number;
  title: string;
  authors: string[];
  year?: number | null;
  venue?: string | null;
  doi?: string | null;
  arxiv_id?: string | null;
  openalex_id?: string | null;
  url?: string | null;
  score?: number | null;
  relevance_level?: string | null;
  reason_tags: string[];
  reason_text?: string | null;
  constraint_evidence: ConstraintEvidenceItem[];
  matched_constraints: string[];
  unmatched_constraints: string[];
  matched_count: number;
  constraint_count: number;
  retrieval_sources: string[];
  citation_path?: CitationPathView | null;
}

export interface PipelineSummary {
  stages: Record<string, unknown>[];
  total_candidates: number;
  returned_results: number;
}

export interface SearchReasoning {
  original_query: string;
  cleaned_query: string;
  constraints: Record<string, unknown>[];
  candidate_subqueries: Record<string, unknown>[];
  academic_anchors: Record<string, unknown>[];
  derived_aliases: string[];
  filters: Record<string, unknown>;
  selected_plans: Record<string, unknown>[];
  execution: Record<string, unknown>[];
}

export interface SearchResponse {
  run_id: string;
  query: string;
  qid?: string | null;
  mode: SearchMode;
  reasoning?: SearchReasoning | null;
  parsed_constraints: Record<string, unknown>[];
  academic_anchors: Record<string, unknown>[];
  query_plan: Record<string, unknown>[];
  results: PaperResult[];
  pipeline: PipelineSummary;
  cost: CostSummary;
  latency_ms: number;
  warnings: string[];
}

export interface QueryListItem {
  qid: string;
  question: string;
  day4_rescue_triggered: boolean;
  day5_citation_triggered: boolean;
}

export interface QueryListResponse {
  total: number;
  items: QueryListItem[];
}

export interface SearchRequest {
  query: string;
  qid: string | null;
  mode: SearchMode;
  top_k: 20 | 50 | 100;
  enable_citation: boolean;
}

export interface ApiErrorPayload {
  error?: {
    code?: string;
    message?: string;
    details?: unknown;
  };
}
