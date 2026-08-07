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
  constraint_evidence: Record<string, unknown>[];
  retrieval_sources: string[];
  citation_path?: CitationPathView | null;
}

export interface PipelineSummary {
  stages: Record<string, unknown>[];
  total_candidates: number;
  returned_results: number;
}

export interface SearchResponse {
  run_id: string;
  query: string;
  qid?: string | null;
  mode: SearchMode;
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
