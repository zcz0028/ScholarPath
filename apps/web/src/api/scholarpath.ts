import { apiFetch } from "./client";
import type { QueryListResponse, SearchRequest, SearchResponse } from "../types/api";

export function getBenchmarkQueries(): Promise<QueryListResponse> {
  return apiFetch<QueryListResponse>("/api/queries");
}

export function searchScholarPath(request: SearchRequest): Promise<SearchResponse> {
  return apiFetch<SearchResponse>("/api/search", {
    method: "POST",
    body: JSON.stringify(request),
  });
}

export function getHealth(): Promise<Record<string, unknown>> {
  return apiFetch<Record<string, unknown>>("/health");
}
