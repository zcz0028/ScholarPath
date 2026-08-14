const API_BASE_URL = (
  import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000"
).replace(/\/$/, "");

const DEFAULT_TIMEOUT_MS = 30_000;

export type ApiErrorKind =
  | "http"
  | "timeout"
  | "network"
  | "aborted"
  | "invalid_response";

export interface ApiFetchOptions {
  timeoutMs?: number;
}

export class ApiError extends Error {
  status: number;
  code?: string;
  kind: ApiErrorKind;
  retryable: boolean;

  constructor(
    message: string,
    status: number,
    code?: string,
    kind: ApiErrorKind = "http",
    retryable = false,
  ) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.code = code;
    this.kind = kind;
    this.retryable = retryable;
  }
}

function isAbortError(error: unknown): boolean {
  return (
    error instanceof DOMException &&
    error.name === "AbortError"
  );
}

function isRetryableHttpStatus(status: number): boolean {
  return (
    status === 408 ||
    status === 429 ||
    status === 502 ||
    status === 503 ||
    status === 504
  );
}

function normalizeTimeoutMs(value: number | undefined): number {
  if (value == null) {
    return DEFAULT_TIMEOUT_MS;
  }

  if (!Number.isFinite(value) || value <= 0) {
    return DEFAULT_TIMEOUT_MS;
  }

  return Math.floor(value);
}

export async function apiFetch<T>(
  path: string,
  init?: RequestInit,
  options?: ApiFetchOptions,
): Promise<T> {
  const timeoutMs = normalizeTimeoutMs(options?.timeoutMs);

  const timeoutController = new AbortController();
  let timedOut = false;

  const timeoutId = window.setTimeout(() => {
    timedOut = true;
    timeoutController.abort();
  }, timeoutMs);

  const externalSignal = init?.signal;
  const externalAbortHandler = () => {
    timeoutController.abort();
  };

  if (externalSignal) {
    if (externalSignal.aborted) {
      window.clearTimeout(timeoutId);

      throw new ApiError(
        "Request was cancelled.",
        0,
        "REQUEST_ABORTED",
        "aborted",
        false,
      );
    }

    externalSignal.addEventListener("abort", externalAbortHandler, {
      once: true,
    });
  }

  try {
    let response: Response;

    try {
      response = await fetch(`${API_BASE_URL}${path}`, {
        ...init,
        signal: timeoutController.signal,
        headers: {
          Accept: "application/json",
          ...(init?.body ? { "Content-Type": "application/json" } : {}),
          ...(init?.headers || {}),
        },
      });
    } catch (error) {
      if (isAbortError(error)) {
        if (timedOut) {
          throw new ApiError(
            `Request timed out after ${timeoutMs}ms.`,
            0,
            "REQUEST_TIMEOUT",
            "timeout",
            true,
          );
        }

        throw new ApiError(
          "Request was cancelled.",
          0,
          "REQUEST_ABORTED",
          "aborted",
          false,
        );
      }

      throw new ApiError(
        "Unable to reach the ScholarPath API. Check whether the backend service is running.",
        0,
        "NETWORK_ERROR",
        "network",
        true,
      );
    }

    if (!response.ok) {
      let message = `Request failed (${response.status})`;
      let code: string | undefined;

      try {
        const payload = await response.json();
        message =
          payload?.error?.message ||
          payload?.detail ||
          message;
        code = payload?.error?.code;
      } catch {
        // Keep the HTTP fallback message when the backend did not return JSON.
      }

      throw new ApiError(
        message,
        response.status,
        code,
        "http",
        isRetryableHttpStatus(response.status),
      );
    }

    try {
      return await response.json() as T;
    } catch {
      throw new ApiError(
        "The ScholarPath API returned an invalid JSON response.",
        response.status,
        "INVALID_JSON_RESPONSE",
        "invalid_response",
        false,
      );
    }
  } finally {
    window.clearTimeout(timeoutId);

    if (externalSignal) {
      externalSignal.removeEventListener(
        "abort",
        externalAbortHandler,
      );
    }
  }
}