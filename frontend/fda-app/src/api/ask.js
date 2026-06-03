const REQUEST_TIMEOUT_MS = 120_000;

export const ASK_MODES = {
  rag: "rag",
  agent: "agent",
};

function resolveApiBase() {
  const base = import.meta.env.VITE_API_BASE;
  if (base) {
    return base.replace(/\/$/, "");
  }
  const legacy = import.meta.env.VITE_API_URL;
  if (legacy) {
    return legacy.replace(/\/ask_agent\/?$/, "").replace(/\/ask\/?$/, "");
  }
  return "http://127.0.0.1:5000";
}

const API_BASE = resolveApiBase();

const ENDPOINTS = {
  [ASK_MODES.rag]: `${API_BASE}/ask`,
  [ASK_MODES.agent]: `${API_BASE}/ask_agent`,
};

export class AskError extends Error {
  constructor(message, { status } = {}) {
    super(message);
    this.name = "AskError";
    this.status = status;
  }
}

export async function askQuestion(query, { mode = ASK_MODES.agent } = {}) {
  const trimmed = query.trim();
  if (!trimmed) {
    throw new AskError("Please enter a question.");
  }

  const url = ENDPOINTS[mode] ?? ENDPOINTS[ASK_MODES.agent];
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);

  try {
    const response = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query: trimmed }),
      signal: controller.signal,
    });

    const data = await response.json().catch(() => ({}));

    if (!response.ok) {
      throw new AskError(data.error || `Request failed (${response.status})`, {
        status: response.status,
      });
    }

    const answer = (data.answer || "").trim();
    if (!answer || answer.toLowerCase() === "query is empty") {
      throw new AskError("Please enter a question.");
    }

    return {
      answer,
      sources: data.sources || [],
      steps: mode === ASK_MODES.agent ? data.steps || [] : [],
      mode,
    };
  } catch (err) {
    if (err instanceof AskError) {
      throw err;
    }
    if (err instanceof Error && err.name === "AbortError") {
      throw new AskError(
        `Request timed out after ${REQUEST_TIMEOUT_MS / 1000} seconds. Try a shorter question or try again.`,
      );
    }
    throw new AskError("Unable to reach the API. Is the backend running?");
  } finally {
    clearTimeout(timeoutId);
  }
}
