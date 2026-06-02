const API_URL = import.meta.env.VITE_API_URL || "http://127.0.0.1:5000/ask";
const REQUEST_TIMEOUT_MS = 120_000;

export class AskError extends Error {
  constructor(message, { status } = {}) {
    super(message);
    this.name = "AskError";
    this.status = status;
  }
}

export async function askQuestion(query) {
  const trimmed = query.trim();
  if (!trimmed) {
    throw new AskError("Please enter a question.");
  }

  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);

  try {
    const response = await fetch(API_URL, {
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

    return { answer, sources: data.sources || [] };
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
