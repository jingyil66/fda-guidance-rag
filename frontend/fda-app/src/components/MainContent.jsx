import { useState } from "react";
import AnswerWithCitations from "./AnswerWithCitations";
import EmptyStateWelcome from "./EmptyStateWelcome";

const API_URL = import.meta.env.VITE_API_URL || "http://127.0.0.1:5000/ask";
const REQUEST_TIMEOUT_MS = 120_000;

function Chatbot() {
  const [query, setQuery] = useState("");
  const [answer, setAnswer] = useState("");
  const [sources, setSources] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [highlightedSource, setHighlightedSource] = useState(null);
  const [expandedSources, setExpandedSources] = useState(() => new Set());

  const showEmptyState = !loading && !error && !answer;

  const toggleSnippet = (sourceNum) => {
    setExpandedSources((prev) => {
      const next = new Set(prev);
      if (next.has(sourceNum)) {
        next.delete(sourceNum);
      } else {
        next.add(sourceNum);
      }
      return next;
    });
  };

  const handleCitationClick = (sourceNum) => {
    setHighlightedSource(sourceNum);
    setExpandedSources((prev) => new Set(prev).add(sourceNum));
    document.getElementById(`source-${sourceNum}`)?.scrollIntoView({
      behavior: "smooth",
      block: "nearest",
    });
  };

  const handleSend = async (text = query) => {
    const trimmed = text.trim();
    if (!trimmed || loading) return;

    setQuery(trimmed);
    setLoading(true);
    setError("");
    setAnswer("");
    setSources([]);
    setHighlightedSource(null);
    setExpandedSources(new Set());

    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);

      const response = await fetch(API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: trimmed }),
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      const data = await response.json().catch(() => ({}));

      if (!response.ok) {
        setError(data.error || `Request failed (${response.status})`);
        return;
      }

      const answerText = (data.answer || "").trim();
      if (!answerText || answerText.toLowerCase() === "query is empty") {
        setError("Please enter a question.");
        return;
      }

      setAnswer(answerText);
      setSources(data.sources || []);
    } catch (err) {
      if (err instanceof Error && err.name === "AbortError") {
        setError(
          `Request timed out after ${REQUEST_TIMEOUT_MS / 1000} seconds. Try a shorter question or try again.`,
        );
      } else {
        setError("Unable to reach the API. Is the backend running?");
      }
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="container mb-4 chat-container">
      {showEmptyState && (
        <EmptyStateWelcome onSelectExample={handleSend} disabled={loading} />
      )}

      <div className="query-form mb-3">
        <input
          type="search"
          className="form-control query-input"
          placeholder="Ask about FDA guidance..."
          aria-label="AI query"
          value={query}
          disabled={loading}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={handleKeyDown}
        />
        <button
          className="btn btn-primary btn-send"
          type="button"
          disabled={loading || !query.trim()}
          onClick={() => handleSend()}
        >
          {loading ? "Thinking..." : "Send"}
        </button>
      </div>

      {!showEmptyState && (
        <div className="card mb-3 answer-card">
          <div className="card-header">Answer</div>
          <div className="card-body answer-body text-start">
            {loading && (
              <div className="d-flex align-items-center gap-2 text-muted">
                <div
                  className="spinner-border spinner-border-sm"
                  role="status"
                  aria-hidden="true"
                />
                <span>Retrieving guidance and generating answer…</span>
              </div>
            )}
            {!loading && error && (
              <div className="alert alert-danger mb-0 py-2" role="alert">
                {error}
              </div>
            )}
            {!loading && !error && answer && (
              <div className="answer-markdown">
                <AnswerWithCitations
                  answer={answer}
                  onCitationClick={handleCitationClick}
                />
              </div>
            )}
          </div>
        </div>
      )}

      {!loading && sources.length > 0 && (
        <div className="card sources-card">
          <div className="card-header">Sources ({sources.length})</div>
          <div className="list-group list-group-flush">
            {sources.map((source, idx) => {
              const sourceNum = idx + 1;
              return (
                <div
                  key={`${source.pdf_id || source.title}-${idx}`}
                  id={`source-${sourceNum}`}
                  className={`list-group-item source-item ${
                    highlightedSource === sourceNum ? "source-highlight" : ""
                  }`}
                >
                  <div className="d-flex justify-content-between align-items-start gap-2 source-item-row">
                    <div className="text-start flex-grow-1">
                      <span className="badge text-bg-secondary me-2">[{sourceNum}]</span>
                      <strong>{source.title || "Unknown"}</strong>
                      <div className="small text-muted mt-1">
                        PDF ID: {source.pdf_id || "—"} · Page {source.page ?? "?"}
                        {source.field_communication_type && (
                          <> · {source.field_communication_type}</>
                        )}
                      </div>
                      {source.snippet && (
                        <div className="mt-2">
                          <button
                            type="button"
                            className="btn btn-link btn-sm p-0 source-snippet-toggle"
                            aria-expanded={expandedSources.has(sourceNum)}
                            onClick={() => toggleSnippet(sourceNum)}
                          >
                            {expandedSources.has(sourceNum)
                              ? "Hide excerpt"
                              : "Show excerpt"}
                          </button>
                          {expandedSources.has(sourceNum) && (
                            <div className="source-snippet">{source.snippet}</div>
                          )}
                        </div>
                      )}
                    </div>
                    {source.url && (
                      <a
                        href={source.url}
                        className="btn btn-sm btn-outline-primary flex-shrink-0 source-open-btn"
                        target="_blank"
                        rel="noreferrer"
                      >
                        Open FDA
                      </a>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}

export default Chatbot;
