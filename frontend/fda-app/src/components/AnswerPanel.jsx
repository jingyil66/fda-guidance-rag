import { ASK_MODES } from "../api/ask";
import AnswerWithCitations from "./AnswerWithCitations";

export default function AnswerPanel({
  mode = ASK_MODES.agent,
  loading,
  error,
  answer,
  onCitationClick,
}) {
  const modeLabel = mode === ASK_MODES.rag ? "Fixed RAG" : "Agent";
  const loadingText =
    mode === ASK_MODES.rag
      ? "Retrieving passages and generating answer…"
      : "Agent is selecting tools and generating answer…";

  return (
    <div className="card mb-3 answer-card">
      <div className="card-header d-flex justify-content-between align-items-center">
        <span>Answer</span>
        {!loading && answer && (
          <span className="badge text-bg-secondary">{modeLabel}</span>
        )}
      </div>
      <div className="card-body answer-body text-start">
        {loading && (
          <div className="d-flex align-items-center gap-2 text-muted">
            <div
              className="spinner-border spinner-border-sm"
              role="status"
              aria-hidden="true"
            />
            <span>{loadingText}</span>
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
              onCitationClick={onCitationClick}
            />
          </div>
        )}
      </div>
    </div>
  );
}
