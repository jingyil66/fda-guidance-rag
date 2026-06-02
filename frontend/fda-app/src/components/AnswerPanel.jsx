import AnswerWithCitations from "./AnswerWithCitations";

export default function AnswerPanel({
  loading,
  error,
  answer,
  onCitationClick,
}) {
  return (
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
              onCitationClick={onCitationClick}
            />
          </div>
        )}
      </div>
    </div>
  );
}
