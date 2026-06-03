function sourcePdfUrl(source) {
  if (source.url) {
    return source.url;
  }
  if (source.pdf_id) {
    return `https://www.fda.gov/media/${source.pdf_id}/download`;
  }
  return "";
}

export default function SourceList({
  sources,
  highlightedSource,
  expandedSources,
  onToggleSnippet,
}) {
  if (sources.length === 0) {
    return null;
  }

  return (
    <div className="card sources-card">
      <div className="card-header">Sources ({sources.length})</div>
      <div className="list-group list-group-flush">
        {sources.map((source, idx) => {
          const sourceNum = idx + 1;
          const pdfUrl = sourcePdfUrl(source);
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
                        onClick={() => onToggleSnippet(sourceNum)}
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
                {pdfUrl && (
                  <a
                    href={pdfUrl}
                    className="btn btn-sm btn-outline-primary flex-shrink-0 source-open-btn"
                    target="_blank"
                    rel="noreferrer"
                  >
                    Open PDF
                  </a>
                )}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
