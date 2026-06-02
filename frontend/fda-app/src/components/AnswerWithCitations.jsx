import ReactMarkdown from "react-markdown";

function linkifyCitations(text) {
  return text.replace(/\[(\d+)\]/g, "[[$1]](#source-$1)");
}

export default function AnswerWithCitations({ answer, onCitationClick }) {
  return (
    <ReactMarkdown
      components={{
        a: ({ href, children }) => {
          if (href?.startsWith("#source-")) {
            const sourceId = Number(href.replace("#source-", ""));
            return (
              <button
                type="button"
                className="citation-link"
                aria-label={`Jump to source ${sourceId}`}
                onClick={(e) => {
                  e.preventDefault();
                  onCitationClick(sourceId);
                }}
              >
                [{sourceId}]
              </button>
            );
          }

          return (
            <a href={href} target="_blank" rel="noreferrer">
              {children}
            </a>
          );
        },
      }}
    >
      {linkifyCitations(answer)}
    </ReactMarkdown>
  );
}
