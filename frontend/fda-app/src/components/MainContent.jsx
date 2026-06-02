import { useState } from "react";
import { AskError, askQuestion } from "../api/ask";
import AnswerPanel from "./AnswerPanel";
import EmptyStateWelcome from "./EmptyStateWelcome";
import QueryInput from "./QueryInput";
import SourceList from "./SourceList";

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
      const result = await askQuestion(trimmed);
      setAnswer(result.answer);
      setSources(result.sources);
    } catch (err) {
      setError(err instanceof AskError ? err.message : "Something went wrong.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container mb-4 chat-container">
      {showEmptyState && (
        <EmptyStateWelcome onSelectExample={handleSend} disabled={loading} />
      )}

      <QueryInput
        query={query}
        loading={loading}
        onChange={(e) => setQuery(e.target.value)}
        onSend={handleSend}
      />

      {!showEmptyState && (
        <AnswerPanel
          loading={loading}
          error={error}
          answer={answer}
          onCitationClick={handleCitationClick}
        />
      )}

      {!loading && (
        <SourceList
          sources={sources}
          highlightedSource={highlightedSource}
          expandedSources={expandedSources}
          onToggleSnippet={toggleSnippet}
        />
      )}
    </div>
  );
}

export default Chatbot;
