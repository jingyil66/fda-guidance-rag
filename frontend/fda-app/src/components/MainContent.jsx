import { useState } from "react";
import { ASK_MODES, AskError, askQuestion } from "../api/ask";
import AgentSteps from "./AgentSteps";
import AnswerPanel from "./AnswerPanel";
import EmptyStateWelcome from "./EmptyStateWelcome";
import ModeToggle from "./ModeToggle";
import QueryInput from "./QueryInput";
import SourceList from "./SourceList";

const MODE_STORAGE_KEY = "fda-ask-mode";

function readStoredMode() {
  try {
    const stored = localStorage.getItem(MODE_STORAGE_KEY);
    if (stored === ASK_MODES.rag || stored === ASK_MODES.agent) {
      return stored;
    }
  } catch {
    /* ignore */
  }
  return ASK_MODES.agent;
}

function Chatbot() {
  const [mode, setMode] = useState(readStoredMode);
  const [query, setQuery] = useState("");
  const [answer, setAnswer] = useState("");
  const [sources, setSources] = useState([]);
  const [steps, setSteps] = useState([]);
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
    setSteps([]);
    setHighlightedSource(null);
    setExpandedSources(new Set());

    try {
      const result = await askQuestion(trimmed, { mode });
      setAnswer(result.answer);
      setSources(result.sources);
      setSteps(result.steps || []);
    } catch (err) {
      setError(err instanceof AskError ? err.message : "Something went wrong.");
    } finally {
      setLoading(false);
    }
  };

  const handleModeChange = (nextMode) => {
    if (nextMode === mode || loading) {
      return;
    }
    setMode(nextMode);
    try {
      localStorage.setItem(MODE_STORAGE_KEY, nextMode);
    } catch {
      /* ignore */
    }
    setAnswer("");
    setSources([]);
    setSteps([]);
    setError("");
    setHighlightedSource(null);
    setExpandedSources(new Set());
  };

  return (
    <div className="container mb-4 chat-container">
      <ModeToggle mode={mode} onChange={handleModeChange} disabled={loading} />

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
        <>
          {mode === ASK_MODES.agent && (
            <AgentSteps steps={steps} loading={loading} />
          )}
          <AnswerPanel
            mode={mode}
            loading={loading}
            error={error}
            answer={answer}
            onCitationClick={handleCitationClick}
          />
        </>
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
