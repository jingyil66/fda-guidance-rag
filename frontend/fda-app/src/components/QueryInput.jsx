export default function QueryInput({
  query,
  loading,
  onChange,
  onSend,
}) {
  const handleKeyDown = (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      onSend();
    }
  };

  return (
    <div className="query-form mb-3">
      <input
        type="search"
        className="form-control query-input"
        placeholder="Ask about FDA guidance..."
        aria-label="AI query"
        value={query}
        disabled={loading}
        onChange={onChange}
        onKeyDown={handleKeyDown}
      />
      <button
        className="btn btn-primary btn-send"
        type="button"
        disabled={loading || !query.trim()}
        onClick={() => onSend()}
      >
        {loading ? "Thinking..." : "Send"}
      </button>
    </div>
  );
}
