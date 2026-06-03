import { ASK_MODES } from "../api/ask";

const MODE_OPTIONS = [
  {
    id: ASK_MODES.rag,
    label: "Fixed RAG",
    hint: "Retrieve passages and generate — always shows sources",
  },
  {
    id: ASK_MODES.agent,
    label: "Agent",
    hint: "Tool-calling — list, search, or document detail as needed",
  },
];

export default function ModeToggle({ mode, onChange, disabled }) {
  return (
    <div className="mode-toggle mb-3 text-start">
      <div className="d-flex flex-wrap align-items-center gap-2 mb-1">
        <span className="small text-muted fw-semibold">Mode</span>
        <div className="btn-group btn-group-sm" role="group" aria-label="Ask mode">
          {MODE_OPTIONS.map((option) => (
            <button
              key={option.id}
              type="button"
              className={`btn ${mode === option.id ? "btn-primary" : "btn-outline-primary"}`}
              aria-pressed={mode === option.id}
              disabled={disabled}
              onClick={() => onChange(option.id)}
            >
              {option.label}
            </button>
          ))}
        </div>
      </div>
      <p className="small text-muted mb-0 mode-toggle-hint">
        {MODE_OPTIONS.find((o) => o.id === mode)?.hint}
      </p>
    </div>
  );
}
