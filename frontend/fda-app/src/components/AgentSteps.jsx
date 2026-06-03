const TOOL_LABELS = {
  search_guidance: "Search guidance",
  list_guidance: "List guidances",
  get_guidance_detail: "Guidance detail",
};

function formatArgValue(value) {
  if (value === null || value === undefined) return "";
  if (typeof value === "object") return JSON.stringify(value);
  return String(value);
}

function formatArgs(args) {
  if (!args || typeof args !== "object") return null;
  const parts = Object.entries(args)
    .filter(([, value]) => value !== null && value !== undefined && value !== "")
    .map(([key, value]) => `${key}: ${formatArgValue(value)}`);
  return parts.length > 0 ? parts.join(" · ") : null;
}

export default function AgentSteps({ steps, loading }) {
  if (!loading && (!steps || steps.length === 0)) {
    return null;
  }

  return (
    <div className="card mb-3 agent-steps-card">
      <div className="card-header d-flex align-items-center justify-content-between">
        <span>Agent steps</span>
        {!loading && steps?.length > 0 && (
          <span className="badge text-bg-secondary">{steps.length}</span>
        )}
      </div>
      <div className="card-body agent-steps-body text-start">
        {loading && (
          <div className="d-flex align-items-center gap-2 text-muted">
            <div
              className="spinner-border spinner-border-sm"
              role="status"
              aria-hidden="true"
            />
            <span>Running tools…</span>
          </div>
        )}
        {!loading &&
          steps?.map((step, index) => {
            const tool = step.tool || "unknown";
            const label = TOOL_LABELS[tool] || tool;
            const argsText = formatArgs(step.args);

            return (
              <div key={`${tool}-${index}`} className="agent-step">
                <div className="agent-step-header">
                  <span className="agent-step-index">{index + 1}</span>
                  <span className="badge text-bg-primary agent-step-tool">{label}</span>
                </div>
                {argsText && <div className="agent-step-args small text-muted">{argsText}</div>}
              </div>
            );
          })}
      </div>
    </div>
  );
}
