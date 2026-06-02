const EXAMPLE_QUESTIONS = [
  "What is REMS?",
  "What human factors documentation is needed for drug applications?",
  "When should sponsors submit a REMS assessment?",
];

export default function EmptyStateWelcome({ onSelectExample, disabled }) {
  return (
    <div className="card empty-state mb-3">
      <div className="card-body text-start">
        <h2 className="h5 mb-2">Welcome</h2>
        <p className="text-muted mb-3 mb-md-4">
          Answers are based <strong>only</strong> on retrieved FDA guidance
          documents—not general medical or legal advice. Each response includes
          numbered sources you can verify.
        </p>
        <p className="small fw-semibold text-secondary mb-2">Try an example:</p>
        <div className="d-flex flex-column gap-2">
          {EXAMPLE_QUESTIONS.map((example) => (
            <button
              key={example}
              type="button"
              className="btn btn-outline-primary btn-sm example-btn text-start"
              disabled={disabled}
              onClick={() => onSelectExample(example)}
            >
              {example}
            </button>
          ))}
        </div>
        <p className="small text-muted mt-3 mb-0">
          Or type your question below and press Enter.
        </p>
      </div>
    </div>
  );
}

export { EXAMPLE_QUESTIONS };
