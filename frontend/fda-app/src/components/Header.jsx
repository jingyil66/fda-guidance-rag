function Header() {
  return (
    <div className="container">
      <header className="d-flex flex-wrap justify-content-center py-3 mb-2 border-bottom">
        <div className="d-flex flex-column mb-3 mb-md-0 me-md-auto text-start">
          <span className="fs-4 fw-semibold">FDA Guidance RAG Assistant</span>
          <small className="text-muted header-subtitle">
            Search 2,000+ FDA medical guidance documents · answers grounded in retrieved sources
          </small>
        </div>
      </header>
    </div>
  );
}

export default Header;
