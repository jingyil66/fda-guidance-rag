import { useState } from "react";

function Chatbot() {
    const [query, setQuery] = useState("");
    const [answer, setAnswer] = useState("");
    const [loading, setLoading] = useState(false);

    const handleSend = async () => {
        if (!query.trim()) return;
        setLoading(true);
        try {
            const response = await fetch("http://127.0.0.1:5000/ask", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ query }),
            });
            const data = await response.json();
            setAnswer(data.answer);
            } catch (err) {
            setAnswer("Error: Unable to get response");
            } finally {
            setLoading(false);
            }
        };

    return (
        <div className="container mb-4 align-items-center" style={{ maxWidth: "700px" }}>
            <div className="input-group mb-3">
            <input
                type="search"
                className="form-control"
                placeholder="Ask AI a question..."
                aria-label="AI query"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
            />
            <button className="btn btn-primary" onClick={handleSend}>
                {loading ? "Thinking..." : "Send"}
                </button>
            </div>

            <div className="card">
                <div className="card-header">Answer</div>
                <div className="card-body" style={{ minHeight: "300px" }}>
                    {answer || "AI answer will appear here..."}
                </div>
            </div>
        </div>
    )

}

export default Chatbot;