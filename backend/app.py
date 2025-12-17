from flask import Flask, request, jsonify
from flask_cors import CORS
from rag import get_answer

app = Flask(__name__)
CORS(app)

# --- RAG ---
@app.route("/ask", methods=["POST"])
def ask():
    data = request.json
    query = data.get("query")
    if not query:
        return jsonify({"error": "No query provided"}), 400
    answer = get_answer(query)
    return jsonify({"answer": answer})

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
