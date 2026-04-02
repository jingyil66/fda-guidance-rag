from flask import request, jsonify
from app.services.rag_service import get_answer

def register_routes(app):
    # --- RAG query ---
    @app.route('/ask', methods=["POST"])
    def ask():
        data = request.json
        query = data.get("query", "")

        if not query:
            return jsonify({"answer": "Query is empty", "sources": []})

        result = get_answer(query)
        return jsonify({
            "success": True,
            "answer": result["answer"],
            "sources": result["sources"]
        })
    # --- Document Management---
    
    # get document metadata
    # @app.route('/documents', methods=["POST"])
    # def ask():
    #     return "ask"
    # upload documents to database
    # update documents
    # delete document, methods=['GET', 'POST']s
