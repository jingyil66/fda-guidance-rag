from flask import request, jsonify
from backend.app.core.config import settings
from backend.app.services.rag_service import get_answer

def register_routes(app):
    @app.route("/health", methods=["GET"])
    def health():
        """Liveness/readiness probe: Flask up + Qdrant collection reachable."""
        collection = settings.DEFAULT_QDRANT_COLLECTION
        try:
            from qdrant_client import QdrantClient

            client = QdrantClient(settings.DEFAULT_QDRANT_URL, timeout=5)
            if not client.collection_exists(collection):
                return jsonify(
                    {
                        "status": "degraded",
                        "qdrant": "collection_missing",
                        "collection": collection,
                    }
                ), 503

            points = client.get_collection(collection).points_count
            return jsonify(
                {
                    "status": "ok",
                    "collection": collection,
                    "points": points,
                }
            )
        except Exception as exc:
            return jsonify({"status": "degraded", "qdrant": str(exc)}), 503

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
