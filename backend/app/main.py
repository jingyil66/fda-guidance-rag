from flask import Flask
from flask_cors import CORS
from backend.app.api.routes import register_routes
# create an instance of this class
app = Flask(__name__)
CORS(app)

register_routes(app)

if __name__ == '__main__':
    app.run(port=5000, debug=True)
