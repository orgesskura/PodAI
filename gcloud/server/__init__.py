from flask import Flask
from flask_cors import CORS
from flask_jwt_extended import JWTManager
from .gauth import auth_bp
from .chat import chat_bp
from .feedback import feedback_bp

def create_app():
    app = Flask(__name__)
    CORS(app)
    
    app.config['JWT_SECRET_KEY'] = 'your-secret-key'  # Change this to a secure secret key
    jwt = JWTManager(app)

    app.register_blueprint(auth_bp, url_prefix='/api')
    app.register_blueprint(chat_bp, url_prefix='/api')
    app.register_blueprint(feedback_bp, url_prefix='/api')

    return app
