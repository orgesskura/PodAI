from flask import Flask, request, jsonify, session
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity
from flask_cors import CORS
from dotenv import load_dotenv
import os
import secrets
from google.cloud.sql.connector import Connector
import sqlalchemy

load_dotenv()

app = Flask(__name__)
#CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}}, supports_credentials=True)
CORS(app, resources={r"/api/*": {"origins": "*"}}, supports_credentials=True)
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "./totally-secret.json"

# app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///app.db'
# app.config['SQLALCHEMY_DATABASE_URI'] = "postgresql://postgres:ir2KrHy#(Egf'R]/@/cloudsql/podai-434000:europe-west2:podai/podai"

def getconn():
    connector = Connector()
    conn = connector.connect(
        "podai-434000:europe-west2:podai",
        "pg8000",
        user="postgres",
        password=r"Z\Sh>6onagS<b8OQ",
        db="postgres"
    )
    return conn

app.config['SQLALCHEMY_DATABASE_URI'] = "postgresql+pg8000://"
app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {
    "creator": getconn
}
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['JWT_SECRET_KEY'] = os.environ.get('JWT_SECRET_KEY', secrets.token_hex(32))

db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
jwt = JWTManager(app)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(60), nullable=False)
    name = db.Column(db.String(100), nullable=True)

class Feedback(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    message = db.Column(db.String(500), nullable=False)
    is_positive = db.Column(db.Boolean, nullable=False)

@app.route('/api/register', methods=['POST'])
def register():
    data = request.json
    if not data or not data.get('email') or not data.get('password'):
        return jsonify({"error": "Email and password are required"}), 400

    if User.query.filter_by(email=data['email']).first():
        return jsonify({"error": "Email already registered"}), 400

    new_user = User(
        email=data['email'],
        password=bcrypt.generate_password_hash(data['password']).decode('utf-8'),
        name=data.get('name')
    )
    db.session.add(new_user)
    db.session.commit()

    access_token = create_access_token(identity=data['email'])
    return jsonify({
        "message": "User registered successfully",
        "token": access_token,
        "name": data.get('name')
    }), 201

@app.route('/api/login', methods=['POST'])
def login():
    data = request.json
    if not data.get('email') or not data.get('password'):
        return jsonify({"error": "Email and password are required"}), 400

    user = User.query.filter_by(email=data['email']).first()
    if user and bcrypt.check_password_hash(user.password, data['password']):
        return jsonify({
            "message": "Login successful",
            "token": create_access_token(identity=data['email']),
            "name": user.name
        }), 200
    return jsonify({"error": "Invalid email or password"}), 401

@app.route('/api/logout', methods=['POST'])
@jwt_required()
def logout():
    return jsonify({"message": "Logout successful"}), 200

@app.route('/chat', methods=['POST'])
@jwt_required(optional=True)
def chat():
    current_user = get_jwt_identity()
    message = request.json.get('message')

    if not message:
        return jsonify({"error": "Message is required"}), 400

    if not current_user:
        query_count = session.get('query_count', 0)
        if query_count >= 3:
            return jsonify({"error": "Please login or register to continue using the chatbot"}), 401
        session['query_count'] = query_count + 1

    # Replace this with your actual chat processing logic
    response = "This is a sample response from the chatbot."

    return jsonify({"response": response}), 200

@app.route('/feedback', methods=['POST'])
@jwt_required()
def submit_feedback():
    user = User.query.filter_by(email=get_jwt_identity()).first()
    data = request.json

    if data.get('message') is None or data.get('is_positive') is None:
        return jsonify({"error": "Message and feedback type are required"}), 400

    new_feedback = Feedback(user_id=user.id, message=data['message'], is_positive=data['is_positive'])
    db.session.add(new_feedback)
    db.session.commit()

    return jsonify({"message": "Feedback submitted successfully"}), 201

@app.route('/feedback', methods=['GET'])
@jwt_required()
def get_feedback():
    user = User.query.filter_by(email=get_jwt_identity()).first()
    feedback = Feedback.query.filter_by(user_id=user.id).all()
    return jsonify({"feedback": [{"message": f.message, "is_positive": f.is_positive} for f in feedback]}), 200

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', 'http://localhost:3000')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    response.headers.add('Access-Control-Allow-Credentials', 'true')
    return response

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))

# if __name__ == '__main__':
#     with app.app_context():
#         db.create_all()
#     app.run(debug=True, host='0.0.0.0', port=5000)