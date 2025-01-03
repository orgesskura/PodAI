from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity
from .utils import get_chat_response

chat_bp = Blueprint('chat', __name__)

@chat_bp.route('/chat', methods=['POST'])
@jwt_required()
def chat():
    data = request.json
    if not data or 'message' not in data:
        return jsonify({"error": "No message provided"}), 400

    user_id = get_jwt_identity()
    user_input = data['message']
    response = get_chat_response(user_input)
    return jsonify({"response": response}), 200
