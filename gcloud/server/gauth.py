from flask import Blueprint, request, jsonify
from flask_jwt_extended import create_access_token
from google.oauth2 import id_token
from google.auth.transport import requests
import os

auth_bp = Blueprint('auth', __name__)

GOOGLE_CLIENT_ID = os.environ.get('GOOGLE_CLIENT_ID')

@auth_bp.route('/auth/google', methods=['POST'])
def google_auth():
    token = request.json.get('token')

    try:
        idinfo = id_token.verify_oauth2_token(token, requests.Request(), GOOGLE_CLIENT_ID)
        if idinfo['iss'] not in ['accounts.google.com', 'https://accounts.google.com']:
            raise ValueError('Wrong issuer.')

        email = idinfo['email']
        # Note: You should check if the user exists in your database here
        # and create a new user if they don't exist

        return jsonify({"token": create_access_token(identity=email)}), 200
    except ValueError:
        return jsonify({"error": "Invalid token"}), 400
