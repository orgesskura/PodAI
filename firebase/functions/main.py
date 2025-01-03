import os
from flask import Flask, request, jsonify, send_from_directory
import firebase_admin
from firebase_admin import credentials, storage, auth
from flask_cors import CORS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import PromptTemplate 
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
import pickle
import time
from openai import RateLimitError
import logging
from dotenv import load_dotenv
from firebase_functions import https_fn
from firebase_admin import initialize_app

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__, static_folder='lex-chat/build/')
CORS(app, resources={r"/*": {"origins": "*"}})

logging.basicConfig(level=logging.INFO)

# Initialize Firebase

cred = credentials.Certificate("podai-8fe41-firebase.json")
firebase_admin.initialize_app(cred, {
    'storageBucket': 'podai-8fe41.appspot.com'
})

# Initialize global variables
models = ["gpt-4o-mini", "gpt-3.5-turbo-16k", "gpt-3.5-turbo"]
llm = None
db = None
bm25_retriever = None
hybrid_retriever = None
chat_histories = {}

bucket_name = 'podai'
bucket = storage.bucket(bucket_name)

BASE_DIR = "/tmp" 

def download_file(folder_directory, file_path):
    bucket = storage.bucket()
    blob = bucket.blob(file_path)
    
    local_path = os.path.join(folder_directory, file_path)
    print(local_path)
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    
    try:
        blob.download_to_filename(local_path)
        return local_path
    except Exception as e:
        print(f"Error downloading file: {e}")
        return None

def download_folder(folder_path):
    bucket = storage.bucket()
    blobs = bucket.list_blobs(prefix=folder_path)
    
    local_folder_path = os.path.join(BASE_DIR, folder_path)
    os.makedirs(local_folder_path, exist_ok=True)
    
    downloaded_files = []
    print(blobs)
    for blob in blobs:
        if blob.name.endswith('/'):  # Skip directories
            continue
        local_file_path = os.path.join(BASE_DIR, blob.name)
        os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
        blob.download_to_filename(local_file_path)
        downloaded_files.append(local_file_path)
    
    return downloaded_files

def initialize_or_load_data():
    print("Loading existing data...")
    global llm, db, bm25_retriever, hybrid_retriever
    local_doc_path = download_file(BASE_DIR ,'all_documents.pkl')
    with open(local_doc_path, "rb") as f:
        all_documents = pickle.load(f)
    bm25_retriever = BM25Retriever.from_documents(all_documents)

    os.makedirs(BASE_DIR+'/faiss_index_all_transcripts', exist_ok=True)

    _ = download_file(BASE_DIR, 'fais_index_all_transcripts/index.faiss')
    _ = download_file(BASE_DIR, 'fais_index_all_transcripts/index.pkl')

    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv('OPENAI_API_KEY'))
    db = FAISS.load_local("/tmp/fais_index_all_transcripts", embeddings, allow_dangerous_deserialization=True)

    faiss_retriever = db.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 7,
            "fetch_k": 10,
            "lambda_mult": 0.7,
            "include_metadata": True,
        },
    )

    hybrid_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever], weights=[0.5, 0.5]
    )

def multi_step_rag(query, retriever, llm):
    docs = retriever.invoke(query)

    analysis_prompt = PromptTemplate.from_template(
        """You are an expert analyst of Lex Fridman's podcasts. 
        You are able to carry a natural conversation with the user.
        Quote the content using quotation marks if necessary.
        Always include the podcast number and speaker name (e.g. Lex Fridman) when referencing information from a specific episode. 
        Do not include speaker A, speaker B, speaker 0, etc. only return real names.
        Provide a concise answer, followed by the relevant episode reference(s).
        Ask the user if they want more information.
        Query: {query}

        Retrieved Information: {docs}

        Provide a concise answer and ask if the user wants more information:"""
    )

    analysis_chain = (
        {"query": RunnablePassthrough(), "docs": lambda x: docs}
        | analysis_prompt
        | llm
        | StrOutputParser()
    )

    return analysis_chain.invoke(query)

# Serve React App
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    if path != "" and os.path.exists(app.static_folder + '/' + path):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, 'index.html')

@app.route("/chat", methods=["POST", "OPTIONS"])
def chat():
    global llm, db, bm25_retriever, hybrid_retriever

    if request.method == "OPTIONS":
        return jsonify({"message": "CORS preflight request handled"}), 200

    try:
        # Initialize data if it hasn't been done yet
        # if db is None or bm25_retriever is None or hybrid_retriever is None:
        #     initialize_or_load_data()

        data = request.json
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400

        user_input = data.get("message")
        if not user_input:
            return jsonify({"error": "No message provided"}), 400

        session_id = data.get("session_id", "default")

        if session_id not in chat_histories:
            chat_histories[session_id] = ChatMessageHistory()

        chat_history = chat_histories[session_id]
    except Exception as e:
        logging.error(f"Error in chat endpoint: {str(e)}", exc_info=True)
        return jsonify({"error": "An unexpected error occurred. Please try again later."}), 500

    # Try different models in case of rate limiting
    for model in models:
        try:
            if llm is None or llm.model_name != model:
                llm = ChatOpenAI(
                    model_name=model,
                    openai_api_key=os.getenv('OPENAI_API_KEY'),
                )

            # Use the multi_step_rag function to generate a response
            response = multi_step_rag(user_input, hybrid_retriever, llm)

            # Update chat history
            chat_history.add_user_message(user_input)
            chat_history.add_ai_message(response)

            return jsonify({"response": response, "model_used": model})

        except RateLimitError as e:
            print(f"Rate limit reached for {model}. Trying next model...")
            if model == models[-1]:
                return jsonify({"error": "All models are currently rate limited. Please try again later."}), 429
            time.sleep(1)
        except Exception as e:
            print(e)
            return jsonify({"error": str(e)}), 500

    return jsonify({"error": "Unexpected error occurred"}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Not Found"}), 404

# @https_fn.on_request()
# def chat_function(request: https_fn.Request) -> https_fn.Response:
#     with app.request_context(request.environ):
#         return app.full_dispatch_request()

if __name__ == "__main__":
    try:
        initialize_or_load_data()
        port = int(os.environ.get("PORT", 8080))
        app.run(debug=os.environ.get('FLASK_ENV') == 'development', host="0.0.0.0", port=port)
    except Exception as e:
        logging.error(f"Error starting the application: {str(e)}", exc_info=True)
        raise