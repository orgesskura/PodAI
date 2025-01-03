import os
from flask import Flask, request, jsonify, send_from_directory
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
from google.cloud import storage
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__, static_folder='lex-chat/build/')
CORS(app, resources={r"/*": {"origins": "*"}})

logging.basicConfig(level=logging.INFO)

# Initialize global variables
models = ["gpt-4o-mini", "gpt-3.5-turbo-16k", "gpt-3.5-turbo"]
llm = None
db = None
bm25_retriever = None
hybrid_retriever = None
chat_histories = {}

# Google Cloud Storage setup
storage_client = storage.Client()
bucket_name = 'podai'
bucket = storage_client.bucket(bucket_name)

def initialize_or_load_data():
    global db, bm25_retriever, hybrid_retriever

    blob = bucket.blob('all_documents.pkl')

    print("Loading existing data...")
    pickle_data = blob.download_as_bytes()
    all_documents = pickle.loads(pickle_data)
    bm25_retriever = BM25Retriever.from_documents(all_documents)

    # Create the directory if it doesn't exist
    os.makedirs('/tmp/faiss_index_all_transcripts', exist_ok=True)

    blob = bucket.blob('faiss_index_all_transcripts/index.pkl')
    blob.download_to_filename('/tmp/faiss_index_all_transcripts/index.pkl')

    blob = bucket.blob('faiss_index_all_transcripts/index.faiss')
    blob.download_to_filename('/tmp/faiss_index_all_transcripts/index.faiss')

    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv('OPENAI_API_KEY'))
    db = FAISS.load_local("/tmp/faiss_index_all_transcripts", embeddings, allow_dangerous_deserialization=True)

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
        if db is None or bm25_retriever is None or hybrid_retriever is None:
            initialize_or_load_data()

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
            return jsonify({"error": str(e)}), 500

    return jsonify({"error": "Unexpected error occurred"}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Not Found"}), 404

if __name__ == "__main__":
    try:
        initialize_or_load_data()
        port = int(os.environ.get("PORT", 8080))
        app.run(debug=os.environ.get('FLASK_ENV') == 'development', host="0.0.0.0", port=port)
    except Exception as e:
        logging.error(f"Error starting the application: {str(e)}", exc_info=True)
        raise