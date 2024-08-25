import modal
from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_community.chat_message_histories import ChatMessageHistory
import pickle
import os
import time
from openai import RateLimitError
from deploy_rag import multi_step_rag, initialize_db

# Define the path to your local directory
LOCAL_DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cleaned_diarized")

# Create a new volume
volume = modal.Volume.from_name("lex-fridman")

# Create the Modal app
app = modal.App("lex-fridman-podcast-rag")

image = modal.Image.debian_slim().pip_install(
    "flask",
    "flask-cors",
    "langchain",
    "langchain_openai",
    "langchain_community",
    "faiss-cpu",
    "openai"
)

@app.function(
    image=image,
    secret=modal.Secret.from_name("my-openai-secret"),
    volumes={"/data": volume},
    mounts=[modal.Mount.from_local_dir(LOCAL_DATA_PATH, remote_path="/app/local")],
    cpu=1,
    memory=4096,
)
@modal.wsgi_app()
def web_app():
    flask_app = Flask(__name__)
    CORS(flask_app, resources={r"/chat": {"origins": "*"}})

    # Initialize your models, retrievers, and chains here
    models = ["gpt-4o-mini","gpt-4-1106-preview", "gpt-3.5-turbo-16k", "gpt-3.5-turbo"]
    llm = None
    db = None
    bm25_retriever = None
    hybrid_retriever = None

    def initialize_or_load_data():
        nonlocal db, bm25_retriever, hybrid_retriever
        if os.path.exists("/data/initialized_data.pkl"):
            print("Loading existing data...")
            with open("/data/initialized_data.pkl", "rb") as f:
                db, bm25_retriever, all_documents = pickle.load(f)
        else:
            print("Initializing database...")
            db, bm25_retriever, all_documents = initialize_db("/app/local/cleaned_diarized")
            with open("/data/initialized_data.pkl", "wb") as f:
                pickle.dump((db, bm25_retriever, all_documents), f)

        faiss_retriever = db.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 10,
                "fetch_k": 15,
                "lambda_mult": 0.7,
                "include_metadata": True
            }
        )


        hybrid_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, faiss_retriever],
            weights=[0.5, 0.5]
        )

    # Initialize chat histories
    chat_histories = {}

    @flask_app.route('/')
    def home():
        return "Welcome to the Lex Fridman Podcast RAG API!"

    @flask_app.route('/chat', methods=['POST'])
    def chat():
        nonlocal llm, db, bm25_retriever, hybrid_retriever
        
        # Initialize data if it hasn't been done yet
        if db is None or bm25_retriever is None or hybrid_retriever is None:
            initialize_or_load_data()
        
        data = request.json
        user_input = data['message']
        session_id = data.get('session_id', 'default')
        
        if session_id not in chat_histories:
            chat_histories[session_id] = ChatMessageHistory()
        
        chat_history = chat_histories[session_id]
        
        # Try different models in case of rate limiting
        for model in models:
            try:
                if llm is None or llm.model_name != model:
                    llm = ChatOpenAI(model_name=model, openai_api_key=os.environ["OPENAI_API_KEY"])
                
                # Use the multi_step_rag function to generate a response
                response = multi_step_rag(user_input, hybrid_retriever, llm)
                
                # Update chat history
                chat_history.add_user_message(user_input)
                chat_history.add_ai_message(response)
                
                return jsonify({'response': response, 'model_used': model})
            
            except RateLimitError as e:
                print(f"Rate limit reached for {model}. Trying next model...")
                if model == models[-1]:
                    return jsonify({'error': 'All models are currently rate limited. Please try again later.'}), 429
                time.sleep(1)  # Wait a bit before trying the next model
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        return jsonify({'error': 'Unexpected error occurred'}), 500

    return flask_app

@app.local_entrypoint()
def main():
    web_app.serve()

if __name__ == "__main__":
    modal.runner.deploy_app(app)