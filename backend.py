import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.schema import StrOutputParser
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.vectorstores import FAISS
from langchain.retrievers import BM25Retriever, EnsembleRetriever
import pickle
from deploy_rag import multi_step_rag, process_transcripts

app = Flask(__name__)
CORS(app, resources={r"/chat": {"origins": "*"}})

# Initialize your models, retrievers, and chains here (reuse the code from the original script)
api_key = "sk-mDDbQXk0nqPVUKjReeOIgiBoIyyjANlaXcERYJqyM6T3BlbkFJ5V6MdFRVmWcxIG2n0JyM1woTAUDps1dWqv6uYvnKsA"
llm = ChatOpenAI(model_name="gpt-4-1106-preview", openai_api_key=api_key)
llm_cheap = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=api_key)

def load_or_create_db(directory_path):
    if os.path.exists("faiss_index_all_transcripts") and os.path.exists("all_documents.pkl"):
        print("Loading existing FAISS index and documents...")
        embeddings = OpenAIEmbeddings(openai_api_key=api_key)
        db = FAISS.load_local("faiss_index_all_transcripts", embeddings, allow_dangerous_deserialization=True)
        
        # Load all_documents
        with open("all_documents.pkl", "rb") as f:
            all_documents = pickle.load(f)
        
        # Recreate BM25 retriever
        bm25_retriever = BM25Retriever.from_documents(all_documents)
        return db, bm25_retriever, all_documents
    else:
        print("Creating new FAISS and BM25 indexes...")
        return process_transcripts(directory_path)

# Load or create the database and retrievers
db, bm25_retriever, all_documents = load_or_create_db("cleaned_diarized")

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

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_input = data['message']
    session_id = data.get('session_id', 'default')
    
    if session_id not in chat_histories:
        chat_histories[session_id] = ChatMessageHistory()
    
    chat_history = chat_histories[session_id]
    
    # Use the multi_step_rag function to generate a response
    response = multi_step_rag(user_input, hybrid_retriever, llm)
    
    # Update chat history
    chat_history.add_user_message(user_input)
    chat_history.add_ai_message(response)
    
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)