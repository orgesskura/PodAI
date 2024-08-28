import os
import re
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

api_key = "sk-mDDbQXk0nqPVUKjReeOIgiBoIyyjANlaXcERYJqyM6T3BlbkFJ5V6MdFRVmWcxIG2n0JyM1woTAUDps1dWqv6uYvnKsA"
llm = ChatOpenAI(model_name="gpt-4o-mini", openai_api_key=api_key)
llm_cheap = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=api_key)

def clean_episode_name(file_name):
    name = file_name.replace("_transcript.txt", "").replace("_", " ")
    return ' '.join(name.split()).title()

def extract_speakers_and_content_llm(llm, transcript):
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an assistant that identifies speakers and their content from a transcript. "
                   "Identify real names of speakers when possible, otherwise use the provided speaker labels."),
        ("human", "Analyze the following transcript excerpt and identify the speakers and their content: \n\n{transcript}\n\n"
                  "Return a JSON object where keys are speaker names and values are lists of their spoken content. "
                  "Example format: {'John Doe': ['Content 1', 'Content 2'], 'Jane Smith': ['Content 3', 'Content 4']}")
    ])
    
    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({"transcript": transcript[:4000]})  # Use first 4000 characters for efficiency
    try:
        return json.loads(result)
    except json.JSONDecodeError:
        print(f"Error parsing speaker content: {result}")
        return {}

def process_transcripts(directory_path):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2500,
        chunk_overlap=250,
        length_function=len,
        separators=["\n\n", "\n", ". ", " "],
    )
    all_documents = []
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)

    for file_name in os.listdir(directory_path):
        if file_name.endswith(".txt"):
            file_path = os.path.join(directory_path, file_name)
            
            episode_name = clean_episode_name(file_name)

            with open(file_path, "r", encoding="utf-8") as file:
                transcript = file.read()

            speaker_content = extract_speakers_and_content_llm(llm_cheap, transcript)
            speakers = list(speaker_content.keys())

            print(
                f"Processing {file_name}\n"
                f"Episode: {episode_name}\n"
                f"Speakers: {', '.join(speakers)}\n"
            )

            for speaker, content in speaker_content.items():
                full_content = ' '.join(content)
                chunks = text_splitter.split_text(full_content)
                documents = [
                    Document(
                        page_content=chunk,
                        metadata={
                            "source": file_name,
                            "episode_name": episode_name,
                            "speaker": speaker,
                            "chunk_index": i,
                        },
                    )
                    for i, chunk in enumerate(chunks)
                ]
                all_documents.extend(documents)

    db = FAISS.from_documents(all_documents, embeddings)
    db.save_local("faiss_index_yctranscripts")

    bm25_retriever = BM25Retriever.from_documents(all_documents)

    with open("yc_documents.pkl", "wb") as f:
        pickle.dump(all_documents, f)

    return db, bm25_retriever, all_documents


def load_or_create_db(directory_path):
    if os.path.exists("faiss_index_yctranscripts") and os.path.exists(
        "yc_documents.pkl"
    ):
        print("Loading existing FAISS index and documents...")
        embeddings = OpenAIEmbeddings(openai_api_key=api_key)
        db = FAISS.load_local(
            "faiss_index_yctranscripts",
            embeddings,
            allow_dangerous_deserialization=True,
        )

        # Load all_documents
        with open("yc_documents.pkl", "rb") as f:
            all_documents = pickle.load(f)

        # Recreate BM25 retriever
        bm25_retriever = BM25Retriever.from_documents(all_documents)
        return db, bm25_retriever, all_documents
    else:
        print("Creating new FAISS and BM25 indexes...")
        return process_transcripts(directory_path)


# Set up retriever and chains
db, bm25_retriever, all_documents = load_or_create_db("cleaned_ycdiarized")

faiss_retriever = db.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 10,
        "fetch_k": 15,
        "lambda_mult": 0.6,
        "include_metadata": True,
    },
)

hybrid_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, faiss_retriever], weights=[0.5, 0.5]
)

contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

system_prompt = (
    "You are an assistant for question-answering tasks about YC's videos. "
    "Use the following pieces of retrieved context to answer the question. "
    "The transcripts are conversations between YC partners or guest speakers "
    "If you see specific names followed by colons (e.g., 'Michael Siebel:' or '[Interviewee Name]:'), "
    "these indicate who is speaking. "
    "these are generic labels for different speakers. In this case, mention that you're "
    "quoting an unnamed speaker. "
    "The metadata for each chunk includes the episode name and list of speakers, which you should reference in your answer. "
    "If you don't know the answer or can't find relevant information, say that you don't know. "
    "Aim to provide a concise answer in 3-5 sentences, followed by the relevant episode reference(s)."
    "\n\n"
    "{context}"
)
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)


# Manage chat history
store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


def multi_step_rag(query, retriever, llm, chat_history):
    docs = retriever.invoke(query)
    
    analysis_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert analyst of YC's podcasts. Analyze the retrieved information and synthesize an answer to the query. "
                   "Include the episode title and speaker name when relevant, especially for questions about who said something. "
                   "You are allowed to provide answers that are not directly from the retrieved information, but make sure they are relevant. "
                   "Provide a concise answer."),
        ("human", "Chat History: {chat_history}\n\nQuery: {query}\n\nRetrieved Information: {docs}\n\nProvide a concise analysis and answer:")
    ])

    analysis_chain = analysis_prompt | llm | StrOutputParser()
    return analysis_chain.invoke({"chat_history": chat_history, "query": query, "docs": docs})

def chat():
    db, bm25_retriever, _ = load_or_create_db("cleaned_ycdiarized")
    hybrid_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, db.as_retriever()],
        weights=[0.5, 0.5]
    )
    
    chat_history = []
    print("Welcome to the YC chatbot with Hybrid Retrieval!")
    print("Type 'exit' to end the conversation.")

    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break

        response = multi_step_rag(user_input, hybrid_retriever, llm, chat_history)
        print(f"\nAssistant: {response}")

        chat_history.append(("Human", user_input))
        chat_history.append(("AI", response))

if __name__ == "__main__":
    chat()