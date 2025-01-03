import os
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
from langchain.schema.runnable import RunnablePassthrough

api_key = "sk-*"
llm = ChatOpenAI(model_name="gpt-4o-mini", openai_api_key=api_key)
llm_cheap = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=api_key)


def extract_podcast_number(llm, file_name):
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an assistant that extracts podcast numbers from file names.",
            ),
            (
                "human",
                "Extract the podcast number from the following file name: {file_name}"
                "Return the answer as just a number. For example if podcast number is 325, just return 325.",
            ),
        ]
    )

    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"file_name": file_name})


def extract_interviewee_names(llm, file_name):
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an assistant that extracts interviewee names from podcast file names.",
            ),
            (
                "human",
                "Extract the interviewee names from the following file name: {file_name}"
                "Return a comma-separated list of names. If you can't determine the names, return 'Unknown Interviewee'.",
            ),
        ]
    )

    chain = prompt | llm | StrOutputParser()
    names = chain.invoke({"file_name": file_name})
    return [name.strip() for name in names.split(",")]


def preprocess_transcript(transcript, interviewees):
    processed = transcript
    speaker_map = {}

    for i, name in enumerate(interviewees):
        speaker_map[f"Speaker {i}:"] = f"{name}:"

    for i, name in enumerate(interviewees):
        speaker_map[f"Speaker {chr(65+i)}:"] = f"{name}:"

    for label, name in speaker_map.items():
        processed = processed.replace(label, name)

    return processed


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
            podcast_number = extract_podcast_number(llm_cheap, file_name)

            interviewees = extract_interviewee_names(llm_cheap, file_name)
            interviewees = ["Lex Fridman"] + interviewees

            # Extract episode name from file name
            episode_name = file_name.replace(".txt", "").replace("_", " ").title()

            print(
                f"Processing {file_name}, Podcast number: {podcast_number}, Episode: {episode_name}, Interviewees: {', '.join(interviewees)}"
            )

            with open(file_path, "r", encoding="utf-8") as file:
                transcript = file.read()

            processed_transcript = preprocess_transcript(transcript, interviewees)

            chunks = text_splitter.split_text(processed_transcript)
            documents = [
                Document(
                    page_content=chunk,
                    metadata={
                        "source": file_name,
                        "podcast_number": podcast_number,
                        "episode_name": episode_name,
                        "chunk_index": i,
                        "interviewees": interviewees,
                    },
                )
                for i, chunk in enumerate(chunks)
            ]
            all_documents.extend(documents)

    db = FAISS.from_documents(all_documents, embeddings)
    db.save_local("faiss_index_all_transcripts")

    bm25_retriever = BM25Retriever.from_documents(all_documents)

    with open("all_documents.pkl", "wb") as f:
        pickle.dump(all_documents, f)

    return db, bm25_retriever, all_documents


def load_or_create_db(directory_path):
    if os.path.exists("faiss_index_all_transcripts") and os.path.exists(
        "all_documents.pkl"
    ):
        print("Loading existing FAISS index and documents...")
        embeddings = OpenAIEmbeddings(openai_api_key=api_key)
        db = FAISS.load_local(
            "faiss_index_all_transcripts",
            embeddings,
            allow_dangerous_deserialization=True,
        )

        # Load all_documents
        with open("all_documents.pkl", "rb") as f:
            all_documents = pickle.load(f)

        # Recreate BM25 retriever
        bm25_retriever = BM25Retriever.from_documents(all_documents)
        return db, bm25_retriever, all_documents
    else:
        print("Creating new FAISS and BM25 indexes...")
        return process_transcripts(directory_path)


def initialize_db(directory_path):
    return load_or_create_db(directory_path)


# Set up retriever and chains
# db, bm25_retriever, all_documents = load_or_create_db("cleaned_diarized")

# faiss_retriever = db.as_retriever(
#     search_type="mmr",
#     search_kwargs={
#         "k": 10,
#         "fetch_k": 15,
#         "lambda_mult": 0.7,
#         "include_metadata": True
#     }
# )

# hybrid_retriever = EnsembleRetriever(
#     retrievers=[bm25_retriever, faiss_retriever],
#     weights=[0.5, 0.5]
# )

# contextualize_q_system_prompt = (
#     "Given a chat history and the latest user question "
#     "which might reference context in the chat history, "
#     "formulate a standalone question which can be understood "
#     "without the chat history. Do NOT answer the question, "
#     "just reformulate it if needed and otherwise return it as is."
# )
# contextualize_q_prompt = ChatPromptTemplate.from_messages([
#     ("system", contextualize_q_system_prompt),
#     MessagesPlaceholder("chat_history"),
#     ("human", "{input}"),
# ])

# system_prompt = (
#     "You are an assistant for question-answering tasks about Lex Fridman's podcasts. "
#     "Use the following pieces of retrieved context to answer the question. "
#     "The transcripts are conversations between Lex Fridman and one or more interviewees. "
#     "If you see specific names followed by colons (e.g., 'Lex Fridman:' or '[Interviewee Name]:'), "
#     "these indicate who is speaking. "
#     "If you see 'Speaker' followed by a number or letter (e.g., 'Speaker 0:', 'Speaker A:'), "
#     "these are generic labels for different speakers. In this case, mention that you're "
#     "quoting an unnamed speaker. "
#     "The metadata for each chunk includes the episode name, podcast number, and list of interviewees, which you should reference in your answer. "
#     "Always include the episode name and podcast number when referencing information from a specific episode. "
#     "If you don't know the answer or can't find relevant information, say that you don't know. "
#     "Aim to provide a concise answer in 3-5 sentences, followed by the relevant episode reference(s)."
#     "\n\n"
#     "{context}"
# )
# qa_prompt = ChatPromptTemplate.from_messages([
#     ("system", system_prompt),
#     MessagesPlaceholder("chat_history"),
#     ("human", "{input}"),
# ])
# question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)


# # Manage chat history
# store = {}

# def get_session_history(session_id: str) -> BaseChatMessageHistory:
#     if session_id not in store:
#         store[session_id] = ChatMessageHistory()
#     return store[session_id]


def multi_step_rag(query, retriever, llm):
    docs = retriever.invoke(query)

    analysis_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an expert analyst of Lex Fridman's podcasts. "
                "You are able to carry a natural conversation with the user, but also able to provide relevant information "
                "Quote the content using quotation marks if necessary"
                "Always include the podcast number and speaker name when referencing information from a specific episode. "
                "Provide a concise answer, followed by the relevant episode reference(s).",
            ),
            (
                "human",
                "Query: {query}\n\nRetrieved Information: {docs}\n\nProvide a concise answer:",
            ),
        ]
    )

    analysis_chain = (
        {"query": RunnablePassthrough(), "docs": lambda x: docs}
        | analysis_prompt
        | llm
        | StrOutputParser()
    )

    return analysis_chain.invoke(query)


# def chat():
#     session_id = "user1"
#     print("Welcome to the Lex Fridman Podcast RAG chatbot with Hybrid Retrieval!")
#     print("Type 'exit' to end the conversation.")

#     while True:
#         user_input = input("\nYou: ")
#         if user_input.lower() == 'exit':
#             print("Goodbye!")
#             break

#         response = multi_step_rag(user_input, hybrid_retriever, llm)

#         print(f"\nAssistant: {response}")

# if __name__ == "__main__":
#     chat()
