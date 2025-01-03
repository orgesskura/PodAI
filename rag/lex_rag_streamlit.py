import streamlit as st
import os
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import StrOutputParser
from langchain.retrievers import BM25Retriever, EnsembleRetriever
import pickle
from langchain.callbacks import StreamlitCallbackHandler
from langchain.schema.runnable import RunnablePassthrough
from langchain.memory import ConversationBufferMemory

# Assuming you have already set up your OpenAI API key in your environment variables
api_key = "sk-*"
llm = ChatOpenAI(model_name="gpt-4o-mini", openai_api_key=api_key)
llm_cheap = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=api_key)


@st.cache_resource
def load_retrievers():
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    db = FAISS.load_local(
        "faiss_index_all_transcripts", embeddings, allow_dangerous_deserialization=True
    )

    with open("all_documents.pkl", "rb") as f:
        all_documents = pickle.load(f)

    bm25_retriever = BM25Retriever.from_documents(all_documents)

    faiss_retriever = db.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 10,
            "fetch_k": 15,
            "lambda_mult": 0.7,
            "include_metadata": True,
        },
    )

    hybrid_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever], weights=[0.5, 0.5]
    )

    return hybrid_retriever


def multi_step_rag(query, retriever, llm, chat_history):
    docs = retriever.invoke(query)
    
    analysis_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert analyst of Lex Fridman's podcasts. "
                   "You are able to carry a natural conversation with the user, but also able to provide relevant information. "
                   "Quote the content using quotation marks if necessary. "
                   "Always include the podcast number and speaker name when referencing information from a specific episode. "
                   "Provide an answer, followed by the relevant episode reference(s). "
                   "Use the chat history to maintain context and keep a continuation of the conversation"
                    "and you don't strictly need to provide a reference to the episode for a user's follow up question."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "Query: {query}\n\nRetrieved Information: {docs}\n\nProvide an answer:"),
    ])
    
    analysis_chain = (
        {
            "query": RunnablePassthrough(),
            "docs": lambda x: docs,
            "chat_history": lambda x: chat_history,
        }
        | analysis_prompt
        | llm
        | StrOutputParser()
    )

    return analysis_chain


st.set_page_config(page_title="Lex Fridman Podcast AI", page_icon="🎙️", layout="wide")

# Sidebar content
st.sidebar.header("About")
st.sidebar.markdown(
    """
This AI-powered chatbot uses Retrieval-Augmented Generation (RAG) to provide insights from Lex Fridman's podcasts.

- 🎙️ Explore episodes
- 👥 Learn about guests
- 🧠 Discover new ideas

Start by asking a question about any topic discussed in the podcasts!
"""
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Popular Topics")
topics = [
    "Elon Musk advice to young",
    "Jeff Bezos on productivity",
    "Yann Lecun on LLM",
    "Sam Altman on AI Safety",
    "Yuval Noah Harari meditation",
]
for topic in topics:
    if st.sidebar.button(topic):
        st.session_state.topic_selected = f"Tell me about {topic.lower()} discussions"

# Main content
st.title("🎙️ Lex Fridman Podcast AI")
st.subheader("Explore the world of ideas with Lex and AI")
st.markdown(
    "Created with 🔥 by Kit and Orges &nbsp;&nbsp;&nbsp; | &nbsp;&nbsp;&nbsp;&nbsp; Check Out [Lex Fridman's Website](https://lexfridman.com/) for more info"
)
st.markdown("---")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize conversation memory
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(return_messages=True)

# Load retrievers
retriever = load_retrievers()

# Chat interface
st.markdown("### Chat with the AI")
chat_container = st.container()

with chat_container:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# User input
prompt = st.chat_input("Ask about Lex Fridman's podcasts...", key="user_input")

# Check if a topic was selected from the sidebar
if "topic_selected" in st.session_state:
    prompt = st.session_state.topic_selected
    del st.session_state.topic_selected

if prompt:
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        response_placeholder = st.empty()

    with st.spinner("🧠 Thinking..."):
        chat_history = st.session_state.memory.chat_memory.messages
        chain = multi_step_rag(prompt, retriever, llm, chat_history)
        response = chain.invoke(prompt)

    response_placeholder.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})

    # Update conversation memory
    st.session_state.memory.chat_memory.add_user_message(prompt)
    st.session_state.memory.chat_memory.add_ai_message(response)

# Clear chat button
if st.button("Clear Chat"):
    st.session_state.messages = []
    st.session_state.memory.clear()
    st.rerun()

