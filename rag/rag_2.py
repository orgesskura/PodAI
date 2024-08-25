import bs4
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain.docstore.document import Document
from langchain.vectorstores import FAISS
from langchain.schema import StrOutputParser
from langchain.document_loaders import TextLoader

llm = ChatOpenAI(
        model_name="gpt-4o-mini",
        openai_api_key="sk-mDDbQXk0nqPVUKjReeOIgiBoIyyjANlaXcERYJqyM6T3BlbkFJ5V6MdFRVmWcxIG2n0JyM1woTAUDps1dWqv6uYvnKsA"
    )


def extract_podcast_number(llm, file_name):
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an assistant that extracts podcast numbers from file names."),
        ("human", "Extract the podcast number from the following file name: {file_name}"
                  "Return the answer as just a number. For example if podcast number is 325, just return 325.")
    ])
    
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"file_name": file_name})

def save_all_embeddings(file_names, api_key):
    llm_cheap = ChatOpenAI(
        model_name="gpt-3.5-turbo",  # Using a smaller model for this task
        openai_api_key=api_key
    )
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    all_texts = []
    embeddings = OpenAIEmbeddings()
    for file_name in file_names:
        transcript_file = f"{file_name}.txt"
        podcast_number = extract_podcast_number(llm_cheap, file_name)

        print(f"Extracted podcast number: {podcast_number}!")
        with open(transcript_file, 'r', encoding='utf-8') as file:
            transcript = file.read()

        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = text_splitter.split_text(transcript)
        documents = [
            Document(
                page_content=chunk,
                metadata={
                    "source": file_name,
                    "podcast_number": podcast_number,
                    "chunk_index": i
                }
            ) for i, chunk in enumerate(texts)
        ]
    
        all_texts.extend(documents)
    db = FAISS.from_documents(all_texts, embeddings)
    db.save_local("faiss_index_multi")
    return db

db = save_all_embeddings(["Michael Levin: Biology, Life, Aliens, Evolution, Embryogenesis & Xenobots | Lex Fridman Podcast #325"], "sk-mDDbQXk0nqPVUKjReeOIgiBoIyyjANlaXcERYJqyM6T3BlbkFJ5V6MdFRVmWcxIG2n0JyM1woTAUDps1dWqv6uYvnKsA")
retriever = db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5, "include_metadata": True}
    )
### Contextualize question ###
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
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)


### Answer question ###
system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
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

rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)


### Statefully manage chat history ###
store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

print(conversational_rag_chain.invoke(
    {"input": "Was Michael Levin on Lex Fridman podcast?"},
    config={
        "configurable": {"session_id": "abc123"}
    },  # constructs a key "abc123" in `store`.
)["answer"])

print(store)

print(conversational_rag_chain.invoke(
    {"input": "What podcast number was he on? Do you know who I am talking about?"},
    config={
        "configurable": {"session_id": "abc123"}
    },  # constructs a key "abc123" in `store`.
)["answer"])

