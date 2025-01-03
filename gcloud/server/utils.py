from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain import PromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
import os

# Initialize global variables
llm = None
db = None
bm25_retriever = None
hybrid_retriever = None

def initialize_retrievers():
    global db, bm25_retriever, hybrid_retriever
    
    # Implement the initialization of retrievers here
    # This should include loading the FAISS index and creating the hybrid retriever
    pass

def get_chat_response(query):
    global llm, hybrid_retriever

    if llm is None:
        llm = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            openai_api_key=os.getenv('OPENAI_API_KEY'),
        )

    if hybrid_retriever is None:
        initialize_retrievers()

    docs = hybrid_retriever.invoke(query)

    analysis_prompt = PromptTemplate.from_template(
        """You are an expert analyst of Lex Fridman's podcasts. 
        You are able to carry a natural conversation with the user, but also able to provide relevant information.
        Quote the content using quotation marks if necessary.
        Always include the podcast number and speaker name when referencing information from a specific episode. 
        Provide a concise answer, followed by the relevant episode reference(s).

        Query: {query}

        Retrieved Information: {docs}

        Provide a concise answer:"""
    )

    analysis_chain = (
        {"query": RunnablePassthrough(), "docs": lambda x: docs}
        | analysis_prompt
        | llm
        | StrOutputParser()
    )

    return analysis_chain.invoke(query)
