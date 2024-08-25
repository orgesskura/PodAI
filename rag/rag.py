import openai
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema import AIMessage, HumanMessage
from operator import itemgetter
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
from langchain_openai import OpenAIEmbeddings

hf_token = "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"
login(token=hf_token)

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

        loader = TextLoader(transcript_file)
        documents = loader.load()
        # 2. Split texts
        texts = text_splitter.split_documents(documents)
        for text in texts:
            text.metadata["podcast_name"] = file_name
            text.metadata["podcast_number"] = podcast_number
        all_texts.extend(texts)
    db = FAISS.from_documents(all_texts, embeddings)
    db.save_local("faiss_index_multi")
    return db

def load_embeddings():
    return FAISS.load_local("faiss_index_multi", OpenAIEmbeddings(), allow_dangerous_deserialization=True)

def rag():
    openai.api_key = "sk-mDDbQXk0nqPVUKjReeOIgiBoIyyjANlaXcERYJqyM6T3BlbkFJ5V6MdFRVmWcxIG2n0JyM1woTAUDps1dWqv6uYvnKsA"
    file_name = "Michael Levin: Biology, Life, Aliens, Evolution, Embryogenesis & Xenobots | Lex Fridman Podcast #325"

    db = save_all_embeddings([file_name],openai.api_key)
    # db = FAISS.load_local("faiss_index_multi", embeddings, allow_dangerous_deserialization=True)

    # Set up retriever
    retriever = db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5, "include_metadata": True}
    )

    # Set up OpenAI LLM
    llm = ChatOpenAI(
        model_name="gpt-4o-mini",
        openai_api_key=openai.api_key
    )
    # llm = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")

    # Create a prompt template
    template = ChatPromptTemplate.from_messages([
        ("system", "You are an assistant that provides answers to questions based on a given context. The context is about Lex Friedman podcast."
                   "Some of the labels have Speaker 0 ,Speaker 1 and so on. In these cases when answering substitute Speaker number with name of person in the podcast."
                   "Answer the question based on the context. If you can't answer the question, reply with 'I don't know'."),
        ("human", "Context: {context}\n\nQuestion: {question}\n\nPrevious conversation:\n{history}"),
        ("human", "{question}")
    ])

    # Set up the output parser
    parser = StrOutputParser()

    # Combine the retriever, prompt, model, and parser into a chain
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def format_metadata(docs):
        return "\n\n".join(str(doc.metadata) for doc in docs)

    chain = (
        {
            "context": itemgetter("question") | retriever | format_docs,
            "metadata": itemgetter("question") | retriever | format_metadata,
            "question": itemgetter("question"),
            "history": itemgetter("history")
        }
        | template
        | llm
        | parser
    )

    # 10. Function to handle conversation
    def chat():
        history = []
        while True:
            query = input("You: ")
            if query.lower() in ['exit', 'quit', 'bye']:
                break

            result = chain.invoke({"question": query, "history": format_history(history)})
            print("Assistant:", result)

            history.append(HumanMessage(content=query))
            history.append(AIMessage(content=result))

    def format_history(history):
        return "\n".join(f"{'Human' if isinstance(msg, HumanMessage) else 'Assistant'}: {msg.content}" for msg in history)

    # 11. Start the conversation
    chat()

if __name__ == "__main__":
    rag()