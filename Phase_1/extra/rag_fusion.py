from langchain_community.vectorstores import DeepLake
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import DeepLake
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_community.document_loaders import TextLoader
import os
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.prompts.chat import ChatPromptTemplate


embeddings = HuggingFaceEmbeddings()


## Load the documents

def load_documents(root_dir=None, file_path=None):
    docs = []
    if file_path:
        loader = PyPDFLoader(file_path)
        documents = loader.load_and_split()
        return documents
    elif root_dir:
        for dirpath, dirnames, filenames in os.walk(root_dir):
            for file in filenames:
                # Corrected the use of endswith by passing a tuple
                if file.endswith((".md", ".mdx", ".pdf", ".py", ".json", ".yaml")) and "*venv/" not in dirpath:
                    try:    
                        loader = TextLoader(os.path.join(dirpath, file), encoding="utf-8")
                        docs.extend(loader.load_and_split())
                        print(f"Loaded {file}")
                        return documents
                    except Exception:
                        pass
    print(f"{len(docs)} documents loaded.")



## Split the documents

def split_documents(documents):
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)
    print(f"{len(docs)}")
    return docs



## Create the vector database


def create_vector_db(docs):
    db = DeepLake.from_documents(docs, embeddings, dataset_path="C:\\Users\\wesla\\open-interpreter\\vector_db", overwrite=True)
    return db


## Chat with the vector database

def chat(text):
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant that generates multiple search queries based on a single input query."),
        ("user", "Generate multiple search queries related to: {original_query}"),
        ("user", "OUTPUT (4 queries):")
    ])

    
    # Assuming 'db' is a pre-configured DeepLake instance, as it's not defined in the provided code
    db = DeepLake(embeddings=embeddings)
    retriever = db.as_retriever()
    retriever.search_kwargs["distance_metric"] = "cos"
    retriever.search_kwargs["fetch_k"] = 20
    retriever.search_kwargs["k"] = 20


    llm = ChatOpenAI(base_url="http://localhost:1234/v1", openai_api_key="lm-studio", temperature=0)
    qa = ConversationalRetrievalChain.from_llm(llm, retriever=retriever)
    
    message = [f"Generate multiple search queries related to: '{text}'"]
    chat_history = []
    qa_dict = {}

    result = qa({"message": message, "chat_history": chat_history})
    chat_history.append((message, result["answer"]))
    qa_dict[message] = result["answer"]

    return result["answer"]

# Example usage
response = chat("how to install python")
print(response)


def rag_fusion(text):
    

    result = qa({"message": message, "chat_history": chat_history})
    