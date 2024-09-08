


import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import logging

import os
import json
import csv
from typing import List, Dict
import logging

from langchain_community.vectorstores import DeepLake
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain_text_splitters import RecursiveJsonSplitter  # Import RecursiveJsonSplitter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)







class VectorStoreUploader:
    def __init__(self, dataset_path: str):
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        self.dataset_path = dataset_path
        self.vector_store = self.load_or_create_db()
        self.json_splitter = RecursiveJsonSplitter(max_chunk_size=2000)  # Use RecursiveJsonSplitter with desired chunk size

    def load_or_create_db(self):
        try:
            return DeepLake(dataset_path=self.dataset_path, embedding=self.embeddings)
        except Exception:
            return DeepLake.from_documents([], embedding=self.embeddings, dataset_path=self.dataset_path)

    def read_csv(self, file_path: str) -> List[Dict]:
        data = []
        with open(file_path, 'r', encoding='utf-8') as file:
            csv_reader = csv.DictReader(file)
            for row in csv_reader:
                # Extract the necessary fields
                content = row['simple_description']
                function = row['function']
                keywords = row['keywords'].strip("[]").replace("'", "").split(", ")
                gene_names = row['gene_names'].strip("[]").replace("'", "").split(", ")

                # Create JSON-like dictionary structure
                json_entry = {
                    "content": content,
                    "metadata": {
                        "function": function,
                        "keywords": keywords,
                        "gene_names": gene_names
                    }
                }
                data.append(json_entry)
        return data

    def read_json(self, file_path: str) -> List[Dict]:
        with open(file_path, 'r', encoding='utf-8') as file:
            return json.load(file)

    def upload_to_vectorstore(self, data: List[Dict]):
        documents = []
        for item in data:
            # Split JSON data into smaller chunks using RecursiveJsonSplitter
            json_chunks = self.json_splitter.split_json(json_data=item)

            for chunk in json_chunks:
                # Extract the content and metadata from each chunk
                chunk_content = chunk.get('content', '')
                # Ensure metadata is correctly assigned from the chunk, handling nested metadata correctly
                chunk_metadata = chunk.get('metadata', {})

                # Check that chunk content is not empty
                if chunk_content.strip():
                    documents.append(Document(page_content=chunk_content, metadata=chunk_metadata))

        # Upload documents with non-empty content and associated metadata
        valid_documents = [doc for doc in documents if doc.page_content.strip()]
        if valid_documents:
            self.vector_store.add_documents(valid_documents)
            logger.info(f"Uploaded {len(valid_documents)} document chunks to the vector store.")
        else:
            logger.warning("No valid documents to upload. Please check the content and structure of your data.")

    def process_file(self, file_path: str):
        _, file_extension = os.path.splitext(file_path)
        if file_extension.lower() == '.csv':
            data = self.read_csv(file_path)
        elif file_extension.lower() == '.json':
            data = self.read_json(file_path)
        else:
            raise ValueError("Unsupported file format. Please use CSV or JSON.")

        self.upload_to_vectorstore(data)

if __name__ == "__main__":
    dataset_path = r"technical_description_molecular_database"
    file_path = r"C:\Users\wes\AutoProtGenerationSystem\Phase_1\technical_instruction_dataset_output_data.json"  # or .json file path

    uploader = VectorStoreUploader(dataset_path)
    uploader.process_file(file_path)
    print(f"Data from {file_path} has been uploaded to the vector store at {dataset_path}")
