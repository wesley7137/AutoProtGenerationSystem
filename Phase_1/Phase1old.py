
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


import re
import logging
from openai import OpenAI
from langchain_community.vectorstores import DeepLake
from langchain_huggingface import HuggingFaceEmbeddings
import tiktoken
import json
from langchain_ollama import ChatOllama


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
llm_model = ChatOllama(model="deepseek-coder-v2:16b-lite-instruct-q6_K ")

from langchain_community.vectorstores import DeepLake
from langchain_huggingface import HuggingFaceEmbeddings
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Comprehensivetechnical_instructionSystem:
    def __init__(self, dataset_path):
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        self.dataset_path = dataset_path
        self.db = self.load_db(self.dataset_path)

    def load_db(self, dataset_path):
        try:
            return DeepLake(dataset_path=dataset_path, embedding=self.embeddings, read_only=True)
        except Exception as e:
            logger.error(f"Error loading database from {dataset_path}: {str(e)}")
            return None

    def retrieve_from_db(self, query):
        if self.db is None:
            logger.error("Database is not loaded.")
            return []

        retriever = self.db.as_retriever()
        retriever.search_kwargs["distance_metric"] = "cos"
        retriever.search_kwargs["fetch_k"] = 2
        retriever.search_kwargs["k"] = 2
        
        try:
            db_context = retriever.invoke(query)
            logger.info(f"Retrieved {len(db_context)} results from the database")
            return db_context
        except Exception as e:
            logger.error(f"Error retrieving from database: {str(e)}")
            return []
        
        
    def parse_technical_instruction(self, response):
        pattern = r'<technical_instruction>(.*?)</technical_instruction>'
        match = re.search(pattern, response, re.DOTALL)
        if match:
            return match.group(1).strip()
        else:
            logger.warning("No technical instruction found in the response")
            return None
        
    def formulate_technical_instruction(self, query, db_context):
        system_message = f"""You are a specialized AI assistant tasked with generating detailed, technical descriptions for sequence generation AI models based on scientific research. Your goal is to extract relevant information from database research results and a user's prompt, then formulate a more technical and detailed description using scientific terms and methods.

        You will be provided both the database results and the user's prompt.
        

        Using the extracted information, formulate a detailed technical description for a sequence generation AI model. Your description should:
        
        Present your response in a SHORT, CONCISE, technical and scientific manner.
        Remember to focus on scientific accuracy and detail in your description. If there are any ambiguities or missing information in the database results or user prompt, state these limitations clearly in your response.
        
        Your response should be formatted in the following structure:
        Example Response 1: 
        <technical_instruction>
        Generate a protein that will upregulate production of ACD-POT1 heterodimer to enhance telomere elongation which increases telomerase processivity
        </technical_instruction>
        
        Example Response 2: 
        <technical_instruction>
        Design a small molecule inhibitor that selectively targets and promotes the degradation of DAF-2 (insulin-like growth factor 1 receptor) to mimic the lifespan extension observed in C. elegans
        </technical_instruction>

    
        Example Response3: 
        <technical_instruction>
        Develop a compound that activates DAF-16 (FOXO transcription factor) to upregulate the expression of stress resistance and longevity-associated genes
        </technical_instruction>

    
        Example Response 4: 
        <technical_instruction>
        Create a dual-action molecule that simultaneously inhibits PI3K (phosphatidylinositol 3-kinase) and AKT kinase to attenuate mTOR signaling and enhance autophagy
        </technical_instruction>
        
        
        User Prompt: {query}
        
        Database Results: {db_context}\n\n\n\n
        
        **IMPORTANT: ONLY RESPOND IN THE FORMAT AS THE EXAMPLES SHOWN OR A SMALL KITTEN WILL BE TORTURED AND DIE. Your response should be formatted as <technical_instruction> and </technical_instruction> and should not contain any other text or formatting. IT SHOULD BE NO LONGER THAN 50 WORDS**
        """
        
        try:
            response = llm_model.invoke(system_message)
            logger.info("Received response from LLM")
            instruction = response.content.strip()
            
            parsed_instruction = self.parse_technical_instruction(instruction)
            if parsed_instruction:
                return {
                    "user_input": query,
                    "technical_instruction": parsed_instruction
                }
            else:
                logger.error("Failed to parse technical instruction from LLM response")
                return {
                    "user_input": query,
                    "technical_instruction": "Failed to generate a valid technical instruction."
                }
        except Exception as e:
            logger.error(f"Error in formulating technical instruction: {str(e)}")
            return {
                "user_input": query,
                "technical_instruction": "An error occurred while generating the technical instruction."
            }
            
            
    def run_pipeline(self, query):
        try:
            db_context = self.retrieve_from_db(query)
            technical_instruction = self.formulate_technical_instruction(query, db_context)
            return technical_instruction
        except Exception as e:
            logger.error(f"\nAn error occurred during the pipeline execution: {str(e)}")
            return {
                "user_input": query,
                "technical_instruction": "An error occurred while generating the technical_instruction. Please try again later."
            }
            
            
            
# Example usage
if __name__ == "__main__":
    system = Comprehensivetechnical_instructionSystem(dataset_path=r"C:\Users\wes\AutoProtGenerationSystem\Phase_1\technical_description_molecular_database")
    print("Dataset path:", system.dataset_path)
    query = "novel cellular regeneration and cell longevity to reverse aging"
    technical_instruction = system.run_pipeline(query)
    print(f"\nGenerated technical_instruction:\n{json.dumps(technical_instruction, indent=2)}")