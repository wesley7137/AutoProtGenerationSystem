import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import os.path as osp
import time
import logging
from typing import List, Dict, Union
import openai
import requests
import backoff
from openai import OpenAI
from langchain_community.vectorstores import DeepLake
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

client = OpenAI(base_url="http://localhost:11434/v1", api_key="lm-studio")
llm_model = "hermes3:8b-llama3.1-q5_0"

def get_response_from_llm(msg, client, model, system_message, msg_history=None, temperature=0.5):
    if msg_history is None:
        msg_history = []

    new_msg_history = msg_history + [{"role": "user", "content": msg}]
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_message},
            *new_msg_history,
        ],
        temperature=temperature,
        max_tokens=512,
    )
    content = response.choices[0].message.content
    new_msg_history = new_msg_history + [{"role": "assistant", "content": content}]

    return content, new_msg_history

def extract_json_between_markers(llm_output):
    json_start_marker = "```json"
    json_end_marker = "```"

    start_index = llm_output.find(json_start_marker)
    if start_index != -1:
        start_index += len(json_start_marker)
        end_index = llm_output.find(json_end_marker, start_index)
    else:
        return None

    if end_index == -1:
        return None

    json_string = llm_output[start_index:end_index].strip()
    try:
        parsed_json = json.loads(json_string)
        return parsed_json
    except json.JSONDecodeError:
        return None

description_first_prompt = """
{task_description}

Generate the next impactful and creative technical description for molecule generation based on the user prompt.
Make sure the description is not overly specific to any particular molecule, and has wider applicability.

Respond in the following format:

THOUGHT:
<THOUGHT>

NEW DESCRIPTION JSON:
```json
<JSON>

In <THOUGHT>, briefly discuss your motivations for the description. Detail your high-level plan and ideal outcomes of the molecule generation.

In <JSON>, provide the new description in JSON format with the following field:

    "technical_instruction": A detailed technical instruction for molecule generation.

This JSON will be automatically parsed, so ensure the format is precise. You will have {num_reflections} rounds to iterate on the description, but do not need to use them all. ONLY GENERATE ONE AT A TIME OTHERWISE SMALL BABY TINY INNOCENT KITTENS WILL DIE """

description_reflection_prompt = """ Round {current_round}/{num_reflections}. In your thoughts, carefully consider the quality, novelty, and feasibility of the description you just created. Include any other factors that you think are important in evaluating the description. Ensure the description is clear, concise, and technically accurate. In the next attempt, try to refine and improve your description. Stick to the spirit of the original description unless there are glaring issues.

Respond in the same format as before: THOUGHT: <THOUGHT>

NEW DESCRIPTION JSON:

<JSON>

If there is nothing to improve, simply repeat the previous JSON EXACTLY after the thought and include "I am done" at the end of the thoughts but before the JSON. ONLY INCLUDE "I am done" IF YOU ARE MAKING NO MORE CHANGES. """

class MoleculeDatabase:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        self.db = self.load_db()

    def load_db(self):
        try:
            return DeepLake(dataset_path=self.dataset_path, embedding=self.embeddings, read_only=True)
        except Exception as e:
            logger.error(f"Error loading database from {self.dataset_path}: {str(e)}")
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

def generate_technical_descriptions(base_dir, client, model, user_prompt, molecule_db, max_num_generations=5, num_reflections=3):
    description_system_prompt = "You are an expert in molecular biology and chemistry, tasked with generating technical descriptions for molecule generation."
    description_str_archive = []

    for _ in range(max_num_generations):
        print(f"\nGenerating description {_ + 1}/{max_num_generations}")
        try:
            prev_descriptions_string = "\n\n".join(description_str_archive)

            msg_history = []
            print(f"Iteration 1/{num_reflections}")
            text, msg_history = get_response_from_llm(
                description_first_prompt.format(
                    task_description=user_prompt,
                    num_reflections=num_reflections,
                ),
                client=client,
                model=model,
                system_message=description_system_prompt,
                msg_history=msg_history,
            )
            json_output = extract_json_between_markers(text)
            assert json_output is not None, "Failed to extract JSON from LLM output"
            print(json_output)

            # Search molecule database
            molecules = molecule_db.retrieve_from_db(json_output["technical_instruction"])
            molecule_info = json.dumps([str(m) for m in molecules]) if molecules else "No molecules found."

            # Iteratively improve description
            if num_reflections > 1:
                for j in range(num_reflections - 1):
                    print(f"Iteration {j + 2}/{num_reflections}")
                    text, msg_history = get_response_from_llm(
                        description_reflection_prompt.format(
                            current_round=j + 2, num_reflections=num_reflections
                        ) + f"\n\nMolecule Database Results:\n{molecule_info}",
                        client=client,
                        model=model,
                        system_message=description_system_prompt,
                        msg_history=msg_history,
                    )
                    json_output = extract_json_between_markers(text)
                    assert json_output is not None, "Failed to extract JSON from LLM output"
                    print(json_output)

                    if "I am done" in text:
                        print(f"Description generation converged after {j + 2} iterations.")
                        break

            description_str_archive.append(json.dumps(json_output))

            # Save descriptions after each generation
            descriptions = [json.loads(desc_str) for desc_str in description_str_archive]
            with open(osp.join(base_dir, "technical_descriptions.json"), "w") as f:
                json.dump(descriptions, f, indent=4)

            yield json_output

        except Exception as e:
            print(f"Failed to generate description: {e}")
            continue

def run_Phase_1(user_prompt, max_generations=5, num_reflections=2):
    base_dir = "molecule_generation"
    os.makedirs(base_dir, exist_ok=True)

    dataset_path = r"C:\Users\wes\AutoProtGenerationSystem\Phase_1\technical_description_molecular_database"
    molecule_db = MoleculeDatabase(dataset_path)

    description_generator = generate_technical_descriptions(
        base_dir,
        client=client,
        model=llm_model,
        user_prompt=user_prompt,
        molecule_db=molecule_db,
        max_num_generations=max_generations,
        num_reflections=num_reflections
    )

    print("\nGenerated Technical Descriptions:")
    generated_descriptions = []
    for desc in description_generator:
        print(f"- {desc['technical_instruction']}")
        generated_descriptions.append(desc)

    return generated_descriptions