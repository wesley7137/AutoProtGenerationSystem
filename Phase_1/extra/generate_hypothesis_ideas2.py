import json
import os.path as osp
from typing import List, Dict, Any
import re
from openai import OpenAI
import jsonschema
from langchain_community.chat_models import ChatLlamaCpp


import json
import os.path as osp
from typing import Dict, Any
from langchain_community.chat_models import ChatLlamaCpp


BASE_DIR = r"C:\Users\wes\Automots\longevity_research_framework\hypothesis_network\research_topics\longevity"
#base_dir = (r"C:/Users/wes/Automots/longevity_research_framework/hypothesis_network/research_topics/longevity")
#BASE_URL = "http://172.27.16.1:1234/v1"
BASE_URL = "http://localhost:1234/v1/chat/completions"
#base_url = "http://localhost:11434"
API_KEY = "lm-studio"


from langchain_ollama import ChatOllama

llm = ChatOllama(
    model="deepseek-coder-v2:16b-lite-instruct-q5_K_S",  # Replace with the appropriate model name
    temperature=0.5,
    max_tokens=1024,
    verbose=True,
    stream=True,
)



def extract_json_between_markers(text: str) -> Dict[str, Any]:
    json_match = re.search(r'\{(?:[^{}]|(?R))*\}', text, re.DOTALL)
    if json_match:
        json_str = json_match.group()
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            cleaned_json_str = re.sub(r'[\n\r\t]', '', json_str)
            cleaned_json_str = re.sub(r',\s*}', '}', cleaned_json_str)  # Remove trailing commas
            try:
                return json.loads(cleaned_json_str)
            except json.JSONDecodeError:
                print("Failed to parse JSON even after cleanup.")
                print(f"Problematic JSON string: {cleaned_json_str}")
                return None
    return None




def generate_response(messages):
    print("Generating response...")
    ai_msg = llm.invoke(messages)
    print("AI Message: ", ai_msg.content)
    return ai_msg.content

def clean_json_response(response: str) -> str:
    # Remove code block markers if present
    response = response.strip()
    if response.startswith("```json"):
        response = response[7:]
    if response.endswith("```"):
        response = response[:-3]
    return response.strip()

def generate_single_idea(base_dir: str) -> Dict[str, Any]:
    # Load topic and seed ideas
    with open(osp.join(base_dir, "topic.json"), "r") as f:
        topic = json.load(f)
        print(f"Loaded topic: {topic['task_description']}")
    with open(osp.join(base_dir, "seed_ideas.json"), "r") as f:
        seed_ideas = json.load(f)
        print(f"Loaded seed ideas: {seed_ideas}")

    idea_system_prompt = topic["system"]
    task_description = topic["task_description"]

    print(f"Task Description: {task_description}")
    print(f"System Prompt: {idea_system_prompt}")

    idea_prompt = f"""
    Based on the following task description and previous ideas, generate a new idea for an experiment on self-improving algorithms with safeguards:

    Task Description: {task_description}

    Previous Ideas:
    {json.dumps(seed_ideas, indent=2)}

    Generate a new, unique idea in the following JSON format. Ensure that you provide a complete, valid JSON object with no additional text before or after:

    {{
        "Name": "unique_identifier",
        "Title": "Descriptive Title",
        "Experiment": "Detailed description of the experiment including specific implementation details",
        "Interestingness": 1,
        "Feasibility": 1,
        "Novelty": 1
    }}

    Replace the placeholder values with appropriate content. Ensure all numeric values are integers between 1 and 10.
    The 'Name' should be a short, snake_case identifier.
    The 'Experiment' description should be detailed and include specific steps for implementation.
    Focus on creating a tangible, implementable algorithm that can autonomously evolve with built-in safeguards.
    """

    messages = [
        ("system", idea_system_prompt),
        ("human", idea_prompt),
    ]

    response = generate_response(messages)
    print("Response: ", response)
    cleaned_response = clean_json_response(response)
    print("Cleaned Response: ", cleaned_response)
    return cleaned_response


def save_idea(cleaned_response: Dict[str, Any], base_dir: str):
    try:
        ideas_file = osp.join(base_dir, "ideas.json")
        if osp.exists(ideas_file):
            with open(ideas_file, "r") as f:
                ideas = json.load(f)
        else:
            ideas = []
        
        new_idea = json.loads(cleaned_response)
        ideas.append(new_idea)
        
        print("Generated idea:")
        print(json.dumps(new_idea, indent=2))
        
        # Save all ideas, including the new one
        with open(ideas_file, "w") as f:
            json.dump(ideas, f, indent=4)
        print(f"New idea appended to {ideas_file}. Total ideas: {len(ideas)}")
        
        # Optionally, save the new idea separately
        with open(osp.join(base_dir, "new_idea.json"), "w") as f:
            json.dump(new_idea, f, indent=4)
        print("New idea also saved separately to new_idea.json")
        
        return ideas
    except json.JSONDecodeError:
        print("Failed to parse the generated idea as JSON. Raw response:")
        print("Cleaned response:")
        print(cleaned_response)
        return None

if __name__ == "__main__":
    steps = 0
    for i in range(10):
        cleaned_response = generate_single_idea(BASE_DIR)
        if cleaned_response:
            save_idea(cleaned_response, BASE_DIR)
            steps += 1
        print("Successfully generated and saved a new idea.")
    else:
        print("Failed to generate a valid idea.")

