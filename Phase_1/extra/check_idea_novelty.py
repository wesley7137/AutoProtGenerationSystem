import json
import os.path as osp
from typing import List, Dict, Any, Union
from .model_implementations import ModelInterface
import requests
import time
import xml.etree.ElementTree as ET

def extract_json_between_markers(text: str) -> Dict[str, Any]:
    start = text.find('{')
    end = text.rfind('}') + 1
    if start != -1 and end != -1:
        json_str = text[start:end]
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            return None
    return None

def search_arxiv(query: str, result_limit: int = 10) -> Union[None, List[Dict]]:
    if not query:
        return None
    
    base_url = 'http://export.arxiv.org/api/query?'
    search_query = f'search_query=all:{query}&start=0&max_results={result_limit}'
    
    response = requests.get(base_url + search_query)
    response.raise_for_status()
    
    # Parse the XML response
    root = ET.fromstring(response.content)
    
    # Define the XML namespace
    namespace = {'arxiv': 'http://www.w3.org/2005/Atom'}
    
    papers = []
    for entry in root.findall('arxiv:entry', namespace):
        paper = {
            'title': entry.find('arxiv:title', namespace).text,
            'authors': [author.find('arxiv:name', namespace).text for author in entry.findall('arxiv:author', namespace)],
            'summary': entry.find('arxiv:summary', namespace).text,
            'published': entry.find('arxiv:published', namespace).text,
            'link': entry.find('arxiv:id', namespace).text
        }
        papers.append(paper)
    
    time.sleep(3.0)  # Be respectful to the arXiv API
    return papers if papers else None

def check_idea_novelty(
    ideas: List[Dict[str, Any]],
    base_dir: str,
    model: ModelInterface,
    max_num_iterations: int = 10,
) -> List[Dict[str, Any]]:
    with open(osp.join(base_dir, "topic.json"), "r") as f:
        topic = json.load(f)
        task_description = topic["task_description"]

    novelty_system_msg = """
    You are an AI tasked with determining the novelty of research ideas. Your goal is to assess whether a given idea is novel in the context of existing literature and the specific task at hand.

    Task Description: {task_description}

    You will interact for up to {num_rounds} rounds. In each round, you will receive an idea and the results of a literature search from arXiv. Based on this information, you should either:
    1. Decide if the idea is novel or not novel, or
    2. Request another literature search with a refined query.

    If you make a decision, include either "Decision made: novel" or "Decision made: not novel" in your response.
    If you need another search, provide a JSON output in the following format:
    {{
        "Query": "Your refined search query"
    }}
    """

    novelty_prompt = """
    This is round {current_round} out of {num_rounds}.

    Idea to evaluate:
    {idea}

    Results from the last query:
    {last_query_results}

    Based on this information, decide if the idea is novel, not novel, or if you need more information through another literature search.
    """

    for idx, idea in enumerate(ideas):
        if "novel" in idea:
            print(f"Skipping idea {idx}, already checked.")
            continue

        print(f"\nChecking novelty of idea {idx}: {idea['Name']}")

        novel = False
        msg_history = []
        papers_str = ""

        for j in range(max_num_iterations):
            try:
                response = model.generate([
                    {"role": "system", "content": novelty_system_msg.format(
                        num_rounds=max_num_iterations,
                        task_description=task_description,
                    )},
                    {"role": "user", "content": novelty_prompt.format(
                        current_round=j + 1,
                        num_rounds=max_num_iterations,
                        idea=json.dumps(idea, indent=2),
                        last_query_results=papers_str,
                    )}
                ])
                
                if "decision made: novel" in response.lower():
                    print("Decision made: novel after round", j + 1)
                    novel = True
                    break
                if "decision made: not novel" in response.lower():
                    print("Decision made: not novel after round", j + 1)
                    break

                json_output = extract_json_between_markers(response)
                assert json_output is not None, "Failed to extract JSON from LLM output"

                query = json_output["Query"]
                papers = search_arxiv(query, result_limit=10)
                if not papers:
                    papers_str = "No papers found."
                else:
                    paper_strings = [
                        f"""{i+1}: {paper['title']}
                        Authors: {', '.join(paper['authors'])}
                        Published: {paper['published']}
                        Summary: {paper['summary']}
                        Link: {paper['link']}"""
                        for i, paper in enumerate(papers)
                    ]
                    papers_str = "\n\n".join(paper_strings)

            except Exception as e:
                print(f"Error: {e}")
                continue

        idea["novel"] = novel

    with open(osp.join(base_dir, "ideas.json"), "w") as f:
        json.dump(ideas, f, indent=4)

    return ideas