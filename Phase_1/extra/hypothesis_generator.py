import nltk
import ssl
import os
import json
import time
import requests
import logging
import xml.etree.ElementTree as ET
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from pydantic import BaseModel
from collections import deque
from urllib.parse import quote_plus
import numpy as np
from langchain_ollama import ChatOllama
from datetime import datetime
nltk.download('punkt_tab')
# SSL Context for NLTK download
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Download required NLTK data

# Constants
MAX_ARXIV_RESULTS = 10
MAX_BIORXIV_RESULTS = 10
RESULTS_DIR = "search_results"

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


client = ChatOllama(model="qwen2-math:7b-instruct-q5_0", base_url="http://localhost:11434/v1", openai_api_key="lm-studio")



# Initialize the rate limiter
class RateLimiter:
    def __init__(self, max_calls, period):
        self.max_calls = max_calls
        self.period = period
        self.calls = deque()

    def __call__(self):
        now = time.time()
        while self.calls and now - self.calls[0] >= self.period:
            self.calls.popleft()
        if len(self.calls) >= self.max_calls:
            sleep_time = self.period - (now - self.calls[0])
            time.sleep(sleep_time)
        self.calls.append(time.time())

rate_limiter = RateLimiter(max_calls=5, period=4)
arxiv_rate_limiter = RateLimiter(max_calls=1, period=1)

def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text.lower())
    return ' '.join([word for word in word_tokens if word not in stop_words])

def search_arxiv(topic, max_results=MAX_ARXIV_RESULTS, max_retries=3):
    base_url = "http://export.arxiv.org/api/query"
    filtered_terms = remove_stopwords(topic)
    search_query = " AND ".join(f'all:"{term}"' for term in filtered_terms)
    encoded_query = quote_plus(search_query)
    url = f"{base_url}?search_query={encoded_query}&start=0&max_results={max_results}"

    for attempt in range(max_retries):
        try:
            arxiv_rate_limiter()
            response = requests.get(url)
            response.raise_for_status()
            root = ET.fromstring(response.content)

            results = []
            for entry in root.findall('{http://www.w3.org/2005/Atom}entry'):
                title = entry.find('{http://www.w3.org/2005/Atom}title').text
                abstract = entry.find('{http://www.w3.org/2005/Atom}summary').text
                results.append({'title': title, 'abstract': abstract})

            return results

        except requests.exceptions.RequestException as e:
            logger.error(f"Error occurred: {e}. Retrying... (Attempt {attempt + 1}/{max_retries})")
            time.sleep(1)

    logger.error("Max retries reached. Unable to fetch data from arXiv.")
    return []

def search_pubmed(filtered_topic, max_results=10, max_retries=3, delay=1, email="your-email@example.com"):
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
    encoded_query = quote_plus(filtered_topic)
    search_url = f"{base_url}esearch.fcgi?db=pubmed&term={encoded_query}&retmax={max_results}&usehistory=y&email={email}"

    for attempt in range(max_retries):
        try:
            time.sleep(delay)
            response = requests.get(search_url)
            response.raise_for_status()
            root = ET.fromstring(response.content)
            id_list = [id_elem.text for id_elem in root.findall(".//Id")]

            if not id_list:
                logger.warning("No results found.")
                return None, None

            web_env = root.find(".//WebEnv").text
            query_key = root.find(".//QueryKey").text

            fetch_url = f"{base_url}efetch.fcgi?db=pubmed&WebEnv={web_env}&query_key={query_key}&rettype=abstract&retmode=xml&email={email}"
            fetch_response = requests.get(fetch_url)
            fetch_response.raise_for_status()

            fetch_root = ET.fromstring(fetch_response.content)
            results = []
            for article in fetch_root.findall(".//PubmedArticle"):
                title = article.find(".//ArticleTitle").text
                abstract_elem = article.find(".//Abstract/AbstractText")
                abstract = abstract_elem.text if abstract_elem is not None else "No abstract available"
                results.append({"title": title, "abstract": abstract})

            logger.info(f"Found {len(results)} papers")
            return {"collection": results}

        except (requests.exceptions.RequestException, ET.ParseError) as e:
            logger.error(f"Error occurred: {e}. Retrying... (Attempt {attempt + 1}/{max_retries})")
            time.sleep(1)

    logger.error("Max retries reached. Unable to fetch data from PubMed.")
    return None, None

def save_search_results(topic, arxiv_results, pubmed_results):
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{RESULTS_DIR}/search_results_{timestamp}.json"
    
    results = {
        "topic": topic,
        "timestamp": timestamp,
        "arxiv_results": arxiv_results if arxiv_results else [],
        "pubmed_results": pubmed_results['collection'] if pubmed_results and 'collection' in pubmed_results else []
    }
    
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Search results saved to {filename}")

class HypothesisResponse(BaseModel):
    hypothesis: str

class HypothesisGenerator:
    def __init__(self, knowledge_graph):
        self.knowledge_graph = knowledge_graph
        self.model = "lmstudio-community/DeepSeek-Coder-V2-Lite-Instruct-GGUF/DeepSeek-Coder-V2-Lite-Instruct-Q8_0.gguf"
        logger.info("Hypothesis Generator initialized")

    def generate_hypothesis(self, topic):
        subgraph = self.knowledge_graph.extract_subgraph(topic)
        graph_text = self._subgraph_to_text(subgraph)
        
        prompt = f"Based on the following information about {topic}, generate a novel hypothesis:\n{graph_text}\nHypothesis:"
        
        completion = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a scientific hypothesis generator."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1024
        )

        content = completion.choices[0].message.content
        try:
            hypothesis_data = json.loads(content)
            hypothesis_response = HypothesisResponse(**hypothesis_data)
        except json.JSONDecodeError:
            hypothesis_response = HypothesisResponse(hypothesis=content.strip())

        return hypothesis_response.hypothesis

    def refine_hypothesis(self, hypothesis):
        prompt = f"Refine and improve the following hypothesis:\n{hypothesis}\nRefined hypothesis:"
        
        completion = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a scientific hypothesis refiner."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1024
        )

        content = completion.choices[0].message.content
        try:
            refined_hypothesis_data = json.loads(content)
            refined_hypothesis_response = HypothesisResponse(**refined_hypothesis_data)
        except json.JSONDecodeError:
            refined_hypothesis_response = HypothesisResponse(hypothesis=content.strip())

        return refined_hypothesis_response.hypothesis

    def _subgraph_to_text(self, subgraph):
        return " ".join([f"{u} {d['relation']} {v}" for u, v, d in subgraph.edges(data=True)])

class HypothesisEvaluator:
    def __init__(self, literature_analyzer):
        self.literature_analyzer = literature_analyzer
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def _assess_relevance(self, hypothesis):
        corpus_embeddings = self.model.encode(self.literature_analyzer.documents)
        hypothesis_embedding = self.model.encode([hypothesis])
        similarities = cosine_similarity(hypothesis_embedding, corpus_embeddings)[0]
        return round(float(np.mean(similarities)), 2)
    
    def evaluate(self, hypothesis):
        scores = {
            "novelty": self._assess_novelty(hypothesis),
            "relevance": self._assess_relevance(hypothesis),
            "feasibility": self._assess_feasibility(hypothesis),
            "ethics": self._assess_ethics(hypothesis)
        }
        return scores

    def _assess_novelty(self, hypothesis):
        logger.info(f"Assessing novelty for hypothesis: {hypothesis}")
        if not self.literature_analyzer.documents:
            logger.warning("No documents available for comparison. Returning default novelty score.")
            return 1.0  # Assume maximum novelty if there are no documents to compare against
        
        all_texts = self.literature_analyzer.documents + [hypothesis]
        tfidf_matrix = self.literature_analyzer.vectorizer.fit_transform(all_texts)
        
        if tfidf_matrix.shape[0] < 2:
            logger.warning("Not enough documents for comparison. Returning default novelty score.")
            return 1.0  # Assume maximum novelty if there's only the hypothesis
        
        cosine_similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
        avg_similarity = np.mean(cosine_similarities)
        novelty_score = 1 - avg_similarity  # Higher novelty for lower similarity
        
        logger.info(f"Calculated novelty score: {novelty_score}")
        return round(novelty_score, 2)

    def _assess_feasibility(self, hypothesis):
        word_count = len(hypothesis.split())
        return 0.9 if word_count < 10 else 0.7 if word_count < 20 else 0.5

    def _assess_ethics(self, hypothesis):
        unethical_keywords = ["harmful", "dangerous", "unethical", "illegal"]
        hypothesis_lower = hypothesis.lower()
        ethics_score = 1 - sum(keyword in hypothesis_lower for keyword in unethical_keywords) / len(unethical_keywords)
        return round(ethics_score, 2)

class AutonomousHypothesisSystem:
    def __init__(self):
        logger.info("Initializing AutonomousHypothesisSystem...")
        self.knowledge_graph = None  # Placeholder: Initialize with actual knowledge graph
        self.literature_analyzer = None  # Placeholder: Initialize with actual literature analyzer
        self.hypothesis_generator = HypothesisGenerator(self.knowledge_graph)
        self.hypothesis_evaluator = HypothesisEvaluator(self.literature_analyzer)
        logger.info("AutonomousHypothesisSystem initialized successfully.")

    def add_literature(self, text):
        logger.info(f"Adding new document. Current document count: {len(self.literature_analyzer.documents)}")
        self.literature_analyzer.add_document(text)
        logger.info(f"Document added. Updated document count: {len(self.literature_analyzer.documents)}")

    def update_knowledge_graph(self, new_relations):
        logger.info(f"Updating knowledge graph with {len(new_relations)} new relations...")
        for entity1, entity2, relation in new_relations:
            self.knowledge_graph.add_relation(entity1, entity2, relation)
        logger.info("Knowledge graph updated successfully.")

    def generate_and_evaluate_hypothesis(self, topic):
        logger.info(f"Generating and evaluating hypothesis for topic: {topic}")
        filtered_topic = remove_stopwords(topic)
        arxiv_data = search_arxiv(filtered_topic)
        if arxiv_data:
            logger.info(f"Found {len(arxiv_data)} relevant papers on arXiv.")
            for paper in arxiv_data:
                self.add_literature(paper['abstract'])

        pubmed_data = search_pubmed(filtered_topic)
        if pubmed_data and 'collection' in pubmed_data:
            logger.info(f"Found {len(pubmed_data['collection'])} relevant papers on PubMed.")
            for paper in pubmed_data['collection']:
                self.add_literature(paper.get('abstract', ''))

        save_search_results(topic, arxiv_data, pubmed_data)
        concepts = self.literature_analyzer.extract_key_concepts()
        relevant_concepts = [c for concept_list in concepts.values() for c in concept_list if filtered_topic.lower() in c.lower()]
        logger.info(f"Found {len(relevant_concepts)} relevant concepts.")

        if topic not in self.knowledge_graph.graph:
            logger.info(f"Topic '{topic}' not found in knowledge graph. Adding it now...")
            self.knowledge_graph.add_node(topic)
            for concept in relevant_concepts:
                self.knowledge_graph.add_relation(topic, concept, "related_to")
            logger.info(f"Added topic '{topic}' and connected it to {len(relevant_concepts)} relevant concepts.")

        hypothesis = self.hypothesis_generator.generate_hypothesis(topic)
        logger.info(f"Initial hypothesis generated: {hypothesis}")
        refined_hypothesis = self.hypothesis_generator.refine_hypothesis(hypothesis)
        logger.info(f"Refined hypothesis: {refined_hypothesis}")
        evaluation = self.hypothesis_evaluator.evaluate(refined_hypothesis)
        logger.info("Hypothesis evaluation completed.")
        logger.info(f"Evaluation: {evaluation}")
        logger.info(f"Relevant Concepts: {relevant_concepts}")
        return refined_hypothesis, evaluation, relevant_concepts

# Example Usage
if __name__ == "__main__":
    hypothesis_system = AutonomousHypothesisSystem()
    topic = "novel cellular regeneration and cell longevity to reverse aging"
    hypothesis, evaluation, relevant_concepts = hypothesis_system.generate_and_evaluate_hypothesis(topic)
    
    print(f"\nGenerated Hypothesis on {topic}:")
    print(hypothesis)
    print(f"\nHypothesis Evaluation:")
    for criterion, score in evaluation.items():
        print(f"{criterion.capitalize()}: {score}")
    print(f"\nRelevant Concepts: {', '.join(relevant_concepts)}")
