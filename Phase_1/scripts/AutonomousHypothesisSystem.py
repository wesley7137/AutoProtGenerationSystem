import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from langchain_openai import ChatOpenAI
from research_and_summarize_utils import extract_key_information
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
from langchain_ollama import ChatOllama
from datetime import datetime
import numpy as np
from Phase_1.scripts.LiteratureAnalyzer import LiteratureAnalyzer
from Phase_1.scripts.KnowledgeGraph import KnowledgeGraph
from openai import OpenAI


client = OpenAI(base_url="http://localhost:11434/v1", api_key="lm-studio")
llm_model = "akjindal53244/Llama-3.1-Storm-8B-GGUF/Llama-3.1-Storm-8B.Q8_0.gguf"
# SSL Context for NLTK download
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Download required NLTK data
nltk.download('punkt')

# Constants
MAX_ARXIV_RESULTS = 10
MAX_BIORXIV_RESULTS = 10
RESULTS_DIR = "search_results"

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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

def search_arxiv(user_input, max_results=MAX_ARXIV_RESULTS, max_retries=3):
    base_url = "http://export.arxiv.org/api/query"
    filtered_terms = remove_stopwords(user_input)
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

def search_pubmed(filtered_user_input, max_results=10, max_retries=3, delay=1, email="your-email@example.com"):
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
    encoded_query = quote_plus(filtered_user_input)
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

def save_search_results(user_input, arxiv_results, pubmed_results):
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{RESULTS_DIR}/search_results_{timestamp}.json"
    
    results = {
        "user_input": user_input,
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
        self.client = OpenAI(base_url="http://localhost:11434/v1", api_key="lm-studio")
        self.model = "phi3.5:3.8b-mini-instruct-q8_0"
        logger.info("Hypothesis Generator initialized")

    def generate_hypothesis(self, user_input):
        subgraph = self.knowledge_graph.extract_subgraph(user_input)
        graph_text = self._subgraph_to_text(subgraph)
        
        prompt = f"Based on the following information about {user_input}, generate a novel hypothesis:\n{graph_text}\nHypothesis:"
        
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a scientific hypothesis generator."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1024
        )

        content = completion.choices[0].message.content
        return self._parse_hypothesis(content)

    def refine_hypothesis(self, hypothesis):
        prompt = f"Refine and improve the following hypothesis:\n{hypothesis}\nRefined hypothesis:"
        
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a scientific hypothesis refiner."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1024
        )

        content = completion.choices[0].message.content
        return self._parse_hypothesis(content)

    def _subgraph_to_text(self, subgraph):
        return " ".join([f"{u} {d['relation']} {v}" for u, v, d in subgraph.edges(data=True)])

    def _parse_hypothesis(self, content):
        try:
            hypothesis_data = json.loads(content)
            hypothesis_response = HypothesisResponse(**hypothesis_data)
            return hypothesis_response.hypothesis
        except json.JSONDecodeError:
            # If it's not JSON, assume it's a plain string hypothesis
            return content.strip()
        except Exception as e:
            logger.error(f"Error parsing hypothesis: {e}")
            return "Failed to generate a valid hypothesis."
        
        
        
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






class AutonomousHypothesisSystem:
    def __init__(self, user_input):
        logger.info("Initializing AutonomousHypothesisSystem...")
        self.knowledge_graph = KnowledgeGraph()
        self.literature_analyzer = LiteratureAnalyzer()
        self.hypothesis_generator = HypothesisGenerator(self.knowledge_graph)
        self.hypothesis_evaluator = HypothesisEvaluator(self.literature_analyzer)
        self.user_input = user_input
        self.knowledge_graph.add_node(self.user_input)  # Add user input to the graph
        logger.info("AutonomousHypothesisSystem initialized successfully.")
        self.client = OpenAI(base_url="http://localhost:11434/v1", api_key="lm-studio")
        self.model = "phi3.5:3.8b-mini-instruct-q8_0"
        nltk.download('stopwords', quiet=True)
        nltk.download('punkt', quiet=True)
        self.stop_words = set(stopwords.words('english'))

    def add_literature(self, text):
        logger.info(f"Adding new document. Current document count: {len(self.literature_analyzer.documents)}")
        self.literature_analyzer.add_document(text)
        logger.info(f"Document added. Updated document count: {len(self.literature_analyzer.documents)}")
        print(f"Added document: {text[:20]}{'...' if len(text) > 20 else ''}")

    def process_literature(self, articles):
        logger.info("Processing literature...")
        
        # Summarize each article
        summaries = []
        for article in articles:
            summary = extract_key_information(article)
            summaries.append(summary)
            if isinstance(summary, str):
                print(f"Article summary: {summary[:50]}...")
            else:
                print(f"Article summary: {str(summary)[:50]}...")

        # Generate a concise summary of all articles using OpenAI API
        combined_summary = self._generate_combined_summary(summaries)
        print(f"Combined summary: {combined_summary[:100]}...")

        # Extract key concepts
        key_concepts = self._extract_key_concepts(combined_summary)
        print(f"Extracted key concepts: {', '.join(key_concepts[:10])}...")

        return key_concepts

    def _generate_combined_summary(self, summaries):
        prompt = f"Summarize the following article summaries into a concise, single paragraph:\n\n"
        for summary in summaries:
            if isinstance(summary, str):
                prompt += summary + "\n\n"
            else:
                prompt += str(summary) + "\n\n"

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a scientific literature summarizer."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=300
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Error generating combined summary: {e}")
            return "Failed to generate combined summary."
    
    
    def _extract_key_concepts(self, text):
        # Tokenize and remove stopwords
        words = word_tokenize(text.lower())
        key_concepts = [word for word in words if word.isalnum() and word not in self.stop_words]
        return list(set(key_concepts))  # Remove duplicates

    def generate_and_evaluate_hypothesis(self):
        logger.info(f"Generating and evaluating hypothesis for user_input: {self.user_input}")
        
        
        for concept, concept_type, _ in relevant_concepts:
            self.knowledge_graph.add_relation(self.user_input, concept, f"related_{concept_type}")
        
        try:
            hypothesis = self.hypothesis_generator.generate_hypothesis(self.user_input)
            logger.info(f"Initial hypothesis generated: {hypothesis}")
            
            refined_hypothesis = self.hypothesis_generator.refine_hypothesis(hypothesis)
            logger.info(f"Refined hypothesis: {refined_hypothesis}")
            
            evaluation = self.hypothesis_evaluator.evaluate(refined_hypothesis)
            logger.info("Hypothesis evaluation completed.")
            logger.info(f"Evaluation: {evaluation}")
        except Exception as e:
            logger.error(f"Error during hypothesis generation and evaluation: {e}")
            return None, None, relevant_concepts
        
        return refined_hypothesis, evaluation, relevant_concepts

# Example usage
if __name__ == "__main__":
    user_input = "novel cellular regeneration and cell longevity to reverse aging"
    hypothesis_system = AutonomousHypothesisSystem(user_input)
    
    sample_articles = [
        "Cellular regeneration involves the repair and replacement of damaged cells. Recent studies have shown promising results in stimulating cellular regeneration through genetic manipulation and targeted drug therapies.",
        "Longevity research focuses on extending the lifespan of cells and organisms. Scientists are exploring various approaches, including telomere preservation, mitochondrial optimization, and epigenetic reprogramming.",
        "Aging is characterized by a decline in cellular function and regenerative capacity. Researchers are investigating the role of senescent cells and developing senolytic drugs to selectively eliminate these cells and potentially reverse aging effects.",
        "Novel approaches to reverse aging target cellular metabolism and DNA repair mechanisms. Cutting-edge techniques like CRISPR gene editing and nanotechnology-based drug delivery systems are being explored to enhance cellular repair and rejuvenation."
    ]
    
    key_concepts = hypothesis_system.process_literature(sample_articles)
    
    print("\nFinal Key Concepts:")
    print(", ".join(key_concepts))

    # Generate and evaluate hypothesis using the extracted key concepts
    hypothesis, evaluation, relevant_concepts = hypothesis_system.generate_and_evaluate_hypothesis()
    
    if hypothesis:
        print(f"\nGenerated Hypothesis:")
        print(hypothesis)
        if evaluation:
            print(f"\nHypothesis Evaluation:")
            for criterion, score in evaluation.items():
                print(f"{criterion.capitalize()}: {score}")
        else:
            print("\nFailed to evaluate the hypothesis.")
    else:
        print("\nFailed to generate a hypothesis.")