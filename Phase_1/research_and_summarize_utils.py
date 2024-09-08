import requests
import spacy
from bs4 import BeautifulSoup
import time
import os
import subprocess
import sys
import logging

logger = logging.getLogger(__name__)

def download_spacy_model(model_name="en_core_web_sm"):
    """
    Downloads the specified spaCy model if it's not already installed.
    
    :param model_name: Name of the spaCy model to download (default: "en_core_web_sm")
    """
    try:
        spacy.load(model_name)
        logger.info(f"spaCy model '{model_name}' is already installed.")
    except OSError:
        logger.info(f"Downloading spaCy model '{model_name}'...")
        subprocess.check_call([sys.executable, "-m", "spacy", "download", model_name])
        logger.info(f"spaCy model '{model_name}' has been successfully downloaded.")

# Ensure the spaCy model is downloaded before loading
download_spacy_model()

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize NLP models

# Rate limiting parameters
RATE_LIMIT = 2  # requests per second
RATE_LIMIT_PERIOD = 1  # second

# Phase 1: Data Ingestion and Search Functionality
def search_academic_sources(keywords, max_results=10):
    """
    Searches multiple academic sources (PubMed, arXiv, CrossRef) for articles based on keywords.
    
    :param keywords: List of keywords for the search query.
    :param max_results: Maximum number of articles to retrieve from each source per keyword.
    :return: List of articles with abstracts and metadata.
    """
    pubmed_articles = search_pubmed(keywords, max_results)
    time.sleep(RATE_LIMIT_PERIOD)  # Rate limiting between different API calls
    arxiv_articles = search_arxiv(keywords, max_results)
    time.sleep(RATE_LIMIT_PERIOD)  # Rate limiting between different API calls
    crossref_articles = search_crossref(keywords, max_results)
    
    all_articles = pubmed_articles + arxiv_articles + crossref_articles
    return all_articles


def search_pubmed(keywords, max_results=5):
    """
    Searches PubMed for articles based on keywords.

    :param keywords: List of keywords for the search query.
    :param max_results: Maximum number of articles to retrieve per keyword.
    :return: List of article abstracts and metadata.
    """
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    all_articles = []

    for keyword in keywords:
        params = {
            "db": "pubmed",
            "term": keyword,
            "retmax": max_results,
            "retmode": "json"
        }

        try:
            response = requests.get(base_url, params=params)
            response.raise_for_status()
            data = response.json()
            logger.info(f"PubMed search response for '{keyword}': {data}")
            article_ids = data.get('esearchresult', {}).get('idlist', [])
            
            if not article_ids:
                logger.warning(f"No PubMed articles found for the keyword: {keyword}")
                continue

            fetch_params = {
                "db": "pubmed",
                "id": ",".join(article_ids),
                "retmode": "xml",
                "rettype": "abstract"
            }
            fetch_response = requests.get(fetch_url, params=fetch_params)
            fetch_response.raise_for_status()

            soup = BeautifulSoup(fetch_response.content, "lxml-xml")
            for article in soup.find_all("PubmedArticle"):
                metadata = {
                    "title": article.find("ArticleTitle").text if article.find("ArticleTitle") else "N/A",
                    "abstract": " ".join([abstract.text for abstract in article.find_all("AbstractText")]),
                    "authors": [author.find("LastName").text for author in article.find_all("Author") if author.find("LastName")],
                    "journal": article.find("Title").text if article.find("Title") else "N/A",
                    "doi": article.find("ELocationID", {"EIdType": "doi"}).text if article.find("ELocationID", {"EIdType": "doi"}) else "N/A",
                    "source": "PubMed",
                    "keyword": keyword
                }
                all_articles.append(metadata)

            time.sleep(1 / RATE_LIMIT)  # Rate limiting

        except requests.RequestException as e:
            logger.error(f"Error fetching PubMed articles for keyword '{keyword}': {e}")

    logger.info(f"Total PubMed articles found: {len(all_articles)}")
    return all_articles

    
def search_arxiv(keywords, max_results=5):
    """
    Searches arXiv for articles based on keywords.

    :param keywords: List of keywords for the search query.
    :param max_results: Maximum number of articles to retrieve.
    :return: List of article abstracts and metadata.
    """
    base_url = "http://export.arxiv.org/api/query"
    query = "+AND+".join(keywords)
    params = {
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": max_results
    }

    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        # Use 'lxml' parser explicitly
        soup = BeautifulSoup(response.content, "lxml-xml")
        articles = []

        for entry in soup.find_all("entry"):
            metadata = {
                "title": entry.find("title").text if entry.find("title") else "N/A",
                "abstract": entry.find("summary").text if entry.find("summary") else "N/A",
                "authors": [author.find("name").text for author in entry.find_all("author")],
                "journal": "arXiv",
                "doi": entry.find("id").text if entry.find("id") else "N/A",
                "source": "arXiv"
            }
            articles.append(metadata)

        time.sleep(1 / RATE_LIMIT)  # Rate limiting
        return articles

    except requests.RequestException as e:
        print(f"Error fetching arXiv articles: {e}")
        return []
    
    
def search_crossref(keywords, max_results=5):
    """
    Searches CrossRef for articles based on keywords.

    :param keywords: List of keywords for the search query.
    :param max_results: Maximum number of articles to retrieve.
    :return: List of article abstracts and metadata.
    """
    base_url = "https://api.crossref.org/works"
    params = {
        "query": " ".join(keywords),
        "rows": max_results
    }

    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()
        items = data.get('message', {}).get('items', [])
        articles = []

        for item in items:
            metadata = {
                "title": item.get("title", ["N/A"])[0],
                "abstract": item.get("abstract", "N/A"),
                "authors": [author.get("family", "Unknown") for author in item.get("author", [])],
                "journal": item.get("container-title", ["N/A"])[0],
                "doi": item.get("DOI", "N/A"),
                "source": "CrossRef"
            }
            articles.append(metadata)

        time.sleep(1 / RATE_LIMIT)  # Rate limiting
        return articles

    except requests.RequestException as e:
        print(f"Error fetching CrossRef articles: {e}")
        return []



import spacy
from transformers import pipeline
from langchain_ollama import ChatOllama
import json
# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Initialize summarization pipeline
from langchain_ollama import ChatOllama

# Initialize the ChatOllama model
summarizer = ChatOllama(
    model="mathstral:7b-v0.1-q6_K",
    temperature=0.2,
    max_tokens=512,
    top_p=0.5,
)

# Initialize the ChatOllama model
chat_model = ChatOllama(
    model="mathstral:7b-v0.1-q6_K",  # You can change this to any available model
    temperature=0.3,
    max_tokens=512,
)

def summarize_text(text):
    """
    Summarizes the given text using the ChatOllama model.

    :param text: The text to summarize.
    :return: Summarized text.
    """
    try:
        prompt = "Summarize the following text in a concise manner:"
        prompt = prompt + text
        
        # Invoke the model with the input text
        completion = summarizer.invoke(prompt)
        summary = completion.content
        print(summary)
        # The response is already a string, so we can return it directly
        return summary
    except Exception as e:
        print(f"Error during summarization: {e}")
        return "Summary not available."
    
    
# Phase 2: NLP for Information Extraction and Summarization
def extract_key_information(text):
    """
    Extracts key information such as entities, methodologies, and results from text using spaCy.

    :param text: The text to analyze.
    :return: Dictionary of extracted entities and key information.
    """
    doc = nlp(text)
    entities = {
        "METHODS": [],
        "RESULTS": [],
        "DISEASES": [],
        "TREATMENTS": [],
        "OTHER": []
    }
    
    # Classify sentences into categories based on keywords
    for sent in doc.sents:
        if "method" in sent.text.lower() or "procedure" in sent.text.lower():
            entities["METHODS"].append(sent.text)
        elif "result" in sent.text.lower() or "finding" in sent.text.lower():
            entities["RESULTS"].append(sent.text)
        else:
            entities["OTHER"].append(sent.text)

    # Extract specific entities like diseases and treatments
    for ent in doc.ents:
        if ent.label_ == "DISEASE":
            entities["DISEASES"].append(ent.text)
        elif ent.label_ == "TREATMENT":
            entities["TREATMENTS"].append(ent.text)
    
    return entities



def generate_description(methods, results):
    """
    Generates a natural language description of methodologies and results using ChatOllama.

    :param methods: List of method descriptions.
    :param results: List of result descriptions.
    :return: Generated description as a string.
    """
    prompt = (
        f"Based on the following methods and results, generate a concise and clear description:\n\n"
        f"Methods:\n{methods}\n\nResults:\n{results}\n\n"
        "Provide a comprehensive summary that explains the significance and implications of these findings."
    )
    
    try:
        # Call ChatOllama to generate the description
        completion = chat_model.invoke(prompt)
        response = completion.content
        print(response)
        # The response is already a string, so we can return it directly
        return response
    except Exception as e:
        print(f"Error generating description: {e}")
        return "Description generation failed."
    
    
    
def process_and_save_article(article):
    # Extract key information
    extracted_info = extract_key_information(article['abstract'])
    
    # Generate summary
    summary = summarize_text(article['abstract'])
    
    # Generate description
    description = generate_description(extracted_info['METHODS'], extracted_info['RESULTS'])
    
    # Combine all information
    processed_article = {
        'title': article['title'],
        'abstract': article['abstract'],
        'summary': summary,
        'extracted_info': extracted_info,
        'generated_description': description
    }
    
    # Save to a file (you could also save to a database)
    with open(f"processed_articles/{article['id']}.json", 'w') as f:
        json.dump(processed_article, f, indent=2)

