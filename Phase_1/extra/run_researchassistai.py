import json
import os
import requests
import spacy
from bs4 import BeautifulSoup
import time
from langchain_ollama import ChatOllama
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize NLP models
nlp = spacy.load("en_core_web_sm")

# Initialize the ChatOllama models
summarizer = ChatOllama(
    model="mathstral:7b-v0.1-q6_K",
    temperature=0.2,
    max_tokens=512,
    top_p=0.5,
)

chat_model = ChatOllama(
    model="mathstral:7b-v0.1-q6_K",  # You can change this to any available model
    temperature=0.3,
    max_tokens=512,
)

# Rate limiting parameters
RATE_LIMIT = 2  # requests per second
RATE_LIMIT_PERIOD = 1  # second

def search_academic_sources(keywords, max_results=10):
    """Searches multiple academic sources (PubMed, arXiv, CrossRef) for articles based on keywords."""
    pubmed_articles = search_pubmed(keywords, max_results)
    time.sleep(RATE_LIMIT_PERIOD)  # Rate limiting between different API calls
    arxiv_articles = search_arxiv(keywords, max_results)
    time.sleep(RATE_LIMIT_PERIOD)  # Rate limiting between different API calls
    crossref_articles = search_crossref(keywords, max_results)
    
    all_articles = pubmed_articles + arxiv_articles + crossref_articles
    return all_articles

def search_pubmed(keywords, max_results=5):
    """Searches PubMed for articles based on keywords."""
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
    """Searches arXiv for articles based on keywords."""
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
        logger.error(f"Error fetching arXiv articles: {e}")
        return []

def search_crossref(keywords, max_results=5):
    """Searches CrossRef for articles based on keywords."""
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
        logger.error(f"Error fetching CrossRef articles: {e}")
        return []

def summarize_text(text):
    """Summarizes the given text using the ChatOllama model."""
    try:
        prompt = "Summarize the following text in a concise manner:\n" + text
        completion = summarizer.invoke(prompt)
        summary = completion.content
        return summary
    except Exception as e:
        logger.error(f"Error during summarization: {e}")
        return "Summary not available."

def extract_key_information(text):
    """Extracts key information such as entities, methodologies, and results from text using spaCy."""
    doc = nlp(text)
    entities = {
        "METHODS": [],
        "RESULTS": [],
        "DISEASES": [],
        "TREATMENTS": [],
        "OTHER": []
    }

    for sent in doc.sents:
        if "method" in sent.text.lower() or "procedure" in sent.text.lower():
            entities["METHODS"].append(sent.text)
        elif "result" in sent.text.lower() or "finding" in sent.text.lower():
            entities["RESULTS"].append(sent.text)
        else:
            entities["OTHER"].append(sent.text)

    for ent in doc.ents:
        if ent.label_ == "DISEASE":
            entities["DISEASES"].append(ent.text)
        elif ent.label_ == "TREATMENT":
            entities["TREATMENTS"].append(ent.text)

    return entities

def generate_description(methods, results):
    """Generates a natural language description of methodologies and results using ChatOllama."""
    prompt = (
        f"Based on the following methods and results, generate a concise and clear description:\n\n"
        f"Methods:\n{methods}\n\nResults:\n{results}\n\n"
        "Provide a comprehensive summary that explains the significance and implications of these findings."
    )

    try:
        completion = chat_model.invoke(prompt)
        response = completion.content
        return response
    except Exception as e:
        logger.error(f"Error generating description: {e}")
        return "Description generation failed."

def main():
    keywords = ["longevity", "mitochondrial", "aging", "protein folding", "autophagy", "bio multi-modal datasets", "machine learning"]
    max_results = 5  # Specify the number of results to fetch

    # Phase 1: Search Academic Sources
    articles = search_academic_sources(keywords, max_results)
    if not articles:
        logger.error("No articles found.")
        return

    # Display the articles retrieved for validation
    logger.info(f"Retrieved {len(articles)} articles from multiple sources.")
    for i, article in enumerate(articles, 1):
        logger.info(f"Article {i}: {article['title']}")

    # Phase 2: Information Extraction and Summarization
    summarized_articles = []
    for article in articles:
        key_info = extract_key_information(article['abstract'])
        summary = summarize_text(article['abstract'])
        generated_desc = generate_description(key_info['METHODS'], key_info['RESULTS'])

        summarized_article = {
            'title': article['title'],
            'abstract': article['abstract'],
            'summary': summary,
            'extracted_info': key_info,
            'generated_description': generated_desc,
            'citation_info': {
                'authors': article.get('authors', []),
                'year': article.get('year', ''),
                'title': article['title'],
                'journal': article.get('journal', 'N/A'),
                'doi': article.get('doi', 'N/A'),
                'publication_date': article.get('publication_date', 'N/A')
            }
        }
        summarized_articles.append(summarized_article)

    # Combine all generated descriptions from Phase 2
    all_generated_descriptions = [article['generated_description'] for article in summarized_articles]

    # Final Technical Description Preparation
    technical_description = f"Task: {keywords}\n\n" \
                            f"Generated Descriptions:\n" + "\n".join(all_generated_descriptions)

    # Display the final technical description
    logger.info("\nFinal Technical Description for Sequence Generation:")
    logger.info(technical_description)

    # Save the final technical description to a file if needed
    os.makedirs("final_output", exist_ok=True)
    with open("final_output/technical_description.txt", "w") as f:
        f.write(technical_description)

    logger.info("Pipeline execution completed.")

if __name__ == "__main__":
    main()
