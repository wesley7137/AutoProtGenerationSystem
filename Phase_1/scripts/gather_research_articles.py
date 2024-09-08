import os
import logging
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import xml.etree.ElementTree as ET
from urllib.parse import quote_plus, urljoin
from bs4 import BeautifulSoup as bs4
from PyPDF2 import PdfReader
from io import BytesIO
import time
from requests.exceptions import RequestException

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ArticleFetcher:
    def __init__(self, articles_folder):
        self.articles_folder = articles_folder
        os.makedirs(self.articles_folder, exist_ok=True)
        self.session = self.create_retry_session()

    def create_retry_session(self, retries=3, backoff_factor=0.3, status_forcelist=(500, 502, 504)):
        session = requests.Session()
        retry = Retry(total=retries, read=retries, connect=retries,
                      backoff_factor=backoff_factor, status_forcelist=status_forcelist)
        adapter = HTTPAdapter(max_retries=retry)
        session.mount('http://', adapter)
        session.mount('https://', adapter)
        return session

    def search_pubmed(self, query, max_results=2):
        base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
        search_url = f"{base_url}esearch.fcgi?db=pubmed&term={quote_plus(query)}&retmax={max_results}&usehistory=y"
        
        response = requests.get(search_url)
        root = ET.fromstring(response.content)
        id_list = [id_elem.text for id_elem in root.findall(".//Id")]
        
        if not id_list:
            logger.warning("No results found in PubMed.")
            return []

        fetch_url = f"{base_url}efetch.fcgi?db=pubmed&id={','.join(id_list)}&rettype=abstract&retmode=xml"
        fetch_response = requests.get(fetch_url)
        fetch_root = ET.fromstring(fetch_response.content)
        
        results = []
        for article in fetch_root.findall(".//PubmedArticle"):
            pmid = article.find(".//PMID").text
            title = article.find(".//ArticleTitle").text
            abstract_elem = article.find(".//Abstract/AbstractText")
            abstract = abstract_elem.text if abstract_elem is not None else "No abstract available"
            
            full_text = self.download_pubmed_fulltext(pmid)
            
            results.append({
                "title": title,
                "abstract": abstract,
                "full_text": full_text
            })
        
        logger.info(f"Found and processed {len(results)} papers from PubMed")
        return results

    def get_pubmed_pdf_url(self, pmid):
        try:
            url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
            response = requests.get(url)
            response.raise_for_status()
            soup = bs4(response.text, 'html.parser')
            pdf_link = soup.find('a', {'class': 'link-item dialog-focus'})
            if pdf_link and 'href' in pdf_link.attrs:
                return urljoin(url, pdf_link['href'])
            return None
        except Exception as e:
            logger.error(f"Error getting PDF URL for PMID {pmid}: {str(e)}")
            return None

    def download_pubmed_fulltext(self, pmid):
        try:
            pmc_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id={pmid}&retmode=xml"
            response = self.session.get(pmc_url, timeout=30)
            response.raise_for_status()
            
            root = ET.fromstring(response.content)
            pmc_id = root.find(".//ArticleId[@IdType='pmc']")
            
            if pmc_id is not None:
                pmc_id = pmc_id.text
                fulltext_url = f"https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi/BioC_xml/{pmc_id}/unicode"
                fulltext_response = self.session.get(fulltext_url, timeout=30)
                fulltext_response.raise_for_status()
                
                fulltext_root = ET.fromstring(fulltext_response.content)
                passages = fulltext_root.findall(".//passage/text")
                full_text = "\n".join([p.text for p in passages if p.text])
                return full_text
            
            pdf_url = self.get_pubmed_pdf_url(pmid)
            if pdf_url:
                return self.download_pdf(pdf_url)
            
            return "Full text not available"
        except Exception as e:
            logger.error(f"Error downloading full text for PMID {pmid}: {str(e)}")
            return f"Error downloading full text: {str(e)}"

    def download_pdf(self, pdf_url):
        try:
            response = self.session.get(pdf_url, timeout=30)
            response.raise_for_status()
            
            pdf_file = BytesIO(response.content)
            
            reader = PdfReader(pdf_file)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
            
            return text
        except Exception as e:
            logger.error(f"Error downloading or processing PDF from {pdf_url}: {str(e)}")
            return f"Error downloading or processing PDF: {str(e)}"

    def search_arxiv(self, query, max_results=2, max_retries=3, retry_delay=5):
        base_url = "http://export.arxiv.org/api/query"
        search_query = quote_plus(query)
        url = f"{base_url}?search_query=all:{search_query}&start=0&max_results={max_results}"
        
        for attempt in range(max_retries):
            try:
                response = requests.get(url, timeout=30)
                response.raise_for_status()
                
                root = ET.fromstring(response.content)
                results = []
                for entry in root.findall('{http://www.w3.org/2005/Atom}entry'):
                    title = entry.find('{http://www.w3.org/2005/Atom}title').text
                    abstract = entry.find('{http://www.w3.org/2005/Atom}summary').text
                    pdf_url = entry.find('{http://www.w3.org/2005/Atom}link[@title="pdf"]').get('href')
                    
                    full_text = self.download_arxiv_pdf(pdf_url)
                    
                    results.append({
                        "title": title,
                        "abstract": abstract,
                        "full_text": full_text
                    })
                
                logger.info(f"Found and processed {len(results)} papers from arXiv")
                return results
            
            except RequestException as e:
                logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt < max_retries - 1:
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    logger.error("Max retries reached. Unable to fetch results from arXiv.")
                    return []

    def download_arxiv_pdf(self, pdf_url):
        try:
            response = requests.get(pdf_url)
            response.raise_for_status()
            
            pdf_file = BytesIO(response.content)
            
            reader = PdfReader(pdf_file)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
            
            return text
        except Exception as e:
            logger.error(f"Error downloading or processing PDF from {pdf_url}: {str(e)}")
            return "Error downloading or processing PDF"

    def fetch_articles(self, query, max_results_per_source=2):
        pubmed_results = self.search_pubmed(query, max_results_per_source)
        arxiv_results = self.search_arxiv(query, max_results_per_source)
        
        all_results = pubmed_results + arxiv_results
        
        for result in all_results:
            self.save_article(result['title'], f"Abstract: {result['abstract']}\n\nFull Text: {result['full_text']}")
        
        return all_results

    def save_article(self, title, content):
        safe_title = "".join([c if c.isalnum() or c in [' ', '-', '_'] else '_' for c in title])
        filename = f"{safe_title[:50]}.txt"
        filepath = os.path.join(self.articles_folder, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        logger.info(f"Saved article: {filepath}")

if __name__ == "__main__":
    articles_folder = r"C:\path\to\your\articles_folder"
    fetcher = ArticleFetcher(articles_folder)
    
    query = "novel cellular regeneration and cell longevity to reverse aging"
    results = fetcher.fetch_articles(query)
    
    print(f"Fetched {len(results)} articles:")
    for result in results:
        print(f"Title: {result['title']}")
        print(f"Abstract: {result['abstract'][:100]}...")
        print("---")