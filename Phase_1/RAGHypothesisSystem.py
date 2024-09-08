


import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import logging

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import re
from openai import OpenAI
from langchain_text_splitters import CharacterTextSplitter
from urllib.parse import urljoin
from langchain.schema import Document
from bs4 import BeautifulSoup as bs4
import xml.etree.ElementTree as ET
from urllib.parse import quote_plus
from langchain_community.vectorstores import DeepLake
from langchain_huggingface import HuggingFaceEmbeddings
import requests
import time
from requests.exceptions import RequestException
from PyPDF2 import PdfReader
from io import BytesIO
# Import stopwords from NLTK
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
# Download stopwords if not already present
import nltk

import tiktoken

nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)



logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
llm_model = "akjindal53244/Llama-3.1-Storm-8B-GGUF"




class ComprehensiveHypothesisSystem:
    def __init__(self, dataset_path_articles, dataset_path_omics, articles_folder):
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        self.dataset_path_articles = dataset_path_articles
        self.dataset_path_omics = dataset_path_omics
        self.db_articles = self.load_or_create_db(self.dataset_path_articles)
        self.db_omics = self.load_or_create_db(self.dataset_path_omics)

        self.documents = []  # Initialize the documents list
        self.vector_store = self.db_articles  # Initialize vector_store as None
        self.articles_folder = articles_folder
        os.makedirs(self.articles_folder, exist_ok=True)
        self.session = self.create_retry_session()

    def save_article(self, title, content):
        # Sanitize the filename
        safe_title = re.sub(r'[^\w\-_\. ]', '_', title)
        filename = f"{safe_title[:50]}.txt"  # Use first 50 chars of sanitized title as filename
        filepath = os.path.join(self.articles_folder, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)

    def process_search_results(self, search_results):
        for result in search_results:
            text = f"Title: {result['title']}\nAbstract: {result['abstract']}\nFull Text: {result['full_text']}"
            self.documents.append(text)
            self.save_article(result['title'], text)
            
    def load_or_create_db(self, dataset_path):
        try:
            return DeepLake(dataset_path=dataset_path, embedding=self.embeddings)
        except Exception:
            return DeepLake.from_documents([], embedding=self.embeddings, dataset_path=dataset_path)
        
        

    def _update_vector_store(self):
        text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        docs = [Document(page_content=text) for text in self.documents]
        split_docs = text_splitter.split_documents(docs)
        
        if self.vector_store is None:
            self.vector_store = DeepLake.from_documents(
                split_docs,
                self.embeddings,
                dataset_path=self.dataset_path_articles,
                overwrite=False
            )
        else:
            self.vector_store.add_documents(split_docs)
        
        self.db_articles = self.vector_store

    def retrieve_from_articles_db(self, query):
        retriever = self.db_articles.as_retriever()
        retriever.search_kwargs["distance_metric"] = "cos"
        retriever.search_kwargs["fetch_k"] = 5
        retriever.search_kwargs["k"] = 5
        
        db_context = retriever.invoke(query)
        print("Articles db_context", db_context)
        return db_context

    def retrieve_from_omics_db(self, query):
        retriever = self.db_omics.as_retriever()
        retriever.search_kwargs["distance_metric"] = "cos"
        retriever.search_kwargs["fetch_k"] = 1
        retriever.search_kwargs["k"] = 1
        
        db_context = retriever.invoke(query)
        print("Omics db_context", db_context)
        return db_context
    
    
    def truncate_text(self, text, max_tokens=500):
        encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        tokens = encoding.encode(text)
        if len(tokens) <= max_tokens:
            return text
        return encoding.decode(tokens[:max_tokens])


    def formulate_hypothesis(self, query, articles_context):
        
        system_message = """You are a specialized AI assistant tasked with generating detailed, technical descriptions for sequence generation AI models based on scientific research. Your goal is to extract relevant information from database research results and a user's prompt, then formulate a more technical and detailed description using scientific terms and methods.

        You will be provided both the database results (from articles and omics databases) and the user's prompt.
        
        Carefully analyze both the database results and the user's prompt. Extract all relevant scientific information, paying particular attention to:
        1. Specific chemical structures
        2. Amino acid sequences
        3. Protein interactions
        4. Molecular mechanisms
        5. Biochemical pathways
        6. Experimental methods and techniques
        
        Using the extracted information, formulate a detailed technical description for a sequence generation AI model. Your description should:
        1. Be highly specific and scientifically accurate
        2. Incorporate relevant technical terminology
        3. Describe chemical structures and interactions in detail
        4. Explain molecular mechanisms and pathways
        5. Reference specific amino acids and their roles
        6. Mention relevant experimental methods or techniques
        Ensure that your description is more technical and detailed than the original user prompt, while still addressing the core concepts or questions presented.
        Present your response in a concise, technical and scientific manner.
        Remember to focus on scientific accuracy and detail in your description. If there are any ambiguities or missing information in the database results or user prompt, state these limitations clearly in your response.
        
        Your response should be formatted as a JSON object with the following structure:
        Example 1: 
        "user_input": "generate a protein that increases telomerase in the cells",
        "hypothesis": "Generate a protein that will upregulate production of ACD-POT1 heterodimer to enhance telomere elongation which increases telomerase processivity",
        
        Example 2: 
        "user_input": "novel cellular regeneration and cell longevity to reverse aging",
        "hypothesis": "Design a small molecule inhibitor that selectively targets and promotes the degradation of DAF-2 (insulin-like growth factor 1 receptor) to mimic the lifespan extension observed in C. elegans.",
    
        Example 3: 
        "user_input": "Generate an algorithm for a self-improving computer system",
        "hypothesis": "Develop a compound that activates DAF-16 (FOXO transcription factor) to upregulate the expression of stress resistance and longevity-associated genes.",
    
        Example 4: 
        "user_input": "Generate a protein that inhibits the growth of cancer cells",
        "hypothesis": "Create a dual-action molecule that simultaneously inhibits PI3K (phosphatidylinositol 3-kinase) and AKT kinase to attenuate mTOR signaling and enhance autophagy.",

        """
        
        
        truncated_articles_context = self.truncate_text(str(articles_context))

        completion = client.chat.completions.create(
            model=llm_model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": f"Articles Database Results: {truncated_articles_context}\n\n\n\nUser Prompt: {query}"}
            ],
        )
        
        print("completion", completion)
        response = completion.choices[0].message.content
        print("response", response)
        return response        
            

          
    def generate_search_queries(self, query):
        completion = client.chat.completions.create(
            model=llm_model,
            messages=[
                {"role": "system", "content": "Generate 3-5 search queries related to the user's prompt. The queries should not contain stop words or be verbose. Only use keywords that would return the data needed to generate a hypothesis. Return the queries as a numbered list."},
                {"role": "user", "content": f"User Prompt: {query}"}
            ],
            max_tokens=512,
            temperature=0.7,
        )
        
        response = completion.choices[0].message.content
        # Split the response into individual queries
        queries = [q.strip() for q in response.split('\n') if q.strip()]
        # Remove numbering if present
        queries = [q.split('. ', 1)[-1] if '. ' in q else q for q in queries]
        # Get English stopwords
        stop_words = set(stopwords.words('english'))
        
        # Remove stopwords from each query
        queries = [' '.join([word for word in word_tokenize(q) if word.lower() not in stop_words]) for q in queries]
        print("queries", queries)
        return queries


        




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
            
            # Try to download full text
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

    def create_retry_session(self, retries=3, backoff_factor=0.3, status_forcelist=(500, 502, 504)):
        session = requests.Session()
        retry = Retry(total=retries, read=retries, connect=retries,
                      backoff_factor=backoff_factor, status_forcelist=status_forcelist)
        adapter = HTTPAdapter(max_retries=retry)
        session.mount('http://', adapter)
        session.mount('https://', adapter)
        return session

    def download_pubmed_fulltext(self, pmid):
        try:
            pmc_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id={pmid}&retmode=xml"
            response = self.session.get(pmc_url, timeout=30)
            response.raise_for_status()
            
            try:
                root = ET.fromstring(response.content)
            except ET.ParseError:
                logger.error(f"Failed to parse XML for PMID {pmid}. Response content: {response.content[:100]}...")
                return "Error parsing XML response"

            pmc_id = root.find(".//ArticleId[@IdType='pmc']")
            
            if pmc_id is not None:
                pmc_id = pmc_id.text
                fulltext_url = f"https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi/BioC_xml/{pmc_id}/unicode"
                fulltext_response = self.session.get(fulltext_url, timeout=30)
                fulltext_response.raise_for_status()
                
                try:
                    fulltext_root = ET.fromstring(fulltext_response.content)
                    passages = fulltext_root.findall(".//passage/text")
                    full_text = "\n".join([p.text for p in passages if p.text])
                    return full_text
                except ET.ParseError:
                    logger.error(f"Failed to parse full text XML for PMC ID {pmc_id}")
            
            # If PMC ID is not available or full text couldn't be fetched, try to get PDF
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
                    
                    # Download PDF and extract text
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
            
            # Create a BytesIO object from the content
            pdf_file = BytesIO(response.content)
            
            # Extract text from PDF
            reader = PdfReader(pdf_file)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
            
            return text
        except Exception as e:
            logger.error(f"Error downloading or processing PDF from {pdf_url}: {str(e)}")
            return "Error downloading or processing PDF"
        
    
    

        
    def run_pipeline(self, query):
        try:
            # Generate multiple search queries
            search_queries = self.generate_search_queries(query)
            print("Generated search queries:", search_queries)
            
            all_results = []
            for search_query in search_queries:
                try:
                    # Search for articles
                    pubmed_results = self.search_pubmed(search_query)
                    print(f"\nPubMed results for '{search_query}': {len(pubmed_results)}")
                    time.sleep(1)  # Rate limiting
                    arxiv_results = self.search_arxiv(search_query)
                    print(f"\narXiv results for '{search_query}': {len(arxiv_results)}")
                    all_results.extend(pubmed_results + arxiv_results)
                except Exception as e:
                    logger.error(f"\nError during search for query '{search_query}': {str(e)}")
            
            print(f"\nTotal results found: {len(all_results)}")
            
            if not all_results:
                logger.warning("\nNo results found from any source.")
                return "\nUnable to generate a hypothesis due to lack of search results."
            
            # Process and vectorize results
            self.process_search_results(all_results)
            self._update_vector_store()
            print("vector store updated")
            try:
                print("about to retrieve from databases")
                articles_context = self.retrieve_from_articles_db(query)
                omics_context = self.retrieve_from_omics_db(query)
                print("contexts retrieved")
                hypothesis = self.formulate_hypothesis(query, articles_context, omics_context)
                print("hypothesis formulated")
                print("hypothesis", hypothesis)
                return hypothesis
            except Exception as e:
                logger.error(f"Error generating hypothesis: {str(e)}")
                raise  # This will trigger a retry
        
        except Exception as e:
            logger.error(f"\nAn error occurred during the pipeline execution: {str(e)}")
            return "\nAn error occurred while generating the hypothesis. Please try again later."
        
        
        
        
# Example usage
if __name__ == "__main__":
    system = ComprehensiveHypothesisSystem(
        dataset_path_articles=r"C:\Users\wes\AutoProtGenerationSystem\Phase_1\hypothesis_db_articles",
        dataset_path_omics=r"C:\Users\wes\vectordb_data_good\data\data\omics_vector_store",
        articles_folder=r"C:\Users\wes\AutoProtGenerationSystem\Phase_1\articles"
    )
    print("Articles Dataset path:", system.dataset_path_articles)
    print("Omics Dataset path:", system.dataset_path_omics)
    print("Articles folder:", system.articles_folder)
    query = "novel cellular regeneration and cell longevity to reverse aging"
    hypothesis = system.run_pipeline(query)
    print(f"\nGenerated Hypothesis:\n{hypothesis}")