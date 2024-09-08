import spacy
from collections import Counter
from langchain_community.vectorstores import DeepLake
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain.schema import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class LiteratureAnalyzer:
    def __init__(self):
        self.documents = []
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        self.vector_store = None
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.vectorizer = TfidfVectorizer()
        self.nlp = spacy.load("en_core_web_sm")

    def add_document(self, text):
        self.documents.append(text)
        if len(self.documents) % 10 == 0:  # Update vector store every 10 documents
            self._update_vector_store()

    def _update_vector_store(self):
        text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        docs = [Document(page_content=text) for text in self.documents]
        split_docs = text_splitter.split_documents(docs)
        self.vector_store = DeepLake.from_documents(
            split_docs,
            self.embeddings,
            dataset_path="./vector_db",
            overwrite=True
        )

    def extract_key_concepts(self):
        all_entities = []
        for doc in self.nlp.pipe(self.documents):
            entities = [(ent.text, ent.label_) for ent in doc.ents]
            all_entities.extend(entities)
        
        # Count occurrences of each entity
        entity_counts = Counter(all_entities)
        
        # Group entities by type
        grouped_entities = {}
        for (entity, label), count in entity_counts.items():
            if label not in grouped_entities:
                grouped_entities[label] = []
            grouped_entities[label].append((entity, count))
        
        # Sort entities in each group by count
        for label in grouped_entities:
            grouped_entities[label].sort(key=lambda x: x[1], reverse=True)
        
        return grouped_entities

    def extract_entities_and_relations(self, text):
        doc = self.nlp(text)
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        relations = []
        for token in doc:
            if token.dep_ in ("nsubj", "dobj", "pobj"):
                subject = token.text
                relation = token.head.text
                object_ = [child for child in token.head.children if child.dep_ in ("dobj", "pobj")]
                object_ = object_[0].text if object_ else ""
                if object_:
                    relations.append((subject, relation, object_))
        return entities, relations

    def search_similar_documents(self, query, k=5):
        if not self.vector_store:
            return []
        results = self.vector_store.similarity_search(query, k=k)
        return [doc.page_content for doc in results]

    def assess_relevance(self, hypothesis):
        if not self.documents:
            return 0.5  # Default score if no documents are available
        corpus_embeddings = self.model.encode(self.documents)
        hypothesis_embedding = self.model.encode([hypothesis])
        similarities = cosine_similarity(hypothesis_embedding, corpus_embeddings)[0]
        return round(float(np.mean(similarities)), 2)

    def assess_novelty(self, hypothesis):
        if not self.documents:
            return 1.0  # Assume maximum novelty if there are no documents to compare against
        all_texts = self.documents + [hypothesis]
        tfidf_matrix = self.vectorizer.fit_transform(all_texts)
        if tfidf_matrix.shape[0] < 2:
            return 1.0  # Assume maximum novelty if there's only the hypothesis
        cosine_similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
        avg_similarity = np.mean(cosine_similarities)
        novelty_score = 1 - avg_similarity  # Higher novelty for lower similarity
        return round(novelty_score, 2)