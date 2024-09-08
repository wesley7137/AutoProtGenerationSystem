from research_and_summarize_utils import search_academic_sources, extract_key_information, summarize_text, generate_description
from Phase_1.scripts.AutonomousHypothesisSystem import AutonomousHypothesisSystem
import os
import json
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import string

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


hypothesis_system = AutonomousHypothesisSystem()


def generate_initial_hypothesis(topic):
    """Generate an initial hypothesis based on the user-defined topic."""
    hypothesis = hypothesis_system.hypothesis_generator.generate_hypothesis(topic)
    print(f"Initial hypothesis generated: {hypothesis}")
    return hypothesis


def run_research_pipeline(keywords, max_results=5):
    """Run the research pipeline to collect and summarize relevant articles."""
    articles = search_academic_sources(keywords, max_results)
    if not articles:
        print("No articles found.")
        return []

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

    return summarized_articles

def extract_keywords(initial_hypothesis, num_keywords=5):
    """
    Extract keywords from the initial hypothesis using TF-IDF and lemmatization.
    
    Args:
    initial_hypothesis (str): The initial hypothesis text
    num_keywords (int): Number of top keywords to extract
    
    Returns:
    list: Top keywords extracted from the hypothesis
    """
    # Tokenize and lowercase the hypothesis
    tokens = word_tokenize(initial_hypothesis.lower())
    
    # Remove punctuation and stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in string.punctuation and token not in stop_words]
    
    # Lemmatize the tokens
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    # Join the lemmatized tokens back into a string
    processed_hypothesis = ' '.join(lemmatized_tokens)
    
    # Use TF-IDF to identify important words
    vectorizer = TfidfVectorizer(max_features=num_keywords)
    tfidf_matrix = vectorizer.fit_transform([processed_hypothesis])
    
    # Get feature names (words) and their TF-IDF scores
    feature_names = vectorizer.get_feature_names_out()
    tfidf_scores = tfidf_matrix.toarray()[0]
    
    # Sort words by TF-IDF score and select top keywords
    word_scores = list(zip(feature_names, tfidf_scores))
    word_scores.sort(key=lambda x: x[1], reverse=True)
    keywords = [word for word, score in word_scores[:num_keywords]]
    
    return keywords


def prepare_technical_description(refined_hypothesis, research_data):
    """Prepare the final technical description by concatenating the refined hypothesis and research data."""
    research_descriptions = "\n".join([article['generated_description'] for article in research_data])
    technical_description = f"Refined Hypothesis:\n{refined_hypothesis}\n\nResearch Insights:\n{research_descriptions}"
    return technical_description

def run_research_and_hypothesize_pipeline (user_input):
    # Step 2: Generate Initial Hypothesis
    initial_hypothesis = generate_initial_hypothesis(user_input)
    keywords = extract_keywords(initial_hypothesis)
    # Step 3: Research Hypotheses
    research_data = run_research_pipeline(keywords, max_results=5)
    if not research_data:
        print("No relevant articles found to refine hypotheses.")
        return

    # Step 4: Refine Hypothesis
    # Step 5: Prepare Technical Description
    technical_description = prepare_technical_description(initial_hypothesis, research_data)

    # Step 6: Output Technical Description
    print("\nFinal Technical Description for Sequence Generation:")
    print(technical_description)

    # Save the final technical description to a file if needed
    os.makedirs("final_output", exist_ok=True)
    with open("final_output/technical_description.txt", "w") as f:
        f.write(technical_description)

    print("Pipeline execution completed.")

if __name__ == "__main__":
    main()
