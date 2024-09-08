import pandas as pd
from sklearn.preprocessing import StandardScaler
from rdflib import Graph, Namespace, RDF, URIRef, Literal
from rdflib.namespace import RDFS, OWL, XSD
import uuid

#Phase 3: Data Integration
def harmonize_data(dataframes):
    """
    Harmonizes data from multiple sources into a single, consistent format.

    :param dataframes: List of pandas DataFrames to harmonize.
    :return: Harmonized pandas DataFrame.
    """
    harmonized_df = pd.concat(dataframes, ignore_index=True)
    
    # Rename columns to ensure consistency
    column_mapping = {
        'authors': 'Author',
        'title': 'Title',
        'abstract': 'Abstract',
        'journal': 'Journal',
        'doi': 'DOI',
        'url': 'URL',
        'publication_date': 'PublicationDate',
        'year': 'Year',
        'source': 'Source'
    }
    
    harmonized_df.rename(columns=column_mapping, inplace=True)
    
    # Ensure all expected columns exist, fill with 'N/A' if missing
    for col in column_mapping.values():
        if col not in harmonized_df.columns:
            harmonized_df[col] = 'N/A'
    
    return harmonized_df


def ontology_mapping(data_frame):
    """ Maps data to an ontology using RDFLib to link related concepts.
    
    :param data_frame: The harmonized pandas DataFrame.
    :return: RDF graph with mapped ontology.
    """
    # Define namespaces
    RESAI = Namespace("http://researchassistai.org/ontology/")
    BIO = Namespace("http://bioportal.bioontology.org/ontologies/")

    # Create an RDF graph
    graph = Graph()
    graph.bind("resai", RESAI)
    graph.bind("bio", BIO)

    for _, row in data_frame.iterrows():
        # Generate a unique URI for each article
        article_uri = URIRef(RESAI[f"article/{uuid.uuid4()}"])

        # Add triples to the graph, checking for column existence and handling potential missing data
        graph.add((article_uri, RDF.type, RESAI.Article))

        if 'Title' in data_frame.columns and pd.notna(row["Title"]):
            graph.add((article_uri, RESAI.title, Literal(row["Title"])))

        if 'Author' in data_frame.columns and pd.notna(row["Author"]):
            authors = row["Author"] if isinstance(row["Author"], list) else [row["Author"]]
            for author in authors:
                graph.add((article_uri, RESAI.author, Literal(author)))

        if 'Abstract' in data_frame.columns and pd.notna(row["Abstract"]):
            graph.add((article_uri, RESAI.abstract, Literal(row["Abstract"])))

        if 'PublicationDate' in data_frame.columns and pd.notna(row["PublicationDate"]):
            try:
                graph.add((article_uri, RESAI.publicationDate, Literal(row["PublicationDate"], datatype=XSD.date)))
            except ValueError:
                # If date parsing fails, add it as a string
                graph.add((article_uri, RESAI.publicationDate, Literal(str(row["PublicationDate"]))))

        if 'Journal' in data_frame.columns and pd.notna(row["Journal"]):
            graph.add((article_uri, RESAI.journal, Literal(row["Journal"])))

        if 'DOI' in data_frame.columns and pd.notna(row["DOI"]):
            graph.add((article_uri, RESAI.doi, Literal(row["DOI"])))

        # Example linking to external ontology (BioPortal)
        if 'Abstract' in data_frame.columns and pd.notna(row["Abstract"]) and "cancer" in row["Abstract"].lower():
            graph.add((article_uri, RDFS.seeAlso, BIO.Cancer))

    # Enrich graph with ontological relationships
    graph.add((RESAI.Article, OWL.sameAs, RESAI.ResearchPaper))
    graph.add((RESAI.Article, OWL.subClassOf, RESAI.Publication))

    return graph

