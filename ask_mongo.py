# ---------------------------------------------------
# Version: 12.02.2025
# Author: M. Weber
# ---------------------------------------------------
# 09.02.2025 added chunks
# 12.02.2025 added parameter for sorting search results
# ---------------------------------------------------

from datetime import datetime
import os
from dotenv import load_dotenv

import ask_llm

from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError

import torch
from transformers import BertTokenizer, BertModel

# Init LLM ----------------------------------
llm = ask_llm.LLMHandler(llm="gemini", local=False)
# llm = ask_llm.LLMHandler(llm="llama3", local=True)

# Init MongoDB Client
load_dotenv()
mongoClient = MongoClient(os.environ.get('MONGO_URI_PV'))
database = mongoClient.pv_data_db
coll_ausgaben = database.ausgaben
coll_artikel = database.artikel
coll_config = database.config

# Load pre-trained model and tokenizer
os.environ["TOKENIZERS_PARALLELISM"] = "false"
model_name = "bert-base-german-cased" # 768 dimensions
# model_name = "bert-base-multilingual-cased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)
# model_name = "sentence-transformers/all-MiniLM-L6-v2"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModel.from_pretrained(model_name)

# Define Database functions ----------------------------------
def generate_abstracts(input_field: str, output_field: str, max_iterations: int = 20) -> None:
    """
    Generates abstracts for documents in the 'ausgaben' collection.
    
    Args:
        input_field (str): The field containing the text to summarize.
        output_field (str): The field to store the generated abstract.
        max_iterations (int, optional): Maximum number of documents to process. Defaults to 20.
    """
    cursor = coll_ausgaben.find({output_field: ""}).limit(max_iterations)
    cursor_list = list(cursor)
    for record in cursor_list:
        abstract = write_summary(str(record[input_field]))
        print(record['titel'][:50])
        print("-"*50)
        coll_ausgaben.update_one({"_id": record.get('_id')}, {"$set": {output_field: abstract}})
    cursor.close()

def write_summary(text: str = "", length: int = 500) -> str:
    """
    Writes a summary for the given text.
    
    Args:
        text (str, optional): The text to summarize. Defaults to "".
        length (int, optional): Maximum length of the summary in words. Defaults to 500.
    
    Returns:
        str: The generated summary.
    """
    if text == "":
        return "empty"
    system_prompt = f"""
                    Du bist ein Redakteur im Bereich Medien mit Schwerpunkt Vertrieb und Digitalisierung im Bereich Tageszeitungen.
                    Du bis Experte dafür, Zusammenfassungen von Fachartikeln zu schreiben.
                    Die maximale Länge der Zusammenfassungen sind {length} Wörter.
                    Wichtig ist nicht die Lesbarkeit, sondern die Kürze und Prägnanz der Zusammenfassung:
                    Was sind die wichtigsten Aussagen und Informationen des Textes?
                    """
    task = """
            Erstelle eine Zusammenfassung des Originaltextes in deutscher Sprache.
            Verwende keine Zeilenumrüche oder Absätze.
            Die Antwort darf nur aus dem eigentlichen Text der Zusammenfassung bestehen.
            """
    return llm.ask_llm(temperature=0.1, question=task, system_prompt=system_prompt, db_results_str=text)
    
def write_takeaways(text: str = "", max_takeaways: int = 5) -> str:
    """
    Writes the key takeaways for the given text.
    
    Args:
        text (str, optional): The text to extract takeaways from. Defaults to "".
        max_takeaways (int, optional): Maximum number of takeaways. Defaults to 5.
    
    Returns:
        str: The generated takeaways.
    """
    if text == "":
        return "empty"
    system_prompt = """
                    Du bist ein Redakteur im Bereich Medien mit Schwerpunkt Vertrieb und Digitalisierung im Bereich Tageszeitungen.
                    Du bis Experte dafür, die wichtigsten Aussagen von Fachartikeln herauszuarbeiten.
                    """
    task = f"""
            Erstelle eine Liste der wichtigsten Aussagen des Textes in deutscher Sprache.
            Es sollten maximal {max_takeaways} Aussagen sein.
            Jede Aussage sollte kurz und prägnant in einem eigenen Satz formuliert sein.
            Die Antwort darf nur aus den eigentlichen Aussagen bestehen.
            """
    return llm.ask_llm(temperature=0.1, question=task, system_prompt=system_prompt, db_results_str=text)

# Chunks ------------------------------------------------
def chunk_text_to_dataframe(text, chunk_size, overlap=0) -> list:
    """
    Splits a text into chunks and stores them in a list.

    Args:
        text: The input text string.
        chunk_size: The desired size of each chunk (number of characters).
        overlap: The number of overlapping characters between chunks. Defaults to 0 (no overlap).

    Returns:
        list: A list where each element represents a chunk of text. Returns None if the input text is None or empty.
    """

    if not text:  # Handle None or empty input
        return None

    if chunk_size <= 0:
      return None

    if overlap < 0 or overlap >= chunk_size:
      return None

    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))  # Ensure we don't go past the end of the text
        chunk = {'start': start, 'end': end, 'text': text[start:end]}
        chunks.append(chunk)
        start += chunk_size - overlap # Move start for the next chunk, accounting for overlap

    return chunks

# Embeddings -------------------------------------------------            
def generate_embeddings(input_field: str, output_field: str, max_iterations: int = 10) -> None:
    """
    Generates embeddings for documents in the 'ausgaben' collection.
    
    Args:
        input_field (str): The field containing the text to generate embeddings for.
        output_field (str): The field to store the generated embeddings.
        max_iterations (int, optional): Maximum number of documents to process. Defaults to 10.
    """
    cursor = coll_ausgaben.find({output_field: []}).limit(max_iterations)
    cursor_list = list(cursor)
    for record in cursor_list:
        article_text = record[input_field]
        if article_text == "":
            article_text = "Fehler: Kein Text vorhanden."
        else:
            embeddings = create_embeddings(text=article_text)
            coll_ausgaben.update_one({"_id": record['_id']}, {"$set": {output_field: embeddings}})
    print(f"\nGenerated embeddings for {max_iterations} records.")

def create_embeddings(text: str) -> list:
    """
    Creates embeddings for the given text using a pre-trained BERT model.
    
    Args:
        text (str): The text to generate embeddings for.
    
    Returns:
        list: The generated embeddings.
    """
    encoded_input = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        model_output = model(**encoded_input)
    return model_output.last_hidden_state.mean(dim=1).squeeze().tolist()

# Keywords ---------------------------------------------------
def generate_keywords(input_field: str, output_field: str, max_iterations: int = 10) -> None:
    """
    Generates keywords for documents in the 'ausgaben' collection.
    
    Args:
        input_field (str): The field containing the text to generate keywords for.
        output_field (str): The field to store the generated keywords.
        max_iterations (int, optional): Maximum number of documents to process. Defaults to 10.
    """
    print(f"Start: {input_field}|{output_field}")
    print(collection)
    cursor = coll_ausgaben.find({output_field: []}).limit(max_iterations)
    if cursor:
        print(f"MongoDB Suche abgeschlossen.")
        cursor_list = list(cursor)
        print(f"Anzahl Records: {len(cursor_list)}")
        for record in cursor_list:
            if record[input_field] == "":
                print("Kein Input-Text.")
                continue
            article_text = record.get(input_field, "Fehler: Kein Text vorhanden.")
            keywords = create_keywords(text=article_text)
            coll_ausgaben.update_one({"_id": record['_id']}, {"$set": {output_field: keywords}})
            print(keywords)
        print(f"\nGenerated keywords for {len(cursor_list)} records.")
    else:
        st.error("No articles without summary found.")
    cursor.close()

def create_keywords(text: str = "", max_keywords: int = 5) -> list:
    """
    Creates keywords for the given text.
    
    Args:
        text (str, optional): The text to generate keywords for. Defaults to "".
        max_keywords (int, optional): Maximum number of keywords. Defaults to 5.
    
    Returns:
        list: The generated keywords.
    """
    if not text:
        return []
    system_prompt = """
                    Du bist ein Redakteur im Bereich Medien mit Schwerpunkt Vertrieb und Digitalisierung im Bereich Tageszeitungen.
                    Du bis Experte dafür, relevante Schlagwörter für die Inhalte von Fachartikeln zu schreiben.
                    """
    task = f"""
            Erstelle Schlagworte für den folgenden Text angegebenen Text.
            Erstelle maximal {max_keywords} Schlagworte.
            Die Antwort darf nur aus den eigentlichen Schlagworten bestehen.
            Das Format ist "Stichwort1, Stichwort2, Stichwort3, ..."
            """
    keywords_str = llm.ask_llm(temperature=0.1, question=task, system_prompt=system_prompt, db_results_str=text)
    keywords_list = [keyword.strip() for keyword in keywords_str.split(',') if keyword.strip()]
    return keywords_list

def list_keywords() -> list:
    """
    Lists all keywords in the 'ausgaben' collection.
    
    Returns:
        list: A list of keywords with their counts.
    """
    pipeline = [
    {'$unwind': '$schlagworte'},
    {'$group': {
        '_id': '$schlagworte', 
        'count': {'$sum': 1}
        }
        },
    {'$sort': {'count': -1}},
    {'$project': {
        '_id': 0, 
        'keyword': '$_id', 
        'count': 1
        }
        }
    ]
    cursor_list = list(coll_ausgaben.aggregate(pipeline))
    return cursor_list

# Query & Filter ------------------------------------------------
def generate_query(question: str = "") -> str:
    """
    Generates search keywords based on the given question.
    
    Args:
        question (str, optional): The question to generate keywords for. Defaults to "".
    
    Returns:
        str: The generated search keywords.
    """
    task = f"""
            Erstelle auf Basis der Frage '{question}' eine Liste von maximal 3 Schlagworten mit deren Hilfe relevante Dokumente zu der Fragestellung in einer Datenbank gefunden werden können.
            Das Format ist "Stichwort1" "Stichwort2" "Stichwort3"
            """
    return llm.ask_llm(temperature=0.1, question=task) 
    
def generate_filter(filter: list, field: str) -> dict:
    """
    Generates a MongoDB filter based on the given list of values.
    
    Args:
        filter (list): The list of values to filter by.
        field (str): The field to filter on.
    
    Returns:
        dict: The generated MongoDB filter.
    """
    return {field: {"$in": filter}} if filter else {}

# Search ------------------------------------------------
def fulltext_search_ausgaben(search_text: str = "*", gen_suchworte: bool = False, sort: str = "score", score: float = 0.0, limit: int = 10) -> (list, str):
    """
    Performs a full-text search on MongoDB collection of issues (Ausgaben).
    This function searches through documents using MongoDB's Atlas Search functionality.
    It can either use the provided search text directly or generate search terms from
    a natural language question.
    Args:
        search_text (str, optional): Text to search for. Use "*" for matching all documents.
            Defaults to "*".
        gen_suchworte (bool, optional): If True, generates search terms from the search_text
            using the generate_query function. Defaults to False.
        sort (str, optional): Field to sort results by. Defaults to "score".
        score (float, optional): Minimum search score threshold for results. 
            Defaults to 0.0.
        limit (int, optional): Maximum number of results to return. Defaults to 10.
    Returns:
        tuple: A tuple containing:
            - list: List of matching documents, each containing fields:
                * _id: Document ID
                * doknr: Document number
                * jahrgang: Year
                * ausgabe: Issue number
                * text: Document text content
                * score: Search relevance score
            - str: The actual search terms used (either original search_text or generated terms)
    Examples:
        >>> results, terms = fulltext_search_ausgaben("climate change", gen_suchworte=True)
        >>> results, terms = fulltext_search_ausgaben("*")  # Match all documents
        >>> results, terms = fulltext_search_ausgaben("energy", score=0.5, limit=20)
    """
    
    # define query ------------------------------------------------
    if search_text == "":
        return [], ""
    if search_text == "*":
        suchworte = "*"
        score = 0.0
        query = {
            "index": "volltext",
            "exists": {"path": "doknr"},
        }
    else:
        suchworte = generate_query(question=search_text) if gen_suchworte else search_text
        query = {
            "index": "volltext",
            "text": {
                "query": suchworte,
                "path": {"wildcard": "*"}
            }
        }

    # define fields ------------------------------------------------
    fields = {
        "_id": 1,
        "doknr": 1,
        "jahrgang": 1,
        "ausgabe": 1,
        "text": 1,
        "score": {"$meta": "searchScore"},
    }

    # define pipeline ------------------------------------------------
    pipeline = [
        {"$search": query},
        {"$project": fields},
        {"$match": {"score": {"$gte": score}}},
        {"$sort": {sort: -1}},
        {"$limit": limit},
    ]

    # execute query ------------------------------------------------
    cursor = coll_ausgaben.aggregate(pipeline)
    return list(cursor), suchworte


def fulltext_search_artikel(search_text: str = "*", gen_suchworte: bool = False, sort: str = "score", score: float = 0.0, limit: int = 10) -> (list, str):
    """
    Performs a fulltext search on articles in MongoDB collection.
    This function searches through articles using MongoDB's text search capabilities. It can either use
    the provided search text directly or generate search terms based on the input question.
    Args:
        search_text (str, optional): Text to search for. Use "*" for matching all documents. Defaults to "*".
        gen_suchworte (bool, optional): Whether to generate search terms from the question. Defaults to False.
        sort (str, optional): Field to sort results by. Defaults to "score".
        score (float, optional): Minimum search score threshold. Defaults to 0.0.
        limit (int, optional): Maximum number of results to return. Defaults to 10.
    Returns:
        tuple: A tuple containing:
            - list: List of matching documents, each containing _id, doknr, start, ende, text and score
            - str: The search terms used (either original search_text or generated terms)
    Example:
        >>> results, terms = fulltext_search_artikel("example search", gen_suchworte=True)
        >>> results, terms = fulltext_search_artikel("*")  # Match all documents
    """
    
    # define query ------------------------------------------------
    if search_text == "":
        return [], ""
    if search_text == "*":
        suchworte = "*"
        score = 0.0
        query = {
            "index": "fulltext_text",
            "exists": {"path": "doknr"},
        }
    else:
        suchworte = generate_query(question=search_text) if gen_suchworte else search_text
        query = {
            "index": "fulltext_text",
            "text": {
                "query": suchworte,
                "path": {"wildcard": "*"}
            }
        }

    # define fields ------------------------------------------------
    fields = {
        "_id": 1,
        "doknr": 1,
        "start": 1,
        "ende": 1,
        "text": 1,
        "score": {"$meta": "searchScore"},
    }

    # define pipeline ------------------------------------------------
    pipeline = [
        {"$search": query},
        {"$project": fields},
        {"$match": {"score": {"$gte": score}}},
        {"$sort": {sort: -1}},
        {"$limit": limit},
    ]

    # execute query ------------------------------------------------
    cursor = coll_artikel.aggregate(pipeline)
    return list(cursor), suchworte


def vector_search(search_text: str = "*", gen_suchworte: bool = False, sort: str = "score", score: float = 0.0, limit: int = 10 ,filter : list = []) -> list[list, str]:
    """
    Performs a vector search on a MongoDB collection using text embeddings.
    This function searches through documents using vector similarity, optionally generating search keywords
    from the input text. It returns a list of matching documents and the search terms used.
    Args:
        search_text (str, optional): Text to search for. Defaults to "*".
        gen_suchworte (bool, optional): Whether to generate search keywords from search_text. Defaults to False.
        sort (str, optional): Field to sort results by. Defaults to "score".
        score (float, optional): Minimum similarity score threshold. Defaults to 0.0.
        limit (int, optional): Maximum number of results to return. Defaults to 10.
        filter (list, optional): List of source IDs to filter results. Defaults to empty list.
    Returns:
        tuple[list, str]: A tuple containing:
            - list: Matching documents with their metadata and similarity scores
            - str: The search terms used (either original or generated)
    Example:
        >>> results, terms = vector_search("machine learning", gen_suchworte=True, limit=5)
        >>> print(f"Found {len(results)} matches using terms: {terms}")
    """
    
    # define query ------------------------------------------------
    suchworte = generate_query(question=search_text) if gen_suchworte else search_text
    embeddings_query = create_embeddings(text=suchworte)
    query = {
            "index": "vector",
            "path": "embeddings",
            "queryVector": embeddings_query,
            "numCandidates": int(limit * 10),
            "limit": limit,
            }
    
    # define fields ------------------------------------------------
    fields = {
            "_id": 1,
            "doknr": 1,
            "jahrgang": 1,
            "ausgabe": 1,
            "text": 1,
            "score": {"$meta": "vectorSearchScore"}
            }
    
    # define pipeline ------------------------------------------------
    pipeline = [
        {"$vectorSearch": query},
        {"$project": fields},
        # {"$match": {"quelle_id": {"$in": filter}}},
        {"$match": {"score": {"$gte": score}}},  # Move this up
        {"$sort": {sort: -1}},
        {"$limit": limit},  # Add this stage
    ]

    # execute query ------------------------------------------------
    cursor = coll_artikel.aggregate(pipeline)
    return list(cursor), suchworte

# Diff ------------------------------------------------
def group_by_field() -> dict:
    """
    Groups documents by the 'quelle_id' field and counts the number of documents in each group.
    
    Returns:
        dict: A dictionary where keys are 'quelle_id' values and values are the counts of documents.
    """
    pipeline = [
            {   
            '$group': {
                '_id': '$quelle_id', 
                'count': {
                    '$sum': 1
                    }
                }
            }, {
            '$sort': {
                'count': -1
                }
            }
            ]
    result = coll_ausgaben.aggregate(pipeline)
    # transfor into dict
    return_dict = {}
    for item in result:
        return_dict[item['_id']] = item['count']
    return return_dict

def list_fields() -> dict:
    """
    Lists all fields in a document from the 'ausgaben' collection.
    
    Returns:
        dict: A dictionary of field names.
    """
    result = coll_ausgaben.find_one()
    return result.keys()

def get_document(id: str) -> dict:
    """
    Retrieves a document from the 'ausgaben' collection by its ID.
    
    Args:
        id (str): The ID of the document to retrieve.
    
    Returns:
        dict: The retrieved document.
    """
    document = coll_ausgaben.find_one({"id": id})
    return document


# Config ------------------------------------------------

def get_system_prompt() -> str:
    """
    Retrieves the system prompt from the 'config' collection.
    
    Returns:
        str: The system prompt.
    """
    result = coll_config.find_one({"key": "systemprompt"})
    return str(result["content"])
    
def update_system_prompt(text: str = ""):
    """
    Updates the system prompt in the 'config' collection.
    
    Args:
        text (str, optional): The new system prompt. Defaults to "".
    """
    result = coll_config.update_one({"key": "systemprompt"}, {"$set": {"content": text}})
