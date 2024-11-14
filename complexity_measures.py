from ast import Tuple
from calendar import c
from curses import nl
import inspect
import os
import re
import math
import pickle
import string
import sys
from tkinter import BOTH
from typing import Dict, List
from cv2 import merge
from flask import g
from pydantic import BaseModel, Field
from regex import P, sub
import spacy
import networkx as nx
import nltk
import scipy.sparse as sp
from collections import Counter, defaultdict
import sympy
from sympy import N
from textblob import TextBlob
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize, sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import openai
from nltk.corpus import cmudict
from transformers import pipeline
import pandas as pd
import datetime as dt
from textstat import flesch_reading_ease


# Initialize the Count Vectorizer
date_str = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
save_path = f"debug_log_complexity_{date_str}.txt"

client = openai.OpenAI()
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    print("[Language Model] OpenAI API key not found.")
    raise ValueError("OpenAI API key not found.")
openai.api_key = openai_api_key

import numpy as np
import logging
from functools import lru_cache


# Set up logging
logging.basicConfig(level=logging.DEBUG, filename="logging_" + save_path, filemode="w")


import scipy.sparse as sp
import numpy as np
import logging


import numpy as np
import logging
from typing import Tuple


def cosine_similarity_custom(vector_a, matrix_b):
    """
    Calculate the cosine similarity between a single vector and either another single vector
    or multiple vectors (rows of a matrix).

    Parameters:
        vector_a (list, numpy array, or scipy.sparse._csr.csr_matrix): The first normalized vector.
        matrix_b (list, numpy array, or scipy.sparse._csr.csr_matrix): The matrix of vectors to compare against,
                                                                      or a single vector.

    Returns:
        numpy array or float: The cosine similarities between the input vector and each row of the matrix,
                              or a single value if comparing two vectors.
    """
    # Input validation
    if not isinstance(vector_a, (list, np.ndarray, sp.csr_matrix)) or not isinstance(
        matrix_b, (list, np.ndarray, sp.csr_matrix)
    ):
        raise TypeError(
            f"Input must be either lists, numpy arrays, or scipy sparse matrices. {type(vector_a)}, {type(matrix_b)}"
        )

    # Convert sparse matrices to numpy arrays if needed
    if isinstance(vector_a, sp.csr_matrix):
        vector_a = vector_a.toarray().flatten()
    if isinstance(matrix_b, sp.csr_matrix):
        matrix_b = matrix_b.toarray()

    # Ensure vectors are numpy arrays
    vector_a = np.array(vector_a, dtype=np.float32)
    matrix_b = np.array(matrix_b, dtype=np.float32)

    # Ensure matrix_b is a 2D array
    if matrix_b.ndim == 1:
        # If matrix_b is a 1D vector, reshape it to be a matrix with one row
        matrix_b = matrix_b.reshape(1, -1)

    # Debug statements to check the shapes
    print(f"Shape of vector_a: {vector_a.shape}")
    print(f"Shape of matrix_b: {matrix_b.shape}")

    # Check if vector and matrix have compatible dimensions
    if vector_a.shape[0] != matrix_b.shape[1]:
        raise ValueError(
            f"Vector and matrix are of mismatched dimensions: {vector_a.shape[0]} != {matrix_b.shape[1]}. "
            f"Please make sure the vector and matrix have compatible dimensions."
        )

    # Calculate cosine similarity
    try:
        # Calculate dot product between vector_a and each row of matrix_b
        dot_product = np.dot(matrix_b, vector_a)

        # Calculate magnitudes
        magnitude_a = np.linalg.norm(vector_a)
        magnitude_b = np.linalg.norm(matrix_b, axis=1)

        # Ensure magnitudes are not zero to avoid division errors
        if magnitude_a == 0 or np.any(magnitude_b == 0):
            raise ValueError(
                "One or more of the vectors have zero magnitude, cannot compute cosine similarity."
            )

        # Calculate cosine similarity
        cosine_similarity_values = dot_product / (magnitude_a * magnitude_b)

        # If the original input was a single vector, return a single float value
        if cosine_similarity_values.shape[0] == 1:
            return cosine_similarity_values[0]
        else:
            return cosine_similarity_values
    except Exception as e:
        logging.error(
            f"An error occurred while calculating cosine similarity: {str(e)}"
        )
        raise ValueError(
            f"An error occurred while calculating cosine similarity: {str(e)}"
        )


# Example usage
vector_a = [0.5, 0.2, 0.3]
matrix_b = [[0.4, 0.8, 0.1], [0.2, 0.1, 0.9], [0.5, 0.5, 0.5]]

try:
    similarities = cosine_similarity_custom(vector_a, matrix_b)
    print("Cosine Similarities:", similarities)
except ValueError as e:
    print(f"Error: {e}")


@lru_cache(maxsize=2048)
def get_embedding(text, model="text-embedding-3-small"):
    """
    Generate an embedding for the given text using OpenAI's embedding model.

    Parameters:
        text (str): The input text string to embed.
        model (str): The model to use for generating the embedding.

    Returns:
        list: The embedding vector for the input text.
    """
    # Input validation
    if not isinstance(text, str) or not text.strip():
        raise ValueError(
            f"Input text must be a non-empty string. Found type: {type(text)} and value: {text}"
        )
    if not isinstance(model, str) or not model.strip():
        raise ValueError(
            "Model name must be a non-empty string. Found type: {type(model)} and value: {model}"
        )

    # Log the request
    logging.info(
        f"Requesting embedding for text: '{text[:20]}...' using model: '{model}'"
    )
    print(f"Requesting embedding for text: '{text[:20]}...' using model: '{model}'")

    # Fetch embedding
    try:
        response = client.embeddings.create(input=text, model=model)
        if not response or not response.data or (not response.data[0].embedding):
            print(f"Invalid response from the embedding model: {response}")
            raise ValueError("Invalid response from the embedding model.")
        else:
            logging.info(f"Embedding fetched successfully for text: '{text[:20]}...'")
            print(f"Embedding fetched successfully for text: '{text[:20]}...'")
        embedding = response.data[0].embedding
    except Exception as e:
        logging.error(f"An error occurred while fetching embedding: {str(e)}")
        print(
            f"An error occurred while fetching embedding: {str(e)} on line {sys.exc_info()[-1].tb_lineno} in {sys.exc_info()[-1].tb_frame.f_code.co_filename} for text: '{text[:20]}...'. Response: {response}"
        )
        raise ConnectionError(f"An error occurred while fetching embedding: {str(e)}")

    return embedding


# Initialize Hugging Face's SRL pipeline
try:
    srl_pipeline = pipeline(
        "token-classification",
        model="dbmdz/bert-large-cased-finetuned-conll03-english",
    )
except Exception as e:
    print(f"Error loading SRL model: {e}")
    srl_pipeline = None


def jaccard_similarity(str1: str, str2: str) -> float:
    """
    Calculates the Jaccard similarity between two strings.

    Args:
        str1 (str): First string.
        str2 (str): Second string.

    Returns:
        float: Jaccard similarity score.
    """
    # Check for empty strings and for type mismatch
    if not isinstance(str1, str) or not isinstance(str2, str):
        raise ValueError("Input must be strings.")
    set1 = set(str1.split())
    set2 = set(str2.split())
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    return float(len(intersection)) / len(union) if union else 0.0


# Ensure necessary NLTK data is downloaded
nltk_packages = ["punkt", "averaged_perceptron_tagger", "wordnet", "cmudict"]
for package in nltk_packages:
    try:
        nltk.data.find(
            f"tokenizers/{package}" if package == "punkt" else f"corpora/{package}"
        )
    except LookupError:
        nltk.download(package)
nltk.download("averaged_perceptron_tagger_eng")
# Initialize spaCy model
try:
    nlp_spacy = spacy.load("en_core_web_trf")
except OSError:
    from spacy.cli import download

    download("en_core_web_trf")
    nlp_spacy = spacy.load("en_core_web_trf")


def get_srl_spacy(input_query: str):
    doc = nlp_spacy(input_query)

    # Extract predicates (verbs) and arguments using dependency parsing
    predicates = [token for token in doc if token.pos_ == "VERB"]
    arguments = [chunk for chunk in doc.noun_chunks]

    substeps = len(predicates)
    max_depth = len(arguments)

    return substeps, max_depth


import tiktoken


def remove_junk_patterns(text: str) -> str:
    """
    Removes known junk patterns from the text.

    Args:
        text (str): The text to clean.

    Returns:
        str: Cleaned text.
    """
    junk_patterns = [
        r"To solve the problem of.*?:\s*\w+",
        r"By following these steps, .*",
        r"Here is the plan:",
        r"The following steps will guide you through the process:",
        # Add more patterns as needed
    ]
    for pattern in junk_patterns:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)
    return text.strip()


def remove_non_sentences(text: str) -> str:
    """
    Removes non-sentence text from the input text.

    Args:
        text (str): The input text to clean.

    Returns:
        str: The cleaned text containing only full sentences.
    """
    sentences = sent_tokenize(text)
    cleaned_sentences = []
    for sentence in sentences:
        # Check if the sentence ends with a punctuation mark and begins with a capital letter
        if (
            sentence.strip()
            and sentence[-1] in string.punctuation
            and sentence[0].isupper()
        ):
            cleaned_sentences.append(sentence)
    return " ".join(cleaned_sentences)


def is_text_readable(text, readability_threshold=60):
    """
    Determines if the text meets a minimum readability score.

    Args:
        text (str): The text to evaluate.
        readability_threshold (int): The minimum Flesch-Kincaid score.

    Returns:
        bool: True if text is readable, False otherwise.
    """
    # First, remove odd puntuation like emojis, hashtags, etc.
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    score = flesch_reading_ease(text)
    print(f"\nReadability Score: {score}\n\n")
    return score >= readability_threshold


def is_real_words(text) -> bool:
    # first tokenize each word
    words = word_tokenize(text)
    real_words = []
    not_words = []
    for word in words:
        doc = nlp_spacy(word)
        if doc[0].is_alpha and doc[0].has_vector or wn.synsets(word):
            real_words.append(word)
        elif (
            doc[0].is_punct
            or doc[0].is_space
            or doc[0].is_stop
            or doc[0].is_digit
            or doc[0].is_currency
            or doc[0].is_quote
            or doc[0].is_bracket
            or doc[0].is_oov
            or doc[0].is_left_punct
            or doc[0].is_right_punct
        ):
            # Append to real words if it is a punctuation, space, stop word, digit, currency, quote, bracket, or out of vocabulary so that it is not removed
            real_words.append(word)
        else:
            print(f"Word not found: {word}")
            not_words.append(word)

    # Return a new string with the real words in the original order, minus the non-words
    results = " ".join(real_words)
    return results


def count_tokens(text, model_name):
    encoding_name = tiktoken.encoding_name_for_model(model_name)
    encoding = tiktoken.get_encoding(encoding_name)
    tokens = encoding.encode(text)
    return len(tokens)


from nltk.util import ngrams


def generate_ngrams(text, n):
    """
    Generate n-grams from a given text string.

    :param text: Input text string.
    :param n: Number of words in each n-gram.
    :return: List of n-grams as tuples.
    """
    tokens = word_tokenize(text.lower())  # Tokenize and convert to lowercase
    return list(ngrams(tokens, n))


# Initialize Machine Learning Model for Classification
model_path_ml = "ml_complexity_model.pkl"
if os.path.exists(model_path_ml):
    with open(model_path_ml, "rb") as f:
        pipeline_ml = pickle.load(f)
else:
    # Create and train a simple mock model
    import pandas as pd

    df_ml = pd.DataFrame(
        {
            "query": [
                "What is the capital of France?",
                "Explain the process of photosynthesis.",
                "List the steps to bake a cake, including preparing the ingredients, mixing, baking, and decorating.",
                "Solve the integral of x^2 dx.",
                "Describe how to set up a machine learning pipeline involving data collection, preprocessing, model training, and evaluation.",
                "Define Newton's second law of motion.",
                "Provide a comprehensive guide to building a web application using Django.",
                "What are the main causes of World War II?",
                "Explain the theory of relativity.",
                "Detail the process of software development lifecycle, including requirements gathering, design, implementation, testing, deployment, and maintenance.",
            ],
            "complex": [0, 0, 1, 0, 1, 0, 1, 0, 0, 1],
        }
    )
    X_ml = df_ml["query"]
    y_ml = df_ml["complex"]
    vectorizer_ml = TfidfVectorizer()
    pipeline_ml = Pipeline(
        [("tfidf", TfidfVectorizer()), ("clf", LogisticRegression())]
    )
    pipeline_ml.fit(X_ml, y_ml)
    with open(model_path_ml, "wb") as f:
        pickle.dump(pipeline_ml, f)

# Initialize TF-IDF Vectorizer for Statistical Analysis
vectorizer_stat = TfidfVectorizer()
df_stat = pd.DataFrame(
    {
        "query": [
            "What is the capital of France?",
            "Explain the process of photosynthesis.",
            "List the steps to bake a cake, including preparing the ingredients, mixing, baking, and decorating.",
            "Solve the integral of x^2 dx.",
            "Describe how to set up a machine learning pipeline involving data collection, preprocessing, model training, and evaluation.",
            "Define Newton's second law of motion.",
            "Provide a comprehensive guide to building a web application using Django.",
            "What are the main causes of World War II?",
            "Explain the theory of relativity.",
            "Detail the process of software development lifecycle, including requirements gathering, design, implementation, testing, deployment, and maintenance.",
        ],
        "complex": [0, 0, 1, 0, 1, 0, 1, 0, 0, 1],
    }
)
tfidf_stat = vectorizer_stat.fit_transform(df_stat["query"])

# Function Definitions


# 1. Natural Language Processing (NLP) and Dependency Parsing
def is_complex_nlp_dependency(
    input_query: str, substep_threshold: int = 3, depth_threshold: int = 1
) -> float:
    """
    Determines complexity based on NLP dependency parsing.

    Args:
        input_query (str): The problem to solve.
        substep_threshold (int): Number of substeps to consider complex.
        depth_threshold (int): Depth of dependency tree to consider complex.

    Returns:
        float: Score between 0 and 1 indicating complexity.
    """
    doc = nlp_spacy(input_query)

    # Count verbs as substeps
    substeps = len([token for token in doc if token.pos_ == "VERB"])

    # Determine the maximum depth of the dependency tree
    def get_depth(token, current_depth=0):
        if not list(token.children):
            return current_depth
        return max(get_depth(child, current_depth + 1) for child in token.children)

    depths = [get_depth(token) for token in doc]
    max_depth = max(depths) if depths else 0

    # Normalize scores
    substeps_score = 1 / (1 + math.exp(-substeps / substep_threshold))
    depth_score = 1 / (1 + math.exp(-max_depth / depth_threshold))

    # Combined score
    score = substeps_score * depth_score
    # Normalize score by using sigmoid function
    # score = 1 / (1 + math.exp(-score))

    print(
        f"[NLP Dependency Parsing] Substeps: {substeps}, Max Depth: {max_depth}, Score: {score}"
    )
    return score


def is_complex_spacy_srl(
    input_query: str, substep_threshold: int = 3, depth_threshold: int = 1
) -> float:
    substeps, max_depth = get_srl_spacy(input_query)

    # Normalize scores
    substeps_score = min(substeps - 1, substep_threshold) / substep_threshold
    depth_score = min(max_depth - 1, depth_threshold) / depth_threshold

    # Combined score
    score = (substeps_score + depth_score) / 2
    print(f"Predicates: {substeps}, Max Argument Depth: {max_depth}, Score: {score}")
    return score


# 2. Semantic Role Labeling (SRL)
def is_complex_srl(
    input_query: str, substep_threshold: int = 3, depth_threshold: int = 1
) -> float:
    """
    Determines complexity based on Semantic Role Labeling using Hugging Face.

    Args:
        input_query (str): The problem to solve.
        substep_threshold (int): Number of predicates to consider complex.
        depth_threshold (int): Depth of argument structures to consider complex.

    Returns:
        float: Score between 0 and 1 indicating complexity.
    """
    if not srl_pipeline:
        print("[SRL] SRL Pipeline not available.")
        return 0.0

    result = srl_pipeline(input_query)
    print(f"[SRL] Result: {result}")

    # Extract predicates (verbs) and arguments from result
    predicates = [entity for entity in result if entity["entity"].startswith("I-V")]
    arguments = [entity for entity in result if entity["entity"].startswith("I-ARG")]

    # Analyze substeps and depth like before
    substeps = len(predicates)
    max_depth = len(arguments)

    # Normalize scores
    substeps_score = 1 / (1 + math.exp(-substeps / substep_threshold))
    depth_score = 1 / (1 + math.exp(-max_depth / depth_threshold))

    # Combined score
    score = substeps_score * depth_score
    print(
        f"[SRL] Predicates: {substeps}, Max Argument Depth: {max_depth}, Score: {score}"
    )
    return score


# 3. Machine Learning Classification
def is_complex_ml(input_query: str, threshold: float = 0.5) -> float:
    """
    Determines complexity using a machine learning classifier.

    Args:
        input_query (str): The problem to solve.
        threshold (float): Probability threshold to consider complex.

    Returns:
        float: Probability between 0 and 1 indicating complexity.
    """
    prob = pipeline_ml.predict_proba([input_query])[0][
        1
    ]  # Probability of being complex
    print(f"[Machine Learning] Probability of Complexity: {prob}")
    return prob


class Subtask(BaseModel):
    """
    Subtask model for representing a subtask in a step of a plan or another subtask."""

    subtask_number: int
    subtask_description: str
    subtask_name: str
    subtask_explanation: str
    subtask_output: str
    subtask_full_text: str
    subtasks: List["Subtask"] = Field(default_factory=list)


class PlanStep(BaseModel):
    """
    PlanStep model for representing a step in a plan."""

    step_number: int
    step_name: str
    step_description: str
    step_explanation: str
    step_output: str
    step_full_text: str
    subtasks: List[Subtask]


class Plan(BaseModel):
    """
    Plan model for representing a step-by-step plan."""

    steps: List[PlanStep]


class MathResponse(BaseModel):
    steps: List[PlanStep]
    final_answer: str


def merge_steps(
    steps: List[PlanStep] | List[Subtask], step_obj: PlanStep | Subtask
) -> PlanStep | Subtask:
    """
    Merges a list of existing steps into a single step object. It does this by making a call to the LLM.

    Args:
        steps (List[PlanStep]): List of existing steps to merge.
        step_obj (PlanStep): PlanStep object to merge into.

    Returns:
        PlanStep: Merged step object.
    """
    if isinstance(step_obj, PlanStep):
        # Prepare the input text for the LLM
        input_text = f"Merge the following steps into a single step:\n"
        input_text += f"["
        for step in steps:
            input_text += f"{step.step_full_text}\n"
        input_text += f"{step_obj.step_full_text}]\n"
    elif isinstance(step_obj, Subtask):
        # Prepare the input text for the LLM
        input_text = f"Merge the following subtasks into a single subtask:\n"
        input_text += f"["
        for subtask in steps:
            input_text += f"{subtask.subtask_full_text}\n"
        input_text += f"{step_obj.subtask_full_text}]\n"
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        print("[Language Model] OpenAI API key not found.")
        return ""
    openai.api_key = openai_api_key
    if isinstance(step_obj, PlanStep):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": f"""You are an assistant that merges multiple versions of text describing the same step into a single coherent step description. Please respond ONLY with the merged step, leaving out any other information, preceding the response with 'Merged PlanStep:'. Ensure that the merged step is a coherent and concise representation of the input steps that includes all relevant information.,
                    Example Input Prompt from User:
                    Please merge the following steps into a single coherent step description:
                        [### PlanStep 1: Define the Project Requirements
                            1. **Identify the Purpose**: Determine what the application is meant to do.
                            2. **List Features**: Write down the essential features you want to implement (e.g., user authentication, data display, etc.).
                            3. **Choose Technology Stack**: Decide on the technologies for the frontend (e.g., React, Vue.js), backend (e.g., Node.js, Django), and database (e.g., PostgreSQL, MongoDB).,
                        PlanStep 1. Define the Project Requirements: Identify the purpose of the application, list essential features like input validation and data storage, and choose the technology stack.,
                        - PlanStep 1: First, identify the purpose of the application to determine its functionality. Next, list the essential features that need to be implemented, such as user authentication and data display. Finally, choose the technology stack, including frontend technologies like Flutter or React, backend technologies like Flask or Django, and database technologies like Redis or GraphQL.,
                        - PlanStep 1: Define the project requirements by identifying the purpose of the application, listing essential features, and choosing the technology stack. The purpose will determine the functionality, the features will outline the user experience, and the technology stack will define the development environment.###]
                    Example Output Response from Assistant:

                    Merged PlanStep:
                        PlanStep 1: Define the Project Requirements
                        - Identify the Purpose: Determine what the application is meant to do and its expected functionality.
                        - List Essential Features: Write down the essential features you want to implement, like user authentication, data display, input validation, and data storage.
                        - Choose Technology Stack: Decide on the technologies for the frontend, backend, and database, such as React, Flutter or Vue.js for the frontend, Node.js, Flask, or Django for the backend, and PostgreSQL, MongoDB, Redis, or GraphQL for the database.""",
                    },
                    {
                        "role": "user",
                        "content": input_text,
                    },
                ],
                n=1,
                stop=None,
                max_completion_tokens=2500,
                temperature=0.5,
            )

            output = response.choices[0].message.content
            # Parse out the merged step from the response
            merged_step_text = output.split("Merged PlanStep:")[1].strip()
            merged_step = PlanStep(
                step_number=step_obj.step_number,
                step_name=step_obj.step_name,
                step_description=step_obj.step_description,
                step_explanation=step_obj.step_explanation,
                step_output=step_obj.step_output,
                step_full_text=merged_step_text,
                subtasks=step_obj.subtasks,
            )

        except Exception as e:
            print(f"[Language Model] Error merging steps: {e}")
            return step_obj
    elif isinstance(step_obj, Subtask):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": f"""You are an assistant that merges multiple versions of text describing the same subtask into a single coherent subtask description. Please respond ONLY with the merged subtask, leaving out any other information, preceding the response with 'Merged Subtask:'. Ensure that the merged subtask is a coherent and concise representation of the input subtasks that includes all relevant information.,
                    Example Input Prompt from User:
                    Please merge the following subtasks into a single coherent subtask description:
                        [3. **Choose Technology Stack**: Decide on the technologies for the frontend (e.g., React, Vue.js), backend (e.g., Node.js, Django), and database (e.g., PostgreSQL, MongoDB).,
                        Subtask 3. Define the Project Requirements: Identify the purpose of the application, list essential features like input validation and data storage, and choose the technology stack.,
                        - Finally, choose the technology stack, including frontend technologies like Flutter or React, backend technologies like Flask or Django, and database technologies like Redis or GraphQL.,
                        - The purpose will determine the functionality, the features will outline the user experience, and the technology stack will define the development environment.###]
                    Example Output Response from Assistant:

                    Merged Subtask:
                        Choose Technology Stack: Decide on the technologies for the frontend, backend, and database, such as React, Flutter or Vue.js for the frontend, Node.js, Flask, or Django for the backend, and PostgreSQL, MongoDB, Redis, or GraphQL for the database.""",
                    },
                    {
                        "role": "user",
                        "content": input_text,
                    },
                ],
                n=1,
                stop=None,
                max_completion_tokens=2500,
                temperature=0.5,
            )

            output = response.choices[0].message.content

            # Parse out the merged subtask from the response
            merged_subtask_text = output.split("Merged Subtask:")[1].strip()
            merged_subtask = Subtask(
                subtask_number=step_obj.subtask_number,
                subtask_name=step_obj.subtask_name,
                subtask_description=step_obj.subtask_description,
                subtask_explanation=step_obj.subtask_explanation,
                subtask_output=step_obj.subtask_output,
                subtask_full_text=merged_subtask_text,
                subtasks=step_obj.subtasks,
            )

            return merged_subtask
        except Exception as e:
            print(f"[Language Model] Error merging subtasks: {e}")
            return step_obj

    # Now merge the subtasks the same way
    if (
        isinstance(step_obj, PlanStep)
        or len(step_obj.subtasks) != 0
        and merged_step is not None
    ):
        for subtask in step_obj.subtasks:
            other_subtask_sets = [step.subtasks for step in steps]
            # Find the corresponding subtask in the other sets
            if all(
                subtask.subtask_number
                in [subtask.subtask_number for subtask in other_subtask_set]
                for other_subtask_set in other_subtask_sets
            ):  # Check if the subtask is present in all other sets
                # Get all the subtasks with the same number
                other_subtasks = [
                    subtask
                    for subtask_set in other_subtask_sets
                    for subtask in subtask_set
                ]
                other_subtasks = [
                    subtask
                    for subtask in other_subtasks
                    if subtask.subtask_number == subtask.subtask_number
                ]
                merged_subtask = merge_steps(other_subtasks, subtask)
                merged_step.subtasks[merged_step.subtasks.index(subtask)] = (
                    merged_subtask
                )
            elif not any(
                subtask.subtask_number
                in [subtask.subtask_number for subtask in other_subtask_set]
                for other_subtask_set in other_subtask_sets
            ):  # Check if the subtask is not present in any other set
                if subtask.subtask_number not in [
                    subtask.subtask_number for subtask in merged_step.subtasks
                ]:
                    merged_step.subtasks[merged_step.subtasks.index(subtask)] = subtask
            else:
                # If the subtask is present in some but not all sets, we need to merge them
                other_subtasks = [
                    subtask
                    for subtask_set in other_subtask_sets
                    for subtask in subtask_set
                ]
                other_subtasks = [
                    subtask
                    for subtask in other_subtasks
                    if subtask.subtask_number == subtask.subtask_number
                ]
                merged_subtask = merge_steps(other_subtasks, subtask)
                merged_step.subtasks[merged_step.subtasks.index(subtask)] = (
                    merged_subtask
                )
    else:
        merged_step.subtasks = step_obj.subtasks if merged_step is not None else []

    return merged_step


# 4. Language Model-Based Analysis
def generate_plan(input_query: str) -> str:
    """
    Generates a step-by-step plan using OpenAI's GPT model.

    Args:
        input_query (str): The problem to solve.

    Returns:
        str: Generated plan as text.
    """
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        print("[Language Model] OpenAI API key not found.")
        return ""
    openai.api_key = openai_api_key

    try:
        response = client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are an assistant that breaks down problems into step-by-step plans that are easy to follow by an LLM.",
                },
                {
                    "role": "user",
                    "content": f"Provide a detailed, LLM-oriented step-by-step plan to solve the following problem:\n\n{input_query}",
                },
            ],
            n=1,
            stop=None,
            max_completion_tokens=2500,
            temperature=0.5,
            response_format=Plan,
        )
        output = response.choices[0].message.parsed

        return output
    except Exception as e:
        print(f"[Language Model] Error generating plan: {e}")
        return ""


convert_instruction_a = f"""
You are an assistant that receives a step-by-step plan and converts it into a structured format by identifying the hierarchical structure of steps and subtasks. You must read the plan step by step and extract the top-level steps and their subtasks. Pay special attention to the nesting level of each step and ensure that subtasks are correctly placed under their respective steps.
Each step should have a step number, a name, description, explanation, expected output, and possibly a list of subtasks. There is also the full text of the identified step.
Each subtask should have a name, description, explanation, expected output, and and can have subtasks of its own if applicable. There is also the full text of the identified subtask.
The assistant should identify the steps and subtasks in the plan and provide a structured representation of the plan based on the identified steps and subtasks and their ordered step numbers, names, descriptions, explanations, and outputs.
For each step: 
    -You will identify or generate (depending on whether it is allowed for that argument or field) the step number, the name of the step, the description of the step, an explanation of the step, the expected output of the step, and the full text of the step. 
    - Identify each step's description and explanation from the full step text verbatim, wherever possible, and only generate when the description is ambiguous.
    -The step number should be a sequential number starting from 1 for the first step. Ensure that the step numbers are sequential and increment by 1 for each subsequent step and maintain the order of the steps as they appear in the plan. PlanStep numbers should not skip or repeat.
    -The name of the step should be a concise title or label for the step, generated based on the step text.
    -The description should be a concise summary of the step, generated based on the step text.
    -If an explanation is not discernible, you can write your own based on the step text. 
    -For the output, you should identify the expected result of the step as best as possible, generating a reasonable expected output result of the step if it is not explicitly stated. 
    -The full text of the step should be the complete text of the identified step, extracted from the plan verbatim.
    -If a step has subtasks, you should identify the name, description, explanation, and sub-subtasks for each subtask.
    -For each step, explicitly check for and include any nested subtasks, making sure that subtask numbers, names, descriptions, and their hierarchy are preserved.
    - Don't generate new steps or subtasks that are not present in the plan.
For each subtask, the same fields as the step should be identified or generated, including the name, description, explanation, expected output, and full text of the subtask, using the same rules. If a subtask has sub-subtasks, you should identify the name, description, explanation, and expected output of each sub-subtask.
- Ensure each subtask is nested properly under its corresponding step.
- Subtasks should always belong to the immediate preceding step, unless explicitly stated otherwise.
Remember to maintain the hierarchical structure of the steps and subtasks, ensuring that subtasks are nested under the appropriate steps and sub-subtasks are nested under the appropriate subtasks.
"""

exemplar_plan = Plan(
    steps=[
        PlanStep(
            step_number=1,
            step_name="Assess Network Requirements",
            step_description="Determine the needs of the network.",
            step_explanation="Identify the number of devices, types of devices, and the required bandwidth to ensure the network meets all user requirements.",
            step_output="A clear understanding of the network requirements, including device count and bandwidth needs.",
            step_full_text="Assess Network Requirements: Determine the needs of the network. Identify the number of devices, types of devices, and the required bandwidth.",
            subtasks=[],
        ),
        PlanStep(
            step_number=2,
            step_name="Choose the Right Equipment",
            step_description="Select appropriate networking hardware.",
            step_explanation="Based on the assessed requirements, purchase a suitable router, switches (if necessary), and wireless access points to build a robust network.",
            step_output="All necessary networking equipment is purchased and ready for installation.",
            step_full_text="Choose the Right Equipment: Select appropriate networking hardware. Purchase a suitable router, switches (if necessary), and wireless access points based on the network requirements.",
            subtasks=[],
        ),
        PlanStep(
            step_number=3,
            step_name="Install the Router",
            step_description="Physically set up the main networking device.",
            step_explanation="Connect the router to the modem and ensure it is powered on to establish the primary network gateway.",
            step_output="Router is connected to the modem and powered on.",
            step_full_text="Install the Router: Physically set up the main networking device. Connect the router to the modem and ensure it is powered on.",
            subtasks=[],
        ),
        PlanStep(
            step_number=4,
            step_name="Configure Router Settings",
            step_description="Set up the router for optimal performance and security.",
            step_explanation="Access the router's admin panel to change default settings, establish a secure Wi-Fi network, and enable advanced security features.",
            step_output="Router is configured with a secure and optimized setup.",
            step_full_text="Configure Router Settings: Set up the router for optimal performance and security.",
            subtasks=[
                Subtask(
                    subtask_number=1,
                    subtask_name="Access Admin Panel",
                    subtask_description="Access the router's administrative interface.",
                    subtask_explanation="Use a web browser to navigate to the router's admin panel using its IP address for configuration.",
                    subtask_output="Successfully accessed the router's admin panel.",
                    subtask_full_text="Access the router's admin panel via a web browser.",
                    subtasks=[],
                ),
                Subtask(
                    subtask_number=2,
                    subtask_name="Change Admin Password",
                    subtask_description="Enhance security by updating the default admin password.",
                    subtask_explanation="Replace the router's default admin password with a strong, unique password to prevent unauthorized access.",
                    subtask_output="Admin password is changed to a secure one.",
                    subtask_full_text="Change the default admin password.",
                    subtasks=[],
                ),
                Subtask(
                    subtask_number=3,
                    subtask_name="Set Up Wi-Fi Network",
                    subtask_description="Establish the Wi-Fi network name and password.",
                    subtask_explanation="Create a unique SSID (network name) and set a strong password to secure the wireless network.",
                    subtask_output="Wi-Fi network name (SSID) and password are set.",
                    subtask_full_text="Set up the Wi-Fi network name (SSID) and a strong password.",
                    subtasks=[],
                ),
                Subtask(
                    subtask_number=4,
                    subtask_name="Enable WPA3 Encryption",
                    subtask_description="Enhance wireless security by enabling WPA3 encryption.",
                    subtask_explanation="Activate WPA3 encryption on the router to provide the latest security standards for the wireless network.",
                    subtask_output="WPA3 encryption is enabled on the Wi-Fi network.",
                    subtask_full_text="Enable WPA3 encryption for enhanced security.",
                    subtasks=[],
                ),
            ],
        ),
        PlanStep(
            step_number=5,
            step_name="Connect Devices to the Network",
            step_description="Ensure all intended devices are connected.",
            step_explanation="Connect each device to the newly created Wi-Fi network using the set SSID and password.",
            step_output="All devices are successfully connected to the wireless network.",
            step_full_text="Connect Devices to the Network: Ensure all intended devices are connected.",
            subtasks=[
                Subtask(
                    subtask_number=1,
                    subtask_name="Search for Wi-Fi Network",
                    subtask_description="Locate the Wi-Fi network on each device.",
                    subtask_explanation="Use the device's network settings to find and select the appropriate Wi-Fi network (SSID).",
                    subtask_output="Wi-Fi network is visible and selectable on each device.",
                    subtask_full_text="Search for the Wi-Fi network on each device.",
                    subtasks=[],
                ),
                Subtask(
                    subtask_number=2,
                    subtask_name="Enter Wi-Fi Password",
                    subtask_description="Authenticate by entering the network password.",
                    subtask_explanation="Input the previously set strong password to establish a secure connection between the device and the network.",
                    subtask_output="Device is connected to the Wi-Fi network.",
                    subtask_full_text="Enter the Wi-Fi password to establish a connection.",
                    subtasks=[],
                ),
            ],
        ),
        PlanStep(
            step_number=6,
            step_name="Test the Network",
            step_description="Verify that the network is functioning correctly.",
            step_explanation="Conduct connectivity and performance tests to ensure the network operates as intended.",
            step_output="Network is confirmed to be operational and meeting performance standards.",
            step_full_text="Test the Network: Verify that the network is functioning correctly.",
            subtasks=[
                Subtask(
                    subtask_number=1,
                    subtask_name="Check Internet Connectivity",
                    subtask_description="Ensure devices can access the internet.",
                    subtask_explanation="Open a web browser on multiple devices to confirm that internet access is available.",
                    subtask_output="Internet connectivity is confirmed on all tested devices.",
                    subtask_full_text="Check internet connectivity on multiple devices.",
                    subtasks=[],
                ),
                Subtask(
                    subtask_number=2,
                    subtask_name="Perform Speed Tests",
                    subtask_description="Measure the network's bandwidth and speed.",
                    subtask_explanation="Use online speed testing tools to verify that the network meets the required bandwidth specifications.",
                    subtask_output="Network speed and bandwidth are within expected ranges.",
                    subtask_full_text="Perform speed tests to ensure adequate bandwidth.",
                    subtasks=[],
                ),
                Subtask(
                    subtask_number=3,
                    subtask_name="Ensure Stable Connections",
                    subtask_description="Confirm that devices maintain a consistent connection.",
                    subtask_explanation="Monitor the network to ensure that devices remain connected without frequent drops or interruptions.",
                    subtask_output="Devices maintain stable and consistent connections to the network.",
                    subtask_full_text="Ensure that all devices maintain a stable connection.",
                    subtasks=[],
                ),
            ],
        ),
        PlanStep(
            step_number=7,
            step_name="Optimize Network Performance",
            step_description="Enhance the efficiency and speed of the network.",
            step_explanation="Implement strategies to maximize network coverage and performance while minimizing interference.",
            step_output="Network performance is optimized for better speed and coverage.",
            step_full_text="Optimize Network Performance: Enhance the efficiency and speed of the network.",
            subtasks=[
                Subtask(
                    subtask_number=1,
                    subtask_name="Position Router Centrally",
                    subtask_description="Place the router in an optimal location for maximum coverage.",
                    subtask_explanation="Locate the router in a central area of the home to ensure even distribution of the wireless signal.",
                    subtask_output="Router is positioned for optimal network coverage.",
                    subtask_full_text="Place the router in a central location to maximize coverage.",
                    subtasks=[],
                ),
                Subtask(
                    subtask_number=2,
                    subtask_name="Update Router Firmware",
                    subtask_description="Ensure the router has the latest software updates.",
                    subtask_explanation="Access the router's admin panel to check for and install any available firmware updates, which can improve performance and security.",
                    subtask_output="Router firmware is updated to the latest version.",
                    subtask_full_text="Update the router's firmware to the latest version.",
                    subtasks=[],
                ),
                Subtask(
                    subtask_number=3,
                    subtask_name="Adjust Channel Settings",
                    subtask_description="Minimize wireless interference by selecting optimal channels.",
                    subtask_explanation="Change the Wi-Fi channel settings on the router to less congested channels to reduce interference from neighboring networks.",
                    subtask_output="Wi-Fi channels are optimized to minimize interference.",
                    subtask_full_text="Adjust channel settings to minimize interference.",
                    subtasks=[],
                ),
            ],
        ),
        PlanStep(
            step_number=8,
            step_name="Secure the Network",
            step_description="Protect the network from unauthorized access and threats.",
            step_explanation="Implement security measures to safeguard the network and connected devices from potential vulnerabilities.",
            step_output="Network security is enhanced to prevent unauthorized access.",
            step_full_text="Secure the Network: Protect the network from unauthorized access and threats.",
            subtasks=[
                Subtask(
                    subtask_number=1,
                    subtask_name="Disable WPS",
                    subtask_description="Turn off Wi-Fi Protected Setup to enhance security.",
                    subtask_explanation="Disable WPS in the router settings to prevent potential security breaches through this feature.",
                    subtask_output="WPS is disabled on the router.",
                    subtask_full_text="Disable WPS (Wi-Fi Protected Setup).",
                    subtasks=[],
                ),
                Subtask(
                    subtask_number=2,
                    subtask_name="Set Up Guest Network",
                    subtask_description="Create a separate network for guests to protect the main network.",
                    subtask_explanation="Enable a guest Wi-Fi network on the router to allow visitors to access the internet without accessing the primary network resources.",
                    subtask_output="Guest network is set up and operational.",
                    subtask_full_text="Set up a guest network for visitors.",
                    subtasks=[],
                ),
                Subtask(
                    subtask_number=3,
                    subtask_name="Monitor Connected Devices",
                    subtask_description="Regularly check devices connected to the network.",
                    subtask_explanation="Use the router's admin panel to view and manage connected devices, ensuring no unauthorized devices are accessing the network.",
                    subtask_output="All connected devices are monitored and managed.",
                    subtask_full_text="Regularly monitor connected devices and update security settings as needed.",
                    subtasks=[],
                ),
            ],
        ),
        PlanStep(
            step_number=9,
            step_name="Document the Network Setup",
            step_description="Keep a record of the network configuration for future reference.",
            step_explanation="Maintain documentation of the network's settings and hardware placement to facilitate troubleshooting and future upgrades.",
            step_output="Comprehensive documentation of the network setup is available.",
            step_full_text="Document the Network Setup: Keep a record of the network configuration for future reference.",
            subtasks=[
                Subtask(
                    subtask_number=1,
                    subtask_name="Record Network Credentials",
                    subtask_description="Write down the network name and password.",
                    subtask_explanation="Document the SSID and Wi-Fi password in a secure location for easy access when needed.",
                    subtask_output="Network name (SSID) and password are recorded.",
                    subtask_full_text="Write down the network name (SSID) and password.",
                    subtasks=[],
                ),
                Subtask(
                    subtask_number=2,
                    subtask_name="Note Admin Credentials",
                    subtask_description="Store the router's admin login details securely.",
                    subtask_explanation="Ensure that the router's admin username and password are documented and stored in a secure place to prevent unauthorized changes.",
                    subtask_output="Router's admin credentials are noted and secured.",
                    subtask_full_text="Note down the router's admin login credentials.",
                    subtasks=[],
                ),
                Subtask(
                    subtask_number=3,
                    subtask_name="Document Hardware Placement",
                    subtask_description="Keep a record of where networking hardware is located.",
                    subtask_explanation="Map out the physical placement of the router, switches, and access points to aid in maintenance and troubleshooting.",
                    subtask_output="Placement of all networking hardware is documented.",
                    subtask_full_text="Document the placement of networking hardware.",
                    subtasks=[],
                ),
            ],
        ),
    ]
)


convert_instruction_b = f"""
You are an assistant that receives a step-by-step plan and converts it into a structured format by identifying the hierarchical structure of steps and subtasks. You must read the plan and extract the top-level steps and their subtasks. 

Each step should have:
- step_number: Sequential number starting from 1.
- step_name: Concise title for the step.
- step_description: Concise summary of the step.
- step_explanation: Detailed explanation of the step.
- step_output: Expected result of the step.
- step_full_text: Complete text of the step.
- subtasks: List of Subtask objects.

Each subtask should have:
- subtask_number: Sequential number within its parent step.
- subtask_name: Concise title for the subtask.
- subtask_description: Concise summary of the subtask.
- subtask_explanation: Detailed explanation of the subtask.
- subtask_output: Expected result of the subtask.
- subtask_full_text: Complete text of the subtask.
- subtasks: (Optional) List of Subtask objects for nested subtasks.

### Exemplars:

**Example 1:**

*Plan Text String:*
To set up a home wireless network, follow this detailed plan:

### PlanStep 1: Assess Network Requirements
- **Objective**: Determine the needs of the network.
- **Action**: Identify the number of devices, types of devices, and the required bandwidth.

### PlanStep 2: Choose the Right Equipment
- **Objective**: Select appropriate networking hardware.
- **Action**: Purchase a suitable router, switches (if necessary), and wireless access points based on the network requirements.

### PlanStep 3: Install the Router
- **Objective**: Physically set up the main networking device.
- **Action**: Connect the router to the modem and ensure it is powered on.

### PlanStep 4: Configure Router Settings
- **Objective**: Set up the router for optimal performance and security.
- **Action**: 
    - **PlanStep 4.1**: Access the router's admin panel via a web browser.
    - **PlanStep 4.2**: Change the default admin password.
    - **PlanStep 4.3**: Set up the Wi-Fi network name (SSID) and a strong password.
    - **PlanStep 4.4**: Enable WPA3 encryption for enhanced security.

### PlanStep 5: Connect Devices to the Network
- **Objective**: Ensure all intended devices are connected.
- **Action**: 
    - **PlanStep 5.1**: Search for the Wi-Fi network on each device.
    - **PlanStep 5.2**: Enter the Wi-Fi password to establish a connection.

### PlanStep 6: Test the Network
- **Objective**: Verify that the network is functioning correctly.
- **Action**: 
    - **PlanStep 6.1**: Check internet connectivity on multiple devices.
    - **PlanStep 6.2**: Perform speed tests to ensure adequate bandwidth.
    - **PlanStep 6.3**: Ensure that all devices maintain a stable connection.

### PlanStep 7: Optimize Network Performance
- **Objective**: Enhance the efficiency and speed of the network.
- **Action**: 
    - **PlanStep 7.1**: Place the router in a central location to maximize coverage.
    - **PlanStep 7.2**: Update the router's firmware to the latest version.
    - **PlanStep 7.3**: Adjust channel settings to minimize interference.

### PlanStep 8: Secure the Network
- **Objective**: Protect the network from unauthorized access and threats.
- **Action**: 
    - **PlanStep 8.1**: Disable WPS (Wi-Fi Protected Setup).
    - **PlanStep 8.2**: Set up a guest network for visitors.
    - **PlanStep 8.3**: Regularly monitor connected devices and update security settings as needed.

### PlanStep 9: Document the Network Setup
- **Objective**: Keep a record of the network configuration for future reference.
- **Action**: 
    - **PlanStep 9.1**: Write down the network name (SSID) and password.
    - **PlanStep 9.2**: Note down the router's admin login credentials.
    - **PlanStep 9.3**: Document the placement of networking hardware.
*Expected Plan Object:*
```python
{exemplar_plan}
```
"""


class TextClassification(BaseModel):
    """
    TextClassification model for representing the classification of a text.
    """

    is_useful: bool


import openai
import os
from typing import Optional
from pydantic import BaseModel


# Define the custom class for structured output
class TextClassification(BaseModel):
    """
    TextClassification model for representing the classification of a text.
    """

    is_useful: bool


def classify_remaining_text_structured(
    leftover_text: str,
    api_key: Optional[str] = None,
    model: str = "gpt-4o-mini",
    temperature: float = 0.0,
    max_completion_tokens: int = 10,
) -> Optional[TextClassification]:
    """
    Classifies the leftover text from a step-by-step plan as 'useful' or 'junk' using a structured output.

    Args:
        leftover_text (str): The text to classify.
        api_key (Optional[str]): Your OpenAI API key. If not provided, the function will
                                 attempt to read it from the OPENAI_API_KEY environment variable.
        model (str): The OpenAI model to use for classification. Defaults to 'gpt-4'.
        temperature (float): Sampling temperature. Defaults to 0.0 for deterministic results.
        max_completion_tokens (int): Maximum number of tokens to generate in the response. Defaults to 10.

    Returns:
        Optional[TextClassification]: An instance of TextClassification with is_useful set to True or False.
                                      Returns None if classification fails.
    """
    # Set the OpenAI API key
    if api_key is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key is None:
            raise ValueError(
                "OpenAI API key not provided and not found in environment variables."
            )
    openai.api_key = api_key
    system_prompt = f"""You are an intelligent assistant that classifies text snippets as 'useful' or 'junk' based on their relevance and informativeness in a generated plan, with the goal of identifying the remaining text as 'useful' or 'junk'. Respond only with a bool that is true for 'useful' and false for 'junk'. This bool will be returned as the 'is_useful' field in the TextClassification class.

### Instructions:
- **Useful**: Contains meaningful information or instructions relevant to a plan, including steps, clarifications, or actionable items.
- **Junk**: Contains non-informative, filler phrases, repetitive instructions, or generic comments that do not contribute directly to the steps or outcome.

Examples:
1. **Text**: '### PlanStep 8: Secure the Network - Protect the network from unauthorized access and threats.'
   **Classification**:
   {{
       "is_useful": true
   }}

2. **Text**: 'Ensure that all devices maintain a stable connection.'
    **Classification**:
    {{
         "is_useful": true
    }}

3. **Text**: 'The process involves a series of steps that will help you achieve your goal.'
    **Classification**:
    {{
         "is_useful": false
    }}

4. **Text**: 'Please carefully follow each step to ensure success.'
    **Classification**:
    {{
         "is_useful": false
    }}

5. **Text**: 'After deployment, monitor the server for any errors or issues.'
    **Classification**:
    {{
         "is_useful": true
    }}

6. **Text**: 'In the following steps, we will guide you through setting up a development environment.'
    **Classification**:
    {{
         "is_useful": false
    }}

7. **Text**: 'PlanStep 5: Configure the network settings - Set up IP addresses and ensure network connectivity.'  
    **Classification**:
    {{
         "is_useful": true
    }}

8. **Text**: 'carefully, and soon it will be ready to use.'
    **Classification**:
    {{
         "is_useful": false
    }}

9. **Text**: 'This step is necessary when c.lrqc'
    **Classification**:
    {{
         "is_useful": false
    }}
"""
    # Define the prompt with few-shot examples related to step-by-step plans
    prompt = f"""
You are an intelligent assistant that classifies text snippets as 'useful' or 'junk' based on their relevance and informativeness in a generated plan.

### Instructions:
- **Useful**: Contains meaningful information or instructions relevant to a plan, including steps, clarifications, or actionable items.
- **Junk**: Contains non-informative, filler phrases, repetitive instructions, or generic comments that do not contribute directly to the steps or outcome.

Respond only with a bool that is true for 'useful' and false for 'junk'. This bool will be returned as the 'is_useful' field in the TextClassification class. 
### Examples:

1.
**Text**: "### PlanStep 2: Set up the development environment - Install Python and create a virtual environment to manage dependencies."
**Classification**:
{{
    "is_useful": true
}}

2.
**Text**: "In the following steps, we will guide you through setting up a development environment."
**Classification**:
{{
    "is_useful": false
}}

3.
**Text**: "PlanStep 5: Configure the network settings - Set up IP addresses and ensure network connectivity."
**Classification**:
{{
    "is_useful": true
}}

4.
**Text**: "The process involves a series of steps that will help you achieve your goal."
**Classification**:
{{
    "is_useful": false
}}

5.
**Text**: "### Final PlanStep: Test the application - Run the application to verify that it meets the specified requirements."
**Classification**:
{{
    "is_useful": true
}}

6.
**Text**: "Please carefully follow each step to ensure success."
**Classification**:
{{
    "is_useful": false
}}

7.
**Text**: "After deployment, monitor the server for any errors or issues."
**Classification**:
{{
    "is_useful": true
}}

### New Text:

**Text**: "{leftover_text}"
**Classification**:
"""

    try:
        # Make the API request with structured output
        response = client.beta.chat.completions.parse(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            n=1,
            response_format=TextClassification,
        )

        # Extract the parsed class object
        parsed_response = response.choices[0].message.parsed

        if isinstance(parsed_response, TextClassification):
            return parsed_response
        else:
            print(f"Unexpected response format: {parsed_response}")
            return None

    except Exception as e:
        # Handle other possible errors
        print(f"Unexpected error: {e}")
        return None


def test_classify_remaining_text_structured():
    # Mock API key (Replace this with your actual OpenAI API key if needed)
    api_key = "your_api_key_here"

    # Define test cases
    test_cases = [
        {
            "input": "### PlanStep 2: Install the required software - Download and install Node.js and npm for package management.",
            "expected": True,
            "description": "A useful step with actionable instructions.",
        },
        {
            "input": "In the following steps, we will guide you through the process.",
            "expected": False,
            "description": "Non-informative introductory phrase.",
        },
        {
            "input": "### PlanStep 5: Deploy the application - Transfer files to the server and configure environment variables.",
            "expected": True,
            "description": "A useful step for deploying an application.",
        },
        {
            "input": "Please carefully follow each step to avoid issues.",
            "expected": False,
            "description": "Generic advice that doesn't add actionable information.",
        },
        {
            "input": "### Final PlanStep: Test the application - Run tests to verify functionality.",
            "expected": True,
            "description": "A clear final step with a specific action.",
        },
        {
            "input": "After deployment, monitor the server for any errors.",
            "expected": True,
            "description": "Actionable instruction for post-deployment.",
        },
        {
            "input": "When the FER35r dl.4et, yes'p",
            "expected": False,
            "description": "Non-informative text with gibberish.",
        },
    ]

    # Run test cases
    for case in test_cases:
        result = classify_remaining_text_structured(
            leftover_text=case["input"],
            api_key=api_key,
            model="gpt-4o-mini",
            temperature=0.0,
            max_completion_tokens=10,
        )

        # Ensure result is not None
        assert result is not None, f"Failed: {case['description']} - Got None response"

        # Check if the output matches the expected value
        assert result.is_useful == case["expected"], (
            f"Failed: {case['description']} - "
            f"Expected: {case['expected']}, Got: {result.is_useful}"
        )

    print("All tests passed!")


# Call the test function
test_classify_remaining_text_structured()


def normalize_whitespace(text):
    return " ".join(text.split())


def remove_punctuation(text):
    return text
    # return text.translate(str.maketrans("", "", string.punctuation))


def remove_converted_text_preserving_order(input_query, steps, case_sensitive=False):
    """
    Removes converted step and subtask texts from the input_query while preserving the order of the remaining text.

    Args:
        input_query (tuple of strings): The original input text.
        steps (list): A list of step objects, each containing subtasks.
        case_sensitive (bool): Whether the removal should be case-sensitive.

    Returns:
        str: The remaining text after removal.
    """

    # Escape only special regex characters, not spaces
    def escape_regex(text):
        return re.sub(r"([.^$*+?{}\[\]|()\\\-\"'])", r"\\\1", text)

    try:
        import re

        # Function to remove punctuation from a string

        print(
            f"Input Query in remove_converted_text_preserving_order: {input_query} \n\n"
        )
        # Create punctuation-free versions of input_query
        input_query_no_punctuation = remove_punctuation(input_query)
        input_query_no_punctuation = normalize_whitespace(  # Normalize whitespace
            input_query_no_punctuation
        )
        if not case_sensitive:
            input_query_no_punctuation = input_query_no_punctuation.lower()
        # Collect all texts to remove, in both original and punctuation-free versions
        texts_to_remove = []
        texts_to_remove_no_punctuation = []
        text_mapping = defaultdict(list)  # Mapping from no_punct to original texts
        input_query_punc_mapping = defaultdict(
            list
        )  # Mapping from original to no_punct input query, separated line by line
        # Create a mapping of each line in the input query to its punctuation-free version
        for line in input_query.split("\n"):
            line_no_punctuation = remove_punctuation(line)
            line_no_punctuation = normalize_whitespace(line_no_punctuation)
            if not case_sensitive:
                line_no_punctuation = line_no_punctuation.lower()
            input_query_punc_mapping[line].append(line_no_punctuation)

        for step in steps:
            step_text = step.step_full_text
            print(f"PlanStep Text Type: {type(step_text)} \n\n")
            # make sure its a string
            if isinstance(step_text, list):
                step_text = " ".join(step_text)
            elif not isinstance(step_text, str):
                step_text = str(step_text)

            if step_text:
                step_text_no_punctuation = normalize_whitespace(
                    remove_punctuation(step_text)
                )
                if not case_sensitive:
                    step_text_no_punctuation = step_text_no_punctuation.lower()
                escaped_step_text_no_punctuation = escape_regex(
                    step_text_no_punctuation
                )
                # Check if step_text is a subset of input_query effectively using the step_text_no_punctuation
                pattern = re.compile(
                    escaped_step_text_no_punctuation, flags=re.IGNORECASE
                )
                matches = list(pattern.finditer(input_query_no_punctuation))
                if len(matches) >= 1:
                    texts_to_remove.append(step_text)
                    texts_to_remove_no_punctuation.append(
                        step_text_no_punctuation
                    )  # Punctuation-free version
                    text_mapping[step_text_no_punctuation].append(step_text)
                else:
                    print(
                        f"No Matches Found in remove_converted_text_preserving_order for PlanStep Text: {step_text} \n\n"
                    )
                    print(
                        f"PlanStep Text No Punctuation in remove_converted_text_preserving_order: {step_text_no_punctuation} \n\n"
                    )
                    print(
                        f"Escaped PlanStep Text No Punctuation in remove_converted_text_preserving_order: {escaped_step_text_no_punctuation} \n\n"
                    )

            for subtask in step.subtasks:
                subtask_text = subtask.subtask_full_text
                print(f"Subtask Text Type: {type(subtask_text)} \n\n")
                # make sure its a string
                if isinstance(step_text, list):
                    step_text = " ".join(step_text)
                elif not isinstance(step_text, str):
                    step_text = str(step_text)

                if subtask_text:
                    # Check if subtask_text is a subset of input_query
                    subtask_text_no_punctuation = normalize_whitespace(
                        remove_punctuation(subtask_text)
                    )
                    if not case_sensitive:
                        subtask_text_no_punctuation = (
                            subtask_text_no_punctuation.lower()
                        )
                    escaped_subtask_text_no_punctuation = escape_regex(
                        subtask_text_no_punctuation
                    )

                    pattern = re.compile(
                        escaped_subtask_text_no_punctuation, flags=re.IGNORECASE
                    )
                    matches = list(pattern.finditer(input_query_no_punctuation))
                    if len(matches) >= 1:
                        texts_to_remove.append(subtask_text)
                        texts_to_remove_no_punctuation.append(
                            subtask_text_no_punctuation
                        )  # Punctuation-free version
                        text_mapping[subtask_text_no_punctuation].append(subtask_text)

        if not texts_to_remove_no_punctuation:
            print(
                f"No texts to remove in remove_converted_text_preserving_order: {texts_to_remove_no_punctuation} \n\n"
            )
            return "", ""
        print(
            f"Texts to Remove No Punctuation in remove_converted_text_preserving_order: {texts_to_remove_no_punctuation} \n\n"
        )
        # Normalize case if not case-sensitive
        flags = 0
        # Normalize and remove punctuation

        # texts_to_remove_no_punctuation = [
        #     normalize_whitespace(text) for text in texts_to_remove_no_punctuation
        # ]

        if not case_sensitive:
            texts_to_remove_no_punctuation = [
                text.lower() for text in texts_to_remove_no_punctuation
            ]
            flags |= re.IGNORECASE

        # Remove duplicates and sort by length descending to handle overlaps
        unique_texts_to_remove_no_punctuation = sorted(
            set(texts_to_remove_no_punctuation), key=len, reverse=True
        )
        print(
            f"Unique Texts to Remove No Punctuation in remove_converted_text_preserving_order: {unique_texts_to_remove_no_punctuation} \n\n"
        )

        # Escape texts for regex

        escaped_texts_no_punctuation = [
            escape_regex(text)
            for text in unique_texts_to_remove_no_punctuation
            if len(str(text)) > 10  # Ensure text is not too short
        ]

        print(
            f"Escaped Texts No Punctuation in remove_converted_text_preserving_order: {escaped_texts_no_punctuation} \n\n"
        )
        assert escaped_texts_no_punctuation  # Ensure there are escaped texts
        # Compile regex pattern (punctuation-free)
        # pattern = re.compile("|".join(escaped_texts_no_punctuation), flags=flags)

        # logging.debug(f"Pattern: {pattern}")
        # print(f"Pattern: {pattern}")

        # Find all matches with their positions in input_query
        # Collect all matches
        all_matches = []
        print(
            f"input_query_no_punctuation in remove_converted_text_preserving_order: {input_query_no_punctuation} \n\n"
        )
        cntr = 0
        unmatched_cntr = 0
        unmatched_texts = []
        for text_no_punct in escaped_texts_no_punctuation:
            # escaped_text = escape_regex(text_no_punct)

            pattern = re.compile(text_no_punct, flags=flags)
            print(
                f"\n Text No Punctuation in remove_converted_text_preserving_order: {text_no_punct} \n Pattern: {pattern} \n\n"
            )
            matches = list(pattern.finditer(input_query_no_punctuation))
            cntr = 0
            for match in matches:
                print(
                    f"Match in remove_converted_text_preserving_order: {match.group()}"
                )
                all_matches.append((match.start(), match.end()))
                cntr += 1
                if cntr > 1:
                    print(
                        f"Multiple Matches Found in remove_converted_text_preserving_order for Text No Punctuation: {text_no_punct} \n\n"
                    )
            if len(matches) < 1:
                print(
                    f"No Matches Found in remove_converted_text_preserving_order for Text No Punctuation: {text_no_punct} \n\n"
                )
                unmatched_texts.append(text_no_punct)
                unmatched_cntr += 1
        if unmatched_cntr > 0 or len(unmatched_texts) > 0:
            print(
                f"Unmatched Texts in remove_converted_text_preserving_order: {unmatched_texts} \n\n"
            )
            print(
                f"Unmatched Counter in remove_converted_text_preserving_order: {unmatched_cntr} \n\n"
            )
            print(
                f"Unmatched Texts Length in remove_converted_text_preserving_order: {len(unmatched_texts)} \n\n"
            )

        # Remove duplicate matches and sort
        all_matches = list(set(all_matches))
        all_matches.sort(key=lambda x: x[0])

        # Proceed with merging overlapping matches as before
        texts_to_remove_str = " ".join(texts_to_remove)
        if len(all_matches) == 0:

            print(
                f"\n\nNo matches found in remove_converted_text_preserving_order. Let's compare the lengths of the input_query and the texts to remove. Then, we will calculate the Jaccard Similarity and Cosine Similarity. \n\n"
            )
            print(f"all_matches: {all_matches} \n\n")
            print(f"Input Query Length: {len(input_query)} \n\n")

            print(f"\n \n Texts to Remove Length: {len(texts_to_remove_str)} \n\n")
            print(f"\nInput Query: {input_query} \n\n")
            print(f"\nTexts to Remove: {texts_to_remove} \n\n")
            exit(3)
            if isinstance(unique_texts_to_remove_no_punctuation, list):
                unique_texts_to_remove_no_punctuation_str = " ".join(
                    unique_texts_to_remove_no_punctuation
                )
                texts_to_remove_no_punctuation_str = " ".join(
                    texts_to_remove_no_punctuation
                )
            elif isinstance(unique_texts_to_remove_no_punctuation, str):
                unique_texts_to_remove_no_punctuation_str = (
                    unique_texts_to_remove_no_punctuation
                )
                texts_to_remove_no_punctuation_str = (
                    texts_to_remove_no_punctuation
                    if isinstance(texts_to_remove_no_punctuation, str)
                    else " ".join(texts_to_remove_no_punctuation)
                )
            print(
                f"Jaccard Similarity: {jaccard_similarity(input_query_no_punctuation, texts_to_remove_no_punctuation_str)} \n\n"
            )
            cosine_sim = cosine_similarity_custom(
                get_embedding(input_query_no_punctuation),
                get_embedding(texts_to_remove_no_punctuation_str),
            )
            print(f"Cosine Similarity: {cosine_sim} \n\n")

            return input_query, input_query  # No matches found
        else:
            match_texts = [input_query[match[0] : match[1]] for match in all_matches]
            print(
                f"Matches No Punctuation in remove_converted_text_preserving_order: {match_texts} \n\n"
            )
        # Merge overlapping matches
        # Merge overlapping matches
        merged_matches = []
        if all_matches:
            prev_start, prev_end = all_matches[0]
            for start, end in all_matches[1:]:
                if start <= prev_end:
                    prev_end = max(prev_end, end)
                else:
                    merged_matches.append((prev_start, prev_end))
                    prev_start, prev_end = start, end
            merged_matches.append((prev_start, prev_end))

        # Remove each match in merged_matches from all_matches
        merged_matches = set(merged_matches)
        all_matches = [m for m in all_matches if m not in merged_matches]
        print(
            f"All Matches after merged_matches loop: {all_matches} \n\n Merge Matches: {merged_matches} \n\n"
        )
        all_matches = list(set(all_matches))
        all_matches.sort(key=lambda x: x[0])

        # Check if any matches were not merged
        def check_unmerged_matches():
            try:
                unmerged_matches = [m for m in all_matches if m not in merged_matches]
                false_unmerged = []
                for m in unmerged_matches:
                    print(
                        f"Unmerged Match in remove_converted_text_preserving_order: {m} \n Match Text: {input_query[m[0]:m[1]]} \n\n"
                    )
                    # Check if the unmerged match is a subset of any merged match
                    subset = False
                    subset_of = None
                    super_set = False
                    super_set_of = None
                    replacements = {}
                    for merged in merged_matches:
                        if m[0] >= merged[0] and m[1] <= merged[1]:
                            subset = True
                            subset_of = merged
                            if m not in false_unmerged:
                                false_unmerged.append(m)
                            break
                        elif merged[0] >= m[0] and merged[1] <= m[1]:
                            super_set = True
                            super_set_of = merged
                            false_unmerged.append(merged)
                            replacements[merged] = m

                    # If the unmerged match is not a subset of any merged match, print a warning
                    merged_matches_to_remove = []
                    new_merged_matches = []
                    for match_text in match_texts:
                        if input_query[m[0] : m[1]] in remove_punctuation(
                            escape_regex(match_text)
                        ):
                            # It was a subset of a match that might or might not have been merged

                            subset = True
                            for mg_match in merged_matches:
                                new_merged = None
                                if (
                                    remove_punctuation(escape_regex(match_text))
                                    in input_query[mg_match[0] : mg_match[1]]
                                ):
                                    # This indicates that the unmerged match is a subset of a merged match and so is remove_punctuation(escape_regex(match_text))
                                    if m[0] >= mg_match[0] and m[1] <= mg_match[1]:
                                        # If the start and end indices of the unmerged match are within the merged match, that means the unmerged match is a subset of the merged match
                                        subset = True
                                        subset_of = mg_match
                                        if m not in false_unmerged:
                                            false_unmerged.append(m)
                                    elif mg_match[0] >= m[0] and mg_match[1] <= m[1]:
                                        # If the start and end indices of the merged match are within the unmerged match, this probably indicates that some text was merged incorrectly
                                        raise Exception(
                                            f" Merged Match start and end indices within Unmerged Match Even Though Unmerged Match text is in Merged Match: {m} \n\n Merged Match: {mg_match} \n\n"
                                        )
                                    else:
                                        print(
                                            f"mg_match[0]: {mg_match[0]} \n\n m[0]: {m[0]} \n\n mg_match[1]: {mg_match[1]} \n\n m[1]: {m[1]} \n\n"
                                        )
                                elif m[0] >= mg_match[0] and m[1] <= mg_match[1]:
                                    # If the start and end indices of the unmerged match are within the merged match, that means the unmerged match is a subset of the merged match
                                    # However, we need to do some further checks on the text. Start with simple checks, then move to more complex ones like regex and similarity measures
                                    if input_query[m[0] : m[1]] in input_query[
                                        mg_match[0] : mg_match[1]
                                    ] and len(input_query[m[0] : m[1]]) < len(
                                        input_query[mg_match[0] : mg_match[1]]
                                    ):
                                        subset_of = mg_match
                                        if m not in false_unmerged:
                                            false_unmerged.append(m)
                                    elif (
                                        input_query[mg_match[0] : mg_match[1]]
                                        in input_query[m[0] : m[1]]
                                    ):
                                        # This indicates that the merged match is a subset of the unmerged match, despite the indices. This is likely an error in merging
                                        raise Exception(
                                            f"Merged Match is a Subset of Unmerged Match even though indices don't indicate that: {m} \n\n Merged Match: {mg_match} \n\n"
                                        )
                                    else:
                                        # More complex checks needed since the indices indicate a subset relationship. Try regex and similarity measures
                                        pat = re.compile(
                                            input_query[m[0] : m[1]],
                                            flags=re.IGNORECASE,
                                        )
                                        mg_pat = re.compile(
                                            input_query[mg_match[0] : mg_match[1]],
                                            flags=re.IGNORECASE,
                                        )
                                        m_match = pat.search(
                                            input_query[mg_match[0] : mg_match[1]]
                                        )
                                        mg_match = mg_pat.search(
                                            input_query[m[0] : m[1]]
                                        )
                                        if m_match:
                                            # The unmerged match text is found in the merged match text
                                            subset_of = mg_match
                                            if m not in false_unmerged:
                                                false_unmerged.append(m)
                                        elif mg_match:
                                            # The merged match text is found in the unmerged match text
                                            raise Exception(
                                                f"Merged Match Text Found in Unmerged Match Text even though indices indicate subset relationship: {m} \n\n Merged Match: {mg_match} \n\n"
                                            )
                                        else:
                                            # Use similarity measures
                                            jaccard_sim = jaccard_similarity(
                                                input_query[m[0] : m[1]],
                                                input_query[mg_match[0] : mg_match[1]],
                                            )
                                            cosine_sim = cosine_similarity_custom(
                                                get_embedding(input_query[m[0] : m[1]]),
                                                get_embedding(
                                                    input_query[
                                                        mg_match[0] : mg_match[1]
                                                    ]
                                                ),
                                            )
                                            if jaccard_sim > 0.5 and cosine_sim > 0.7:
                                                # The texts are similar enough to indicate a subset relationship
                                                subset_of = mg_match
                                                if m not in false_unmerged:
                                                    false_unmerged.append(m)
                                            else:
                                                # The texts are not similar enough to indicate a subset relationship
                                                raise Exception(
                                                    f"Unmerged Match Text Not Subset of Merged Match Text: {m} \n\n Merged Match: {mg_match} \n\n"
                                                )

                                elif mg_match[0] >= m[0] and mg_match[1] <= m[1]:
                                    # If the start and end indices of the merged match are within the unmerged match, that means the merged match is a subset of the unmerged match and should be replaced by the unmerged match
                                    if input_query[
                                        mg_match[0] : mg_match[1]
                                    ] in input_query[m[0] : m[1]] and (
                                        mg_match[1] - mg_match[0]
                                    ) < (
                                        m[1] - m[0]
                                    ):
                                        merged_matches_to_remove.append(mg_match)
                                        new_merged_matches.append((m[0], m[1]))

                                elif input_query[
                                    mg_match[0] : mg_match[1]
                                ] in remove_punctuation(escape_regex(match_text)):
                                    if mg_match[0] >= m[0] and mg_match[1] <= m[1]:
                                        # If the start and end indices of the merged match are within the unmerged match, that means the merged match is a subset of the unmerged match and should be replaced by the unmerged match
                                        merged_matches_to_remove.append(mg_match)
                                        new_merged_matches.append((m[0], m[1]))

                                    elif m[0] >= mg_match[0] and m[1] <= mg_match[1]:
                                        # If the start and end indices of the unmerged match are within the merged match, that means the unmerged match is a subset of the merged match
                                        subset_of = mg_match
                                        if m not in false_unmerged:
                                            false_unmerged.append(m)
                                    else:
                                        print(
                                            f"mg_match[0]: {mg_match[0]} \n\n m[0]: {m[0]} \n\n mg_match[1]: {mg_match[1]} \n\n m[1]: {m[1]} \n\n"
                                        )
                                elif re.search(
                                    re.compile(
                                        remove_punctuation(escape_regex(match_text)),
                                        flags=re.IGNORECASE,
                                    ),
                                    input_query[mg_match[0] : mg_match[1]],
                                ):
                                    subset_of = mg_match
                                    if m not in false_unmerged:
                                        false_unmerged.append(m)
                                elif re.search(
                                    re.compile(
                                        input_query[mg_match[0] : mg_match[1]],
                                        flags=re.IGNORECASE,
                                    ),
                                    remove_punctuation(escape_regex(match_text)),
                                ):
                                    merged_matches_to_remove.append(mg_match)
                                    new_merged_matches.append((m[0], m[1]))
                                elif re.search(
                                    re.compile(
                                        input_query[m[0] : m[1]], flags=re.IGNORECASE
                                    ),
                                    input_query[mg_match[0] : mg_match[1]],
                                ):
                                    subset_of = mg_match
                                    if m not in false_unmerged:
                                        false_unmerged.append(m)
                                elif re.search(
                                    re.compile(
                                        input_query[mg_match[0] : mg_match[1]],
                                        flags=re.IGNORECASE,
                                    ),
                                    input_query[m[0] : m[1]],
                                ):
                                    merged_matches_to_remove.append(mg_match)
                                    new_merged_matches.append((m[0], m[1]))
                                else:
                                    # More complex checks needed since the indices indicate a subset relationship. Try regex and similarity measures
                                    pat = re.compile(
                                        input_query[m[0] : m[1]], flags=re.IGNORECASE
                                    )
                                    mg_pat = re.compile(
                                        input_query[mg_match[0] : mg_match[1]],
                                        flags=re.IGNORECASE,
                                    )
                                    m_match = pat.search(
                                        input_query[mg_match[0] : mg_match[1]]
                                    )
                                    mg_match = mg_pat.search(input_query[m[0] : m[1]])
                                    if m_match:
                                        # The unmerged match text is found in the merged match text
                                        subset_of = mg_match
                                        if m not in false_unmerged:
                                            false_unmerged.append(m)
                                    elif mg_match:
                                        # The merged match text is found in the unmerged match text
                                        raise Exception(
                                            f"Merged Match Text Found in Unmerged Match Text even though indices indicate subset relationship: {m} \n\n Merged Match: {mg_match} \n\n"
                                        )
                                    else:

                                        raise Exception(
                                            f"Unmerged Match Text Not Subset of Merged Match Text: {m} \n\n Merged Match: {mg_match} \n\n"
                                        )

                            if m not in false_unmerged:
                                false_unmerged.append(m)
                        elif (
                            remove_punctuation(escape_regex(match_text))
                            in input_query[m[0] : m[1]]
                        ):
                            # It was a superset of a match that might or might not have been merged
                            print(
                                f"Match Text: {remove_punctuation(escape_regex(match_text))} \n\n"
                            )
                            # If the remove_punctuation(escape_regex(match_text)) is in merged_matches, remove it and add the unmerged match

                            for mg_match in merged_matches:
                                if (
                                    remove_punctuation(escape_regex(match_text))
                                    in input_query[mg_match[0] : mg_match[1]]
                                    or input_query[mg_match[0] : mg_match[1]]
                                    in remove_punctuation(escape_regex(match_text))
                                    or remove_punctuation(escape_regex(match_text))
                                    == input_query[mg_match[0] : mg_match[1]]
                                ) and m not in merged_matches:
                                    replacements[mg_match] = m
                                    # Add the unmerged match to false_unmerged
                                    if m not in false_unmerged:
                                        false_unmerged.append(m)

                                # Add the unmerged match to false_unmerged
                            else:
                                # Add the the superset match to merged_matches
                                if m not in merged_matches:
                                    merged_matches.append(m)
                            # Add the unmerged match to false_unmerged (assuming m was successfully added to merged_matches)
                            if m not in false_unmerged and m in m:
                                false_unmerged.append(m)

                    if not subset:
                        print(
                            f"Unmerged Match Not Subset of merged_match entry: {m} \n\n"
                        )
                    # If the unmerged match is a subset of a merged match, print the subset information and then remove the unmerged match
                    else:
                        print(
                            f"Unmerged Match IS a Subset of merged_match entry: {m} \n\n m Subset of: {subset_of} \n\n"
                        )
                    if not super_set:
                        print(
                            f"Unmerged Match Not Super Set of merged_match entry: {m} \n\n"
                        )
                    # If the unmerged match is a super set of a merged match, print the super set information and then remove the merged match
                    else:
                        print(
                            f"Unmerged Match IS a Super Set of merged_match entry: {m} \n\n m Super Set of: {super_set_of} \n\n"
                        )
                for false_unmerged_match in false_unmerged:
                    if false_unmerged_match in all_matches:
                        all_matches.remove(false_unmerged_match)
                    if false_unmerged_match in unmerged_matches:
                        unmerged_matches.remove(false_unmerged_match)
                for merged_match in merged_matches_to_remove:
                    if merged_match in merged_matches:
                        merged_matches.remove(merged_match)
                for new_merged_match in new_merged_matches:
                    if new_merged_match not in merged_matches:
                        merged_matches.append(new_merged_match)
                for to_be_replaced, replacement in replacements.items():
                    if to_be_replaced in merged_matches:
                        merged_matches.remove(to_be_replaced)
                    if replacement not in merged_matches:
                        merged_matches.append(replacement)
                print(
                    f"False Unmerged Matches in remove_converted_text_preserving_order: {false_unmerged} \n\n"
                )
            except Exception as e:
                print(
                    f"Error in check_unmerged_matches in remove_converted_text_preserving_order: {e} on line {sys.exc_info()[-1].tb_lineno} \n\n"
                )

        if all_matches:
            all_matches = list(set(all_matches))
            all_matches.sort(key=lambda x: x[0])
        if merged_matches:
            merged_matches = list(set(merged_matches))
            merged_matches.sort(key=lambda x: x[0])

        if len(merged_matches) != len(match_texts) or len(all_matches) > 0:
            check_unmerged_matches()
            if len(all_matches) > 0:
                unmerged_matches_b = [m for m in all_matches if m not in merged_matches]
                print(
                    f"\n Unmerged Matches in remove_converted_text_preserving_order: {all_matches} \n\n"
                )
                print(
                    f"\n Unmerged Matches B in remove_converted_text_preserving_order: {unmerged_matches_b} \n\n"
                )
        print(
            f"Merged Matches in remove_converted_text_preserving_order: {merged_matches} \n\n"
        )
        # Track the parts of input_query to keep
        remaining_parts = []
        removed_parts = []

        last_index = 0
        print(
            f"Text mapping in remove_converted_text_preserving_order: {text_mapping}\n\n"
        )
        # This loop will process the original input_query while using matches based on the punctuation-free version
        for start, end in merged_matches:
            # Map indices back to the original input_query
            orig_start = start
            orig_end = end

            # Extract the corresponding original text from input_query
            original_text = text_mapping[input_query_no_punctuation[start:end]][0]
            # Append text before the match to remaining_parts
            remaining_parts.append(input_query[last_index:orig_start])

            # Append the matched text to removed_parts
            removed_parts.append(original_text)

            # Update last_index
            last_index = orig_end

        # Add any remaining text after the last match
        if last_index < len(input_query):
            remaining_parts.append(input_query[last_index:])
        print(f"Remaining Parts after loop over merged_matches: {remaining_parts} \n\n")
        if len(removed_parts) == 0:

            print(
                f"No text was removed in remove_converted_text_preserving_order. Let's compare the lengths of the input_query and the texts to remove. Then, we will calculate the Jaccard Similarity and Cosine Similarity. \n\n"
            )
            print(f"Input Query Length: {len(input_query)} \n\n")
            print(f"\n \n Texts to Remove Length: {len(texts_to_remove_str)} \n\n")
            print(f"\nInput Query: {input_query} \n\n")
            print(f"\nTexts to Remove: {texts_to_remove} \n\n")
            print(f"\n Texts removed: {removed_parts} \n\n")
            if isinstance(texts_to_remove, list):
                texts_to_remove_str = texts_to_remove_str
            elif isinstance(texts_to_remove, str):
                texts_to_remove_str = texts_to_remove
            print(
                f"Jaccard Similarity: {jaccard_similarity(input_query, texts_to_remove_str)} \n\n"
            )
            cosine_sim = cosine_similarity_custom(
                get_embedding(input_query), get_embedding(texts_to_remove_str)
            )
            print(f"Cosine Similarity: {cosine_sim} \n\n")
            return input_query, ""

        # Join the remaining and removed parts
        remaining_parts = [
            part
            for part in remaining_parts
            if len(remove_junk_patterns(part).split()) >= 5
        ]
        print(f"Remaining Parts after removing junk patterns: {remaining_parts} \n\n")

        # remaining_parts = [
        #     part
        #     for part in remaining_parts
        #     if len(part.split()) >= 5 and is_text_readable(part)
        # ]
        # print(f"Remaining Parts after removing unreadable text: {remaining_parts} \n\n")
        # remaining_parts = [
        #     is_real_words(remove_junk_patterns(part)) for part in remaining_parts
        # ]
        # print(f"Remaining Parts after removing non-real words: {remaining_parts} \n\n")

        # Remove non-sentences
        # remaining_parts = [remove_non_sentences(part) for part in remaining_parts]
        # print(f"Remaining Parts after removing non-sentences: {remaining_parts} \n\n")
        not_useful_parts = []
        for part in remaining_parts:
            classification_ = classify_remaining_text_structured(part)
            if not classification_.is_useful:
                not_useful_parts.append(part)
        for part in not_useful_parts:
            remaining_parts.remove(part)
        print(f"Not Useful Parts: {not_useful_parts} \n\n")
        print(f"Remaining Parts after last match: {remaining_parts} \n\n")
        remaining_text = "".join(remaining_parts)
        converted_text = "".join(removed_parts)

        # Clean up whitespace
        remaining_text = " ".join(remaining_text.split())
        converted_text = " ".join(converted_text.split())
        assert isinstance(remaining_text, str)
        assert isinstance(converted_text, str)
        print(f"Remaining Text: {remaining_text} \n\n")
        print(f"Converted Text: {converted_text} \n\n")
        print(f"Remaining Text Length: {len(remaining_text)} \n\n")
        print(f"Converted Text Length: {len(converted_text)} \n\n")
        if (
            len(
                re.sub(
                    r"\s+",
                    "",
                    normalize_whitespace(remove_punctuation(remaining_text)).strip(),
                )
            )
            == 0
        ):
            # If the remaining text is empty, return the original input_query because we are finished
            print(
                f"Successfully removed all converted text in remove_converted_text_preserving_order. \n\n"
            )
            return "", converted_text
        if (
            len(
                re.sub(
                    r"\s+", "", normalize_whitespace(remove_punctuation(converted_text))
                ).strip()
            )
            == 0
        ):
            # If the converted text is empty, this means that nothing was removed from the input_query
            print(
                f"No text was removed in remove_converted_text_preserving_order. Let's compare the lengths of the input_query and the texts to remove. Then, we will calculate the Jaccard Similarity and Cosine Similarity. \n\n"
            )
            print(f"Input Query Length: {len(input_query)} \n\n")
            print(f"\n \n Texts to Remove Length: {len(texts_to_remove_str)} \n\n")
            print(f"\nInput Query: {input_query} \n\n")
            print(f"\nTexts to Remove: {texts_to_remove} \n\n")
            if isinstance(texts_to_remove, list):
                texts_to_remove_str = texts_to_remove_str
            elif isinstance(texts_to_remove, str):
                texts_to_remove_str = texts_to_remove
            print(
                f"Jaccard Similarity: {jaccard_similarity(input_query, texts_to_remove_str)} \n\n"
            )
            cosine_sim = cosine_similarity_custom(
                get_embedding(input_query), get_embedding(texts_to_remove_str)
            )
            print(f"Cosine Similarity: {cosine_sim} \n\n")
            raise Exception(
                "No text was removed in remove_converted_text_preserving_order"
            )
            return input_query, ""
        print(
            f"Jaccard Similarity: {jaccard_similarity(remaining_text, converted_text)} \n\n"
        )
        cosine_sim = cosine_similarity_custom(
            get_embedding(remaining_text), get_embedding(converted_text)
        )
        print(f"Cosine Similarity: {cosine_sim} \n\n")

        return remaining_text, converted_text
    except Exception as e:
        print(
            f"Error in remove_converted_text_preserving_order: {e} Error on line: {sys.exc_info()[-1].tb_lineno}"
        )
        return input_query, ""


def convert_plan(
    input_query: str, existing_plan: Plan = None, recursive_count: int = 1
) -> Plan:
    """
    Converts a step-by-step plan to a more structured format using OpenAI's GPT model.

    Args:
        input_query (str): The plan to convert.

    Returns:
        Plan: Structured representation of the plan.

    """
    model = "gpt-4o-mini"
    instructions = convert_instruction_a

    plan_token_count = count_tokens(input_query + instructions, model)
    print(f"\n[Language Model] Plan Token Count: {plan_token_count} \n\n")
    with open(save_path, "a") as f:
        f.write(f"\n[Language Model] Plan Token Count: {plan_token_count} \n\n")
    if plan_token_count > 16384:
        instructions = convert_instruction_b
        plan_token_count = count_tokens(input_query + instructions, model)
        if plan_token_count > 16384:
            print("[Language Model] Plan too long for conversion.")
            with open(save_path, "a") as f:
                f.write("[Language Model] Plan too long for conversion.")
            exit()
    max_completion_tokens = 16384 - plan_token_count
    print(
        f"[Language Model] Max Tokens for Plan Conversion: {max_completion_tokens} \n\n"
    )
    with open(save_path, "a") as f:
        f.write(
            f"[Language Model] Max Tokens for Plan Conversion: {max_completion_tokens} \n\n"
        )
    if instructions == convert_instruction_a:

        print(f"[Language Model] Converting Plan with Instruction A \n\n")
        with open(save_path, "a") as f:
            f.write(f"[Language Model] Converting Plan with Instruction A \n\n")
    elif instructions == convert_instruction_b:
        print(f"[Language Model] Converting Plan with Instruction B \n\n")
        with open(save_path, "a") as f:
            f.write(f"[Language Model] Converting Plan with Instruction B \n\n")
    else:
        print(f"[Language Model] Converting Plan without Instructions?? \n\n")
        with open(save_path, "a") as f:
            f.write(f"[Language Model] Converting Plan without Instructions?? \n\n")

    if recursive_count > 1:
        print(f"[Language Model] Recursive Count: {recursive_count} \n\n")
        with open(save_path, "a") as f:
            f.write(f"[Language Model] Recursive Count: {recursive_count} \n\n")
        instructions = instructions.replace(
            "You are an assistant that receives a step-by-step plan and converts it into a structured format by identifying",
            "You are an assistant that receives a PARTIAL step-by-step plan, that is missing some steps and subtasks (usually the beginning parts but can be any portion), and (without generating new content or steps) converts the remaining plan steps and remaining subtasks of each step into a structured format by identifying",
        )
        # add recursive instructions to end of instructions
        instructions = (
            instructions
            + "If the plan is complete and there is no meaningful text to convert, please indicate that the conversion is complete by replying simply with the word 'Complete' and nothing else, rather than an empty response. Remember not to generate new steps or subtasks that are not present in the plan. Only convert the existing steps and subtasks that are part of the plan, simply replying with the word 'Complete' if there is no meaningful text to convert into steps and subtasks."
        )
    try:
        response = client.beta.chat.completions.parse(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": instructions,
                },
                {
                    "role": "user",
                    "content": (
                        f"Parse the following plan and provide a structured representation of the steps and subtasks:\n\n{input_query}"
                        if recursive_count == 1
                        else f"""
Parse the remaining steps and subtasks of the following plan and provide a structured representation. If the plan is complete and there is no meaningful text to convert, please indicate that the conversion is by replying simply with the single word 'Complete' only. This is a recursive call number {recursive_count} with a maximum of 5 recursive calls. 
 - Text that has already been converted should not be repeated, and the conversion should continue from where it left off. 
 - Text that does not pertain to steps and subtasks should be ignored. Text that does not pertain to steps and subtasks or is not meaningful should be skipped.
 - If the plan is complete and there is no meaningful text to convert, please indicate that the conversion is complete by replying simply with 'Complete' rather than an empty response. Examples of such text include: 'Of course, here is a plan for you:', 'Here is the plan:', 'Here is a step-by-step plan:', or even text that may pertain to the plan but is not meaningful to convert, such as 'The following steps will guide you through the process:', or even detailed information about the plan or the topic but is not directly related to the steps and subtasks of achieving the goal of the plan.
 - It is important to maintain the order of the steps and subtasks as they appear in the plan. If the order is not maintained, the conversion will be incorrect.
 - There should also not be any repetition of steps or subtasks. If a step or subtask has already been converted, it should not be repeated in the conversion even if it is reiterated, restated or repeated in the text. 
 - Please also avoid inventing or generating new steps or subtasks that are not present in the plan. Only convert the existing steps and subtasks that are part of the plan.
 Follows is the current plan that the assistant has converted so far:
{existing_plan}
Here is the remaining text to be converted:
{input_query}"""
                    ),
                },
            ],
            n=1,
            stop=None,
            temperature=0.2 if recursive_count > 1 else 0.2 + 0.1 * recursive_count,
            response_format=Plan,
            max_completion_tokens=max_completion_tokens,
        )
        output = None
        if response.choices[0].finish_reason == "length":
            print(f"[Language Model] Plan conversion incomplete due to token limit.")
            with open(save_path, "a") as f:
                f.write(
                    f"[Language Model] Plan conversion incomplete due to token limit."
                )
        if (
            response.choices[0].message.parsed is None
            or response.choices[0].message.parsed == ""
            or response.choices[0].message.content.strip() == ""
        ):
            print(
                f"[Language Model] Plan conversion failed. Empty response with finish reason: {response.choices[0].finish_reason}. \n Refusal?: {response.choices[0].message.refusal} \n For input: {input_query}. \n The existing plan is: {existing_plan}"
            )
            with open(save_path, "a") as f:
                f.write(f"[Language Model] Plan conversion failed.")
            print(
                print(
                    f"\nReturning converted plan from line {inspect.currentframe().f_lineno + 1}\n"
                )
            )
            return ""

        elif existing_plan is not None and re.search(
            r"\bComplete\b", response.choices[0].message.content
        ):
            print(
                f"[Language Model] Plan conversion complete on recursive loop {recursive_count}. \n\n {response.choices[0].message.content} \n\n"
            )
            with open(save_path, "a") as f:
                f.write(
                    f"[Language Model] Plan conversion complete on recursive loop {recursive_count}. \n\n {response.choices[0].message.content} \n\n"
                )
            print(
                print(
                    f"\nReturning converted plan from line {inspect.currentframe().f_lineno + 1}\n"
                )
            )
            return existing_plan
        elif (
            response.choices[0].message.parsed is not None
            and response.choices[0].message.parsed != ""
            and response.choices[0].message.content.strip() != ""
        ):
            print(
                f"\n\n[Language Model] Plan Conversion Response in recursive loop {recursive_count}: {response} \n\n"
            )
            output = response.choices[0].message.parsed
            response_tokens_string = ""
            response_tokens = [token.step_full_text.split() for token in output.steps]
            if response_tokens is not None and len(response_tokens) > 0:
                print(f"[Language Model] Response Tokens Type: {type(response_tokens)}")

                print(
                    f"[Language Model] Response Tokens Item Type: {type(response_tokens[0])}"
                )
                # print(f"[Language Model] Response Tokens: {response_tokens}")
                with open(save_path, "a") as f:
                    f.write(
                        f"[Language Model] Response Tokens Type: {type(response_tokens)}"
                    )
                    f.write(
                        f"[Language Model] Response Tokens Item Type: {type(response_tokens[0])}"
                    )
                    # f.write(f"[Language Model] Response Tokens: {response_tokens}")
                response_tokens.extend(
                    [
                        token.subtask_full_text.split()
                        for step in output.steps
                        for token in step.subtasks
                    ]
                )
                print(f"[Language Model] Response Tokens Type: {type(response_tokens)}")
                print(
                    f"[Language Model] Response Tokens Item Type: {type(response_tokens[0])}"
                )

                # print(f"[Language Model] Response Tokens: {response_tokens}")
                with open(save_path, "a") as f:
                    f.write(
                        f"[Language Model] Response Tokens Type: {type(response_tokens)}"
                    )
                # Convert response_tokens list of strings to a single string
                response_tokens_string = " ".join(
                    [str(token) for token in response_tokens]
                )
                response_tokens_string = (
                    response_tokens_string.replace("[", "")
                    .replace("]", "")
                    .replace("'", "")
                    .replace(",", "")
                )
                assert isinstance(response_tokens_string, str)
                print(
                    f"[Language Model] Response Tokens String Type: {type(response_tokens_string)}"
                )
                print(
                    f"[Language Model] Response Tokens String: {response_tokens_string}"
                )
                with open(save_path, "a") as f:
                    f.write(
                        f"[Language Model] Response Tokens String Type: {type(response_tokens_string)}"
                    )
                    f.write(
                        f"[Language Model] Response Tokens String: {response_tokens_string}"
                    )
            else:
                print(
                    f"[Language Model] Empty Response Tokens: {response_tokens} of type {type(response_tokens)} and length {len(response_tokens)}. Output: {output} \n\n Full Response: {response}"
                )
                with open(save_path, "a") as f:
                    f.write(
                        f"[Language Model] Empty Response Tokens: {response_tokens} of type {type(response_tokens)} and length {len(response_tokens)}. Output: {output} \n\n Full Response: {response}"
                    )

        response_token_count = count_tokens(response_tokens_string, model)
        print(
            f"Token count of response: {response_token_count} out of {max_completion_tokens} max tokens."
        )
        with open(save_path, "a") as f:
            f.write(
                f"Token count of response: {response_token_count} out of {max_completion_tokens} max tokens."
            )
        if (
            existing_plan is not None
            and recursive_count > 1
            and recursive_count <= 5
            and output is not None
            and output.steps is not None
            and len(output.steps) > 0
        ):
            # Merge the newly converted steps and subtasks with the existing plan. If the step_number already exists, then we will compare the cosine similarity of the new step and the existing step. If the similarity is high, we will not add the new step to the plan.
            # Check for duplicate steps, favoring the existing step if the cosine similarity is high and appending/adding the new step to the old step if the cosine similarity is low.
            # Also, check for duplicate subtasks within the same step, favoring the existing subtask if the cosine similarity is high and appending/adding the new subtask to the old subtask if the cosine similarity is low.

            # Before initializing the vectorizer, lets think step by step the best way to use it for this use case. For example, what, if any, arguments should be passed to the TfidfVectorizer constructor? What methods should be called on the vectorizer object? What should be done with the output of those methods?
            # Thinking on it, the best way to use the TfidfVectorizer is to use the default arguments. The default arguments are already set to the best values for this use case. The TfidfVectorizer will be used to convert the text to count vectors. The fit_transform method will be called on the vectorizer object to convert the text to count vectors. The output of the fit_transform method will be stored in a variable called count_matrix. The cosine_similarity_custom method will be called on the count_matrix to calculate the cosine similarity between the existing step and the new step. The cosine similarity will be stored in a variable called similarity.
            print(
                f"\n[Language Model] This is a recursive call. Current/last plan: {existing_plan} \n\n"
            )
            for step in output.steps:
                existing_step = next(
                    (
                        existing_step
                        for existing_step in existing_plan.steps
                        if existing_step.step_number == step.step_number
                    ),
                    None,
                )
                if existing_step is not None:
                    if isinstance(existing_step, list) and len(existing_step) > 1:
                        # This indicates that the step has been added multiple times. We need to remove the duplicates by merging them. Call the llm to merge the steps.
                        print(
                            f"[Language Model] Duplicate step found in existing plan. Merging steps: {existing_step} \n\n"
                        )
                        with open(save_path, "a") as f:
                            f.write(
                                f"[Language Model] Duplicate step found in existing plan. Merging steps: {existing_step} \n\n"
                            )
                        merged = merge_steps(existing_step, step)
                        if (
                            merged
                            and merged.step_full_text != existing_step.step_full_text
                        ):
                            existing_step.step_full_text = merged.step_full_text
                            existing_step.subtasks = merged.subtasks

                    existing_step_text = existing_step.step_full_text
                    new_step_text = step.step_full_text

                    similarity = cosine_similarity_custom(
                        get_embedding(existing_step_text), get_embedding(new_step_text)
                    )

                    jac_sim = jaccard_similarity(existing_step_text, new_step_text)
                    print(
                        f"[Language Model] Cosine Similarity between existing step and new step: {similarity}"
                    )
                    with open(save_path, "a") as f:
                        f.write(
                            f"[Language Model] Cosine Similarity between existing step and new step: {similarity}"
                        )
                    if similarity < 0.6 and jac_sim < 0.6:
                        step.step_full_text = (
                            existing_step.step_full_text + " " + step.step_full_text
                        )
                    else:
                        for subtask in step.subtasks:
                            existing_subtask = next(
                                (
                                    existing_subtask
                                    for existing_subtask in existing_step.subtasks
                                    if existing_subtask.subtask_number
                                    == subtask.subtask_number
                                ),
                                None,
                            )
                            if existing_subtask is not None:
                                existing_subtask_text = (
                                    existing_subtask.subtask_full_text
                                )
                                new_subtask_text = subtask.subtask_full_text

                                # Convert texts to count vectors

                                similarity = cosine_similarity_custom(
                                    get_embedding(existing_subtask_text),
                                    get_embedding(new_subtask_text),
                                )

                                print(
                                    f"[Language Model] Cosine Similarity between existing subtask and new subtask: {similarity}"
                                )
                                with open(save_path, "a") as f:
                                    f.write(
                                        f"[Language Model] Cosine Similarity between existing subtask and new subtask: {similarity}"
                                    )
                                if similarity < 0.7 or jac_sim < 0.7:
                                    existing_step.subtasks.append(subtask)
                else:
                    existing_plan.steps.append(step)

            print(f"\n[Language Model] Merging Plans: {existing_plan} \n\n")
            with open(save_path, "a") as f:
                f.write(
                    f"\n[Language Model] This is a recursive call. Current/last plan: {existing_plan} \n\n"
                )
                f.write(f"\n[Language Model] Merging Plans: {existing_plan} \n\n")
        elif output is None or output.steps is None or len(output.steps) == 0:
            print(
                f"[Language Model] Plan conversion failed. No steps or subtasks found in response. \n\n {response} \n\n"
            )
            with open(save_path, "a") as f:
                f.write(
                    f"[Language Model] Plan conversion failed. No steps or subtasks found in response. \n\n {response} \n\n"
                )
            if existing_plan is not None:
                print(
                    print(
                        f"\nReturning converted plan from line {inspect.currentframe().f_lineno + 1}\n"
                    )
                )
                return existing_plan
            else:
                raise Exception(
                    "Plan conversion failed. No steps or subtasks found in response."
                )

        elif recursive_count > 5:
            print(
                f"[Language Model] Plan conversion incomplete after 5 recursive calls."
            )
            with open(save_path, "a") as f:
                f.write(
                    f"[Language Model] Plan conversion incomplete after 5 recursive calls."
                )
            new_input_query = input_query
            steps = output.steps
            # Order the steps by their step_number
            steps = sorted(steps, key=lambda x: x.step_number)

            new_input_query, converted_text = remove_converted_text_preserving_order(
                new_input_query, steps
            )
            # Check if either the new_input_query or the converted_text are empty. If new_input_query is empty, then the conversion is complete. If converted_text is empty, then nothing was converted in the last recursive loop.
            if normalize_whitespace(remove_punctuation(new_input_query)).strip() == "":
                print(
                    f"[Language Model] Plan conversion complete. Returning the final structured plan. Response token count: {response_token_count}. Plan token count: {plan_token_count}, Recursive count: {recursive_count}"
                )
                print(
                    f"\nReturning converted plan from line {inspect.currentframe().f_lineno + 1}\n"
                )
                return existing_plan
            if normalize_whitespace(remove_punctuation(converted_text)).strip() == "":
                print(
                    f"[Language Model] Plan conversion incomplete in last recursive loop. Remaining text to be converted: {new_input_query}"
                )
                with open(save_path, "a") as f:
                    f.write(
                        f"[Language Model] Plan conversion incomplete in last recursive loop. Remaining text to be converted: {new_input_query}"
                    )
                print(
                    f"\nReturning converted plan from line {inspect.currentframe().f_lineno + 1}\n"
                )

                return existing_plan
            print(
                f"[Language Model] Plan conversion incomplete in last recursive loop. Remaining text to be converted: {new_input_query}"
            )
            with open(save_path, "a") as f:
                f.write(
                    f"[Language Model] Plan conversion incomplete in last recursive loop. Remaining text to be converted: {new_input_query}"
                )
            print(
                f"\nReturning converted plan from line {inspect.currentframe().f_lineno + 1}\n"
            )

            return existing_plan

        if (
            response_token_count < plan_token_count
            and plan_token_count - response_token_count > 10
        ):

            # Some parts of the plan may not be converted. We need to re-run the conversion with the remaining text.
            # We will determine the remaining text by comparing the `step_full_text` of each step and `subtask_full_text` of each subtask with the input text. The remaining text will be the part of the input text that was not converted.
            new_input_query = input_query
            steps = output.steps
            # Order the steps by their step_number
            steps = sorted(steps, key=lambda x: x.step_number)

            new_input_query, converted_text = remove_converted_text_preserving_order(
                new_input_query, steps
            )

            if normalize_whitespace(remove_punctuation(converted_text)).strip() == "":
                print(
                    f"[Language Model] Plan conversion incomplete in last recursive loop. Remaining text to be converted: {new_input_query}"
                )
                with open(save_path, "a") as f:
                    f.write(
                        f"[Language Model] Plan conversion incomplete in last recursive loop. Remaining text to be converted: {new_input_query}"
                    )
                if existing_plan is not None:
                    print(
                        f"\nReturning converted plan from line {inspect.currentframe().f_lineno + 1}\n"
                    )
                    return existing_plan
                elif recursive_count == 1:

                    print(
                        f"\nReturning converted plan from line {inspect.currentframe().f_lineno + 1}\n"
                    )

                    return output
                else:
                    print(
                        f"Plan conversion incomplete in last recursive loop. Remaining text to be converted: {new_input_query} Recursive count: {recursive_count}"
                    )

                    raise Exception(
                        f"Plan conversion incomplete in last recursive loop. Remaining text to be converted: {new_input_query} Recursive count: {recursive_count}"
                    )

            if normalize_whitespace(remove_punctuation(new_input_query)).strip() == "":
                print(
                    f"[Language Model] Plan conversion complete. Returning the final structured plan. Response token count: {response_token_count}. Plan token count: {plan_token_count}, Recursive count: {recursive_count}"
                )
                print(
                    f"\nReturning converted plan from line {inspect.currentframe().f_lineno + 1}\n"
                )
                if existing_plan is None:
                    print(
                        f"For some reason, the existing plan is None. Returning the output."
                    )
                return existing_plan if existing_plan is not None else output

            # Calculate the Jaccard similarity between the original input text and the remaining text to be converted
            jaccard_sim = jaccard_similarity(
                normalize_whitespace(remove_punctuation(input_query)),
                normalize_whitespace(remove_punctuation(new_input_query)),
            )
            cosine_sim = cosine_similarity_custom(
                get_embedding(normalize_whitespace(remove_punctuation(input_query))),
                get_embedding(
                    normalize_whitespace(remove_punctuation(new_input_query))
                ),
            )
            # Subtract new_input_query from input_query to get the text that was converted this iteration
            print(
                f"[Language Model] Jaccard Similarity between input_query and new_input_query: {jaccard_sim}. Note that a Jaccard similarity of 1 means the text is the same. These should be different, looking for a value close to 0."
            )
            print(
                f"[Language Model] Cosine Similarity between input_query and new_input_query: {cosine_sim}. Note that a Cosine similarity of 1 means the text is the same. These should be different, looking for a value close to 0."
            )
            # Check if the converted text is the same as the response. If it is too similar, print both texts to check for differences.
            if (
                jaccard_sim > 0.7
                or cosine_sim > 0.7
                or new_input_query == input_query
                or (cosine_sim > 0.5 and jaccard_sim > 0.5)
            ) and (
                abs(len(new_input_query) - len(input_query)) < (len(input_query) * 0.6)
            ):
                print(
                    f"\n[Language Model] input_query text and new_input_query are the same. input_query text: {input_query} \n\n\n new_input_query: {new_input_query}"
                )
                with open(save_path, "a") as f:
                    f.write(
                        f"\n[Language Model] input_query text and new_input_query are the same. input_query text: {input_query} \n\n\n new_input_query: {new_input_query}"
                    )

            jac_sim_converted_response = jaccard_similarity(
                normalize_whitespace(remove_punctuation(converted_text)),
                normalize_whitespace(remove_punctuation(response_tokens_string)),
            )
            cosine_sim_converted_response = cosine_similarity_custom(
                get_embedding(normalize_whitespace(remove_punctuation(converted_text))),
                get_embedding(
                    normalize_whitespace(remove_punctuation(response_tokens_string))
                ),
            )
            print(
                f"\n[Language Model] Jaccard Similarity between converted text and response: {jac_sim_converted_response}. Note that a Jaccard similarity of 1 means the text is the same. Thinking step by step, this should be close to 1 because the converted text should be similar to the response."
            )
            print(
                f"[Language Model] Cosine Similarity between converted text and response: {cosine_sim_converted_response}. Note that a Cosine similarity of 1 means the text is the same. Thinking step by step, this should be close to 1 because the converted text should be similar to the response."
            )

            if (
                jac_sim_converted_response == 0.0
                or jac_sim_converted_response < 0.5
                or cosine_sim_converted_response < 0.5
                or cosine_sim_converted_response == 0.0
            ):
                print(
                    f"\n[Language Model] Converted text and response are different. Converted text: {converted_text}. \nResponse: {response_tokens_string}"
                )
                with open(save_path, "a") as f:
                    f.write(
                        f"\n[Language Model] Converted text and response are different. Converted text: {converted_text}. \nResponse: {response_tokens_string}"
                    )
            with open(save_path, "a") as f:
                f.write(
                    f"[Language Model] Jaccard Similarity: {jaccard_sim}. Note that a Jaccard similarity of 1 means the text is the same."
                )

            # print left over text still to be converted
            print(
                f"[Language Model] Plan conversion incomplete. Remaining text to be converted: {new_input_query} at recursive count {recursive_count}"
            )
            with open(save_path, "a") as f:
                f.write(
                    f"[Language Model] Plan conversion incomplete. Remaining text to be converted: {new_input_query} at recursive count {recursive_count}"
                )
            # We will then recursively call the conversion function with the remaining text until the entire plan is converted.
            # We will also  merge the converted parts of the plan into a single Plan object in the next recursive call.
            if recursive_count == 1 or existing_plan is None:
                if not existing_plan and recursive_count > 1:
                    print(
                        f"[Language Model] Recursive call {recursive_count} with no existing plan. Creating new plan. Must debug why existing plan is None."
                    )

                existing_plan = output

            recursive_count += 1
            if count_tokens(new_input_query, model) > 20:
                print(
                    f"\nReturning converted plan from line {inspect.currentframe().f_lineno + 1}\n"
                )

                return convert_plan(new_input_query, existing_plan, recursive_count)

            print(
                f"[Language Model] Plan conversion incomplete. Re-running conversion for remaining text of length {len(new_input_query)} of the original {len(input_query)} characters."
            )
            with open(save_path, "a") as f:
                f.write(
                    f"[Language Model] Plan conversion incomplete. Re-running conversion for remaining text of length {len(new_input_query)} of the original {len(input_query)} characters."
                )
        else:
            print(
                f"[Language Model] Plan conversion complete. Returning the final structured plan. Response token count: {response_token_count}. Plan token count: {plan_token_count}, Recursive count: {recursive_count}"
            )
            return existing_plan if existing_plan is not None else output
        print(
            print(
                f"\nReturning converted plan from line {inspect.currentframe().f_lineno + 1}\n"
            )
        )
        return existing_plan
    except Exception as e:
        print(
            f"[Language Model] Error generating plan: {e}. Error occurred on line {sys.exc_info()[-1].tb_lineno}"
        )
        print(
            f"\nReturning converted plan from line {inspect.currentframe().f_lineno + 1}\n"
        )
        return ""


def generate_plan_legacy(input_query: str) -> str:
    """
    Generates a step-by-step plan using OpenAI's GPT model.

    Args:
        input_query (str): The problem to solve.

    Returns:
        str: Generated plan as text.
    """
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        print("[Language Model] OpenAI API key not found.")
        return ""
    openai.api_key = openai_api_key

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are an assistant that breaks down problems into step-by-step plans that are easy to follow by an LLM.",
                },
                {
                    "role": "user",
                    "content": f"Provide a detailed, LLM-oriented step-by-step plan to solve the following problem:\n\n{input_query}",
                },
            ],
            max_completion_tokens=2500,
            n=1,
            stop=None,
            temperature=0.5,
        )
        plan = response.choices[0].message.content

        return plan
    except Exception as e:
        print(f"[Language Model] Error generating plan: {e}")
        return ""


def is_complex_llm(
    input_query: str,
    substep_threshold: int = 4,
    depth_threshold: int = 1,
    step_length_threshold: int = 12,
    unique_subtask_threshold: int = 3,
    substep_weight: float = 0.6,
    depth_weight: float = 0.2,
    step_length_weight: float = 0.1,  # New weight for step length
    unique_subtask_weight: float = 0.1,  # New weight for unique subtask types
    sigmoid_steepness: float = 1.0,  # New parameter for sigmoid steepness
) -> Tuple[float, Plan]:
    """
    Determines complexity using LLM-generated plan analysis.

    Args:
        input_query (str): The problem to solve.
        substep_threshold (int): Number of substeps to consider complex.
        depth_threshold (int): Depth of subtasks to consider complex.
        step_length_threshold (int): Average step length to consider complex.
        unique_subtask_threshold (int): Number of unique subtasks to consider complex.
        substep_weight (float): Weight for substeps.
        depth_weight (float): Weight for depth.
        step_length_weight (float): Weight for step length.
        unique_subtask_weight (float): Weight for unique subtasks.
        sigmoid_steepness (float): Steepness of the sigmoid function.

    Returns:
        float: Score between 0 and 1 indicating complexity.
    """
    plan_str = generate_plan_legacy(input_query)
    if not plan_str:
        return 0.0
    print(f"[Language Model] Generated Plan (Legacy):\n{plan_str} \n\n")
    with open(save_path, "a") as f:
        f.write(f"[Language Model] Generated Plan (Legacy):\n{plan_str} \n\n")
    plan = convert_plan(plan_str)
    print(f"[Language Model] Generated Plan:{plan} \n\n")
    print(f"[Language Model] Plan Steps: {len(plan.steps)} \n\n")
    print(f"[Language Model] Plan Type: {type(plan)} \n\n")
    with open(save_path, "a") as f:
        f.write(f"[Language Model] Generated Plan:\n{plan} \n\n")
        f.write(f"[Language Model] Plan Steps: {len(plan.steps)} \n\n")
        f.write(f"[Language Model] Plan Type: {type(plan)} \n\n")
    assert isinstance(plan, Plan)
    if not plan:
        return 0.0

    # Extract Steps class from the generated plan
    steps = plan.steps
    substeps = len(steps)

    # Calculate average step length by tokenizing each step and summing the lengths. Tokenize using spaCy
    total_length = 0

    for step in steps:
        total_length += len(nlp_spacy(step.step_full_text))
        for subtask in step.subtasks:
            total_length += len(nlp_spacy(subtask.subtask_full_text))
    avg_step_length = total_length / substeps if substeps > 0 else 0

    # Collect all subtasks and count unique names
    subtasks = []
    for step in steps:
        subtasks.extend(step.subtasks)
    unique_subtask_count = len(set(subtask.subtask_name for subtask in subtasks))

    depth = 2
    nested_subtasks = 0

    def recursive_subtask_count(subtask, current_depth):
        nested_subtasks = 1  # Count the current subtask
        max_depth = current_depth
        for subsubtask in subtask.subtasks:
            count, depth = recursive_subtask_count(subsubtask, current_depth + 1)
            nested_subtasks += count
            max_depth = max(max_depth, depth)
        return nested_subtasks, max_depth

    total_nested_subtasks = 0
    max_subtask_depth = 0
    print(f"[Language Model] Number of Subtasks: {len(subtasks)}")
    with open(save_path, "a") as f:
        f.write(f"[Language Model] Number of Subtasks: {len(subtasks)}")
    for subtask in subtasks:
        count, depth = recursive_subtask_count(subtask, 2)  # Start at depth 2
        total_nested_subtasks += count
        max_subtask_depth = max(max_subtask_depth, depth)

    calculated_substeps = (substeps - substep_threshold) * substep_weight
    calculated_depth = (max_subtask_depth - depth_threshold) * depth_weight
    calculated_step_length = avg_step_length * step_length_weight
    calculated_unique_subtasks = unique_subtask_count * unique_subtask_weight
    # Normalize scores
    substeps_score = 1 / (
        1 + math.exp(-sigmoid_steepness * calculated_substeps / substep_threshold)
    )
    depth_score = 1 / (
        1 + math.exp(-sigmoid_steepness * calculated_depth / depth_threshold)
    )
    step_length_score = 1 / (
        1
        + math.exp(-sigmoid_steepness * calculated_step_length / step_length_threshold)
    )
    unique_subtask_score = 1 / (
        1
        + math.exp(
            -sigmoid_steepness * calculated_unique_subtasks / unique_subtask_threshold
        )
    )

    # Combined score
    score = (
        substep_weight * substeps_score
        + depth_weight * depth_score
        + step_length_weight * step_length_score
        + unique_subtask_weight * unique_subtask_score
    )
    sigmoid_threshold = 1 / (1 + math.exp(-sigmoid_steepness))
    print(
        f"sigmoid_steepness: {sigmoid_steepness}. Sigmoid Threshold: {sigmoid_threshold} Pre-Sigmoid Score: {score}"
    )
    with open(save_path, "a") as f:
        f.write(
            f"sigmoid_steepness: {sigmoid_steepness}. Sigmoid Threshold: {sigmoid_threshold} Pre-Sigmoid Score: {score}"
        )

    # score = 1 / (1 + math.exp(-sigmoid_steepness * (score - sigmoid_threshold)))
    print(
        f"[Language Model] Substeps: {substeps}, Nested Subtasks: {total_nested_subtasks}, Score: {score}, Depth Score: {depth_score}, Substeps Score: {substeps_score} at a depth of {max_subtask_depth}"
    )
    with open(save_path, "a") as f:
        f.write(
            f"[Language Model] Substeps: {substeps}, Nested Subtasks: {total_nested_subtasks}, Score: {score}, Depth Score: {depth_score}, Substeps Score: {substeps_score} at a depth of {max_subtask_depth}"
        )
    return (score, plan)


def is_complex_llm_legacy(
    input_query: str,
    substep_threshold: int = 10,
    depth_threshold: int = 2,
    substep_weight: float = 0.8,
    depth_weight: float = 0.2,
) -> float:
    """
    Determines complexity using LLM-generated plan analysis.

    Args:
        input_query (str): The problem to solve.
        substep_threshold (int): Number of substeps to consider complex.
        depth_threshold (int): Depth of subtasks to consider complex.

    Returns:
        float: Score between 0 and 1 indicating complexity.
    """
    plan = generate_plan(input_query)
    if not plan:
        return 0.0
    print(f"[Language Model] Generated Plan:\n{plan}")

    # Extract steps using regex (assuming steps are numbered or bulleted)
    steps = re.findall(r"^\s*(?:\d+\.|\-|\*|\•)\s*(.*)", plan, re.MULTILINE)
    substeps = len(steps)

    # Estimate depth by counting nested steps (e.g., substeps indicated by indentation or additional numbering)
    # For comprehensiveness, let's not assume a specific format for subtasks
    depth = 2
    nested_subtasks = 0
    while True:
        pattern = rf"^\s{{{depth},}}.*"
        if not re.search(pattern, plan, re.MULTILINE):
            break
        nested_subtasks += len(re.findall(pattern, plan, re.MULTILINE))
        depth += 1
    if nested_subtasks == 0:

        # If nesting isnt detected, try to detect subtasks with a different pattern
        nested_subtasks = len(re.findall(r"^\s*-\s*.*", plan, re.MULTILINE))
        print(
            f"[Language Model] No nested subtasks detected. Trying a different pattern. Latest Nested Subtasks: {nested_subtasks}"
        )

    if nested_subtasks == 0:
        # Pattern to detect nested subtasks (lines starting with optional whitespace and a bullet point)
        nested_subtasks_pattern = r"^\s*[-*•]\s*.*"
        nested_subtasks = len(re.findall(nested_subtasks_pattern, plan, re.MULTILINE))

        # Pattern to detect top-level tasks (lines starting with a bullet point)
        top_level_tasks_pattern = r"^[-*•]\s*.*"
        top_level_tasks = len(re.findall(top_level_tasks_pattern, plan, re.MULTILINE))

        # Remove counts of all the top-level tasks/subtasks from nested_subtasks
        nested_subtasks -= top_level_tasks
        print(
            f"[Language Model] No nested subtasks detected. Trying a different pattern a second time. Latest Nested Subtasks: {nested_subtasks}"
        )

    depth = depth - 1  # Subtract 1 to account for the main task
    max_depth = nested_subtasks + 1  # Add 1 to account for the main task
    calculated_substeps = (substeps - substep_threshold) * substep_weight
    calculated_depth = (depth - depth_threshold) * depth_weight
    # Normalize scores
    substeps_score = 1 / (1 + math.exp(-calculated_substeps / substep_threshold))
    depth_score = 1 / (1 + math.exp(-calculated_depth / depth_threshold))

    # Combined score
    score = substep_weight * substeps_score + depth_weight * depth_score
    score = 1 / (1 + math.exp(-score))
    print(
        f"[Language Model] Substeps: {substeps}, Nested Subtasks: {nested_subtasks}, Score: {score}, Depth Score: {depth_score}, Substeps Score: {substeps_score} at a depth of {depth}"
    )
    return score


# 6. Graph-Based Approach
def is_complex_graph(
    input_query: str, substep_threshold: int = 5, depth_threshold: int = 1
) -> float:
    """
    Determines complexity based on a graph representation of tasks.

    Args:
        input_query (str): The problem to solve.
        substep_threshold (int): Number of tasks to consider complex.
        depth_threshold (int): Depth of task dependencies to consider complex.

    Returns:
        float: Score between 0 and 1 indicating complexity.
    """
    doc = nlp_spacy(input_query)

    G = nx.DiGraph()

    # Simple heuristic: Each verb is a task; dependencies based on subjects and objects
    for token in doc:
        if token.pos_ == "VERB":
            task = token.lemma_
            G.add_node(task)
            for child in token.children:
                if child.dep_ in ("nsubj", "dobj", "prep"):
                    if child.pos_ == "NOUN":
                        dependent = child.lemma_
                        G.add_edge(task, dependent)

    substeps = G.number_of_nodes()

    try:
        longest_path = nx.dag_longest_path(G)
        max_depth = len(longest_path) - 1 if len(longest_path) > 1 else 0
    except nx.NetworkXUnfeasible:
        # Graph has cycles; consider it complex
        max_depth = depth_threshold + 1

    # Normalize scores
    substeps_score = 1 / (1 + math.exp(-substeps / substep_threshold))
    depth_score = 1 / (1 + math.exp(-max_depth / depth_threshold))

    # Combined score
    score = substeps_score * depth_score
    print(
        f"[Graph-Based Approach] Substeps: {substeps}, Max Depth: {max_depth}, Score: {score}"
    )
    with open(save_path, "a") as f:
        f.write(
            f"[Graph-Based Approach] Substeps: {substeps}, Max Depth: {max_depth}, Score: {score}"
        )
    return score


# 7. Recursive Task Decomposition
def is_complex_recursive(
    input_query: str,
    substep_threshold: int = 5,
    depth_threshold: int = 1,
    max_allowed_depth: int = 3,
) -> float:
    """
    Determines complexity using recursive task decomposition.

    Args:
        input_query (str): The problem to solve.
        substep_threshold (int): Number of tasks to consider complex.
        depth_threshold (int): Depth of task dependencies to consider complex.
        max_allowed_depth (int): Maximum recursion depth.

    Returns:
        float: Score between 0 and 1 indicating complexity.
    """
    doc = nlp_spacy(input_query)

    G = nx.DiGraph()

    # Identify main tasks
    main_tasks = [token for token in doc if token.pos_ == "VERB"]

    def decompose_task(token, current_depth, max_allowed_depth, G):
        """
        Recursively decomposes a task into subtasks.
        """
        if current_depth > max_allowed_depth:
            return
        for child in token.children:
            if child.dep_ in ("xcomp", "ccomp", "conj") and child.pos_ == "VERB":
                G.add_edge(token.lemma_, child.lemma_)
                decompose_task(child, current_depth + 1, max_allowed_depth, G)

    for task in main_tasks:
        G.add_node(task.lemma_)
        decompose_task(task, 1, max_allowed_depth, G)

    substeps = G.number_of_nodes()

    try:
        longest_path = nx.dag_longest_path(G)
        max_depth = len(longest_path) - 1 if len(longest_path) > 1 else 0
    except nx.NetworkXUnfeasible:
        max_depth = depth_threshold + 1

    # Normalize scores
    substeps_score = 1 / (1 + math.exp(-substeps / substep_threshold))
    depth_score = 1 / (1 + math.exp(-max_depth / depth_threshold))

    # Combined score
    score = substeps_score * depth_score
    print(
        f"[Recursive Task Decomposition] Substeps: {substeps}, Max Depth: {max_depth}, Score: {score}"
    )
    with open(save_path, "a") as f:
        f.write(
            f"[Recursive Task Decomposition] Substeps: {substeps}, Max Depth: {max_depth}, Score: {score}"
        )
    return score


# 8. Ontological Mapping
def is_complex_ontology(input_query: str, max_depth_threshold: int = 3) -> float:
    """
    Determines complexity based on ontological mapping using WordNet.

    Args:
        input_query (str): The problem to solve.
        max_depth_threshold (int): Depth in ontology to consider complex.

    Returns:
        float: Score between 0 and 1 indicating complexity.
    """
    tokens = nltk.word_tokenize(input_query)
    nouns = [word for word, pos in nltk.pos_tag(tokens) if pos.startswith("NN")]

    max_depth = 0
    for noun in nouns:
        synsets = wn.synsets(noun, pos=wn.NOUN)
        if not synsets:
            continue
        synset = synsets[0]
        depth = synset.max_depth()
        if depth > max_depth:
            max_depth = depth

    # Normalize score using sigmoid function and threshold
    score = (
        1 / (1 + math.exp(-max_depth - max_depth_threshold)) / (max_depth_threshold * 2)
    )
    print(f"[Ontological Mapping] Max Ontology Depth: {max_depth}, Score: {score}")
    with open(save_path, "a") as f:
        f.write(
            f"[Ontological Mapping] Max Ontology Depth: {max_depth}, Score: {score}"
        )
    return score


# 9. Cognitive Complexity Metrics
def is_complex_cognitive(input_query: str, complexity_threshold: float = 5.0) -> float:
    """
    Determines complexity based on cognitive complexity metrics.

    Args:
        input_query (str): The problem to solve.
        complexity_threshold (float): Complexity score to consider complex.

    Returns:
        float: Score between 0 and 1 indicating complexity.
    """
    # Define rules: each occurrence adds to complexity
    rules = {
        "if": 1,
        "while": 1,
        "for": 1,
        "else": 1,
        "elif": 1,
        "switch": 1,
        "case": 1,
        "when": 1,
        "and": 0.5,
        "or": 0.5,
        "but": 0.5,
        "first": 0.5,
        "next": 0.5,
        "then": 0.5,
        "finally": 0.5,
    }

    # Normalize input
    input_lower = input_query.lower()

    complexity_score = 0
    for keyword, score in rules.items():
        matches = re.findall(r"\b" + re.escape(keyword) + r"\b", input_lower)
        count = len(matches)
        complexity_score += count * score
        print(
            f"[Cognitive Metrics] Keyword '{keyword}' occurrences: {count}, adding {count * score} points."
        )
        with open(save_path, "a") as f:
            f.write(
                f"[Cognitive Metrics] Keyword '{keyword}' occurrences: {count}, adding {count * score} points."
            )

    # Normalize score (assuming max possible score is, say, 15)
    normalized_score = min(complexity_score / complexity_threshold, 1.0)
    print(
        f"[Cognitive Metrics] Total Complexity Score: {complexity_score}, Normalized Score: {normalized_score}"
    )
    with open(save_path, "a") as f:
        f.write(
            f"[Cognitive Metrics] Total Complexity Score: {complexity_score}, Normalized Score: {normalized_score}"
        )
    return normalized_score


# 10. Abstract Syntax Tree (AST) Generation
def is_complex_ast(
    input_query: str, substep_threshold: int = 8, depth_threshold: int = 2
) -> float:
    """
    Determines complexity based on AST-like structure of the query.

    Args:
        input_query (str): The problem to solve.
        substep_threshold (int): Number of nodes to consider complex.
        depth_threshold (int): Depth of AST to consider complex.

    Returns:
        float: Score between 0 and 1 indicating complexity.
    """
    doc = nlp_spacy(input_query)

    G = nx.DiGraph()

    # Build AST-like graph based on dependencies
    for token in doc:
        for child in token.children:
            G.add_edge(token.text, child.text)

    substeps = G.number_of_nodes()

    try:
        longest_path = nx.dag_longest_path(G)
        max_depth = len(longest_path) - 1 if len(longest_path) > 1 else 0
    except nx.NetworkXUnfeasible:
        max_depth = depth_threshold + 1

    # Normalize scores
    substeps_score = 1 / (1 + math.exp(-substeps / substep_threshold))
    depth_score = 1 / (1 + math.exp(-max_depth / depth_threshold))

    # Combined score that considers both substeps and depth and normalizes them to a float between 0 and 1
    score = substeps_score * depth_score
    # score = 1 / (1 + math.exp(-score)) / (substep_threshold + depth_threshold)
    print(
        f"[AST Generation] Substeps: {substeps}, Max Depth: {max_depth}, Score: {score}"
    )
    with open(save_path, "a") as f:
        f.write(
            f"[AST Generation] Substeps: {substeps}, Max Depth: {max_depth}, Score: {score}"
        )
    return score


# 11. Statistical Analysis of Historical Data
def is_complex_statistical(input_query: str, threshold: float = 0.5) -> float:
    """
    Determines complexity based on similarity to historical complex queries.

    Args:
        input_query (str): The problem to solve.
        threshold (float): Similarity threshold to consider complex.

    Returns:
        float: Weighted complexity score between 0 and 1.
    """
    try:
        # import cosine_similarity from sklearn.metrics.pairwise
        from sklearn.metrics.pairwise import cosine_similarity
    except ImportError:
        print(
            "scikit-learn is not installed. Please install scikit-learn to use this function."
        )
        return 0.0
    input_vec = vectorizer_stat.transform([input_query])
    print(f"input_vec dimensions: {len(input_vec.shape)}")
    print(f"tfidf_stat dimensions: {len(tfidf_stat.shape)}")
    similarities = cosine_similarity(input_vec, tfidf_stat).flatten()

    # Get top 3 similar queries
    top_indices = np.argsort(similarities)[::-1][:3]
    top_similarities = similarities[top_indices]
    top_labels = df_stat["complex"].iloc[top_indices]

    # Weighted average complexity
    if sum(top_similarities) == 0:
        weighted_complexity = 0
    else:
        weighted_complexity = sum(
            sim * lbl for sim, lbl in zip(top_similarities, top_labels)
        ) / sum(top_similarities)
    # weighted_complexity = 1 / (
    #     1 + math.exp(-weighted_complexity)
    # )  # Normalize to [0, 1]
    print(f"[Statistical Analysis] Weighted Complexity Score: {weighted_complexity}")
    with open(save_path, "a") as f:
        f.write(
            f"[Statistical Analysis] Weighted Complexity Score: {weighted_complexity}"
        )
    return weighted_complexity


# 13. Interactive Query Expansion
def generate_follow_up_questions(input_query: str, num_questions: int = 3) -> list:
    """
    Generates follow-up questions to expand the input query.

    Args:
        input_query (str): The problem to solve.
        num_questions (int): Number of follow-up questions to generate.

    Returns:
        list: List of follow-up questions.
    """
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        print("[Interactive Query Expansion] OpenAI API key not found.")
        return []
    openai.api_key = openai_api_key

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are an assistant that generates follow-up questions to clarify problem statements.",
                },
                {
                    "role": "user",
                    "content": f"Generate {num_questions} follow-up questions to clarify the following problem statement:\n\n{input_query}",
                },
            ],
            max_completion_tokens=150,
            n=1,
            stop=None,
            temperature=0.7,
        )
        questions = response.choices[0].message.content
        # Split into individual questions
        question_list = re.findall(r"\d+\.\s+(.*)", questions)
        print(f"[Interactive Query Expansion] Generated Questions: {question_list}")
        with open(save_path, "a") as f:
            f.write(
                f"[Interactive Query Expansion] Generated Questions: {question_list}"
            )
        return question_list
    except Exception as e:
        print(
            f"[Interactive Query Expansion] Error generating follow-up questions: {e}"
        )
        with open(save_path, "a") as f:
            f.write(
                f"[Interactive Query Expansion] Error generating follow-up questions: {e}"
            )
        return []


def is_complex_query_expansion(
    input_query: str, substep_threshold: int = 5, depth_threshold: int = 2
) -> float:
    """
    Determines complexity using interactive query expansion.

    Args:
        input_query (str): The problem to solve.
        substep_threshold (int): Number of follow-up questions to consider complex.
        depth_threshold (int): Depth of follow-up questions to consider complex.

    Returns:
        float: Score between 0 and 1 indicating complexity.
    """
    follow_up_questions = generate_follow_up_questions(input_query)

    # Simple heuristic: Number of follow-up questions indicates complexity
    substeps = len(follow_up_questions)
    max_depth = 1  # As we don't have nested follow-ups in this simple approach

    # Normalize scores using thresholds and sigmoid function
    substeps_score = 1 / (1 + math.exp(-substeps / substep_threshold))
    depth_score = 1 / (1 + math.exp(-max_depth / depth_threshold))

    # Combined score
    score = substeps_score * depth_score
    print(
        f"[Interactive Query Expansion] Substeps (Follow-ups): {substeps}, Max Depth: {max_depth}, Score: {score}"
    )
    with open(save_path, "a") as f:
        f.write(
            f"[Interactive Query Expansion] Substeps (Follow-ups): {substeps}, Max Depth: {max_depth}, Score: {score}"
        )
    return score


# 14. Psycholinguistic Metrics
def flesch_kincaid_grade(text: str) -> float:
    """
    Calculates the Flesch-Kincaid Grade Level for the given text.

    Args:
        text (str): The text to analyze.

    Returns:
        float: Flesch-Kincaid Grade Level.
    """
    sentences = sent_tokenize(text)
    words = word_tokenize(text)
    syllables = 0
    d = cmudict.dict()

    for word in words:
        word_lower = word.lower()
        if word_lower in d:
            syllables += [
                len(list(y for y in x if y[-1].isdigit())) for x in d[word_lower]
            ][0]
        else:
            # Fallback: count vowels as syllables
            syllables += len(re.findall(r"[aeiouy]+", word_lower))

    if len(words) == 0 or len(sentences) == 0:
        return 0.0

    # Flesch-Kincaid formula
    fk_grade = (
        0.39 * (len(words) / len(sentences)) + 11.8 * (syllables / len(words)) - 15.59
    )
    return round(fk_grade, 2)


def is_complex_psycholinguistic(
    input_query: str, grade_threshold: float = 12.0
) -> float:
    """
    Determines complexity based on Flesch-Kincaid Grade Level.

    Args:
        input_query (str): The problem to solve.
        grade_threshold (float): Grade level to consider complex.

    Returns:
        float: Score between 0 and 1 indicating complexity.
    """
    grade = flesch_kincaid_grade(input_query)
    score = min((grade - grade_threshold) / grade_threshold, 1.0)
    # score = 1 / (1 + math.exp(-score))
    print(
        f"[Psycholinguistic Metrics] Flesch-Kincaid Grade Level: {grade}, Score: {score}"
    )
    with open(save_path, "a") as f:
        f.write(
            f"[Psycholinguistic Metrics] Flesch-Kincaid Grade Level: {grade}, Score: {score}"
        )
    return score


# 16. Emotional Sentiment Analysis
def is_complex_sentiment(
    input_query: str, keywords: list = None, sentiment_threshold: float = 0.0
) -> float:
    """
    Determines complexity based on emotional sentiment analysis.

    Args:
        input_query (str): The problem to solve.
        keywords (list): List of sentiment-indicative keywords.
        sentiment_threshold (float): Sentiment polarity threshold to consider complex.

    Returns:
        float: Score between 0 and 1 indicating complexity.
    """
    if keywords is None:
        keywords = [
            "difficult",
            "complex",
            "challenging",
            "complicated",
            "intricate",
            "elaborate",
            "hard",
        ]

    # Compute sentiment polarity
    blob = TextBlob(input_query)
    sentiment = blob.sentiment.polarity  # Range [-1.0, 1.0]
    print(f"[Sentiment Analysis] Sentiment Polarity: {sentiment}")
    with open(save_path, "a") as f:
        f.write(f"[Sentiment Analysis] Sentiment Polarity: {sentiment}")

    # Check for indicative keywords
    keyword_present = False
    for kw in keywords:
        if re.search(r"\b" + re.escape(kw) + r"\b", input_query.lower()):
            keyword_present = True
            print(f"[Sentiment Analysis] Keyword '{kw}' found.")
            with open(save_path, "a") as f:
                f.write(f"[Sentiment Analysis] Keyword '{kw}' found.")
            break

    # Determine complexity
    if sentiment < sentiment_threshold or keyword_present:
        score = 1.0
    else:
        score = 0.0
    print(f"[Sentiment Analysis] Complexity Score: {score}")
    with open(save_path, "a") as f:
        f.write(f"[Sentiment Analysis] Complexity Score: {score}")
    return score


# 17. Automated Theorem Proving Techniques
def is_complex_theorem_proving(input_query: str, step_threshold: int = 5) -> float:
    """
    Determines complexity based on automated theorem proving.

    Args:
        input_query (str): The problem to solve (mathematical).
        step_threshold (int): Number of steps to consider complex.

    Returns:
        float: Score between 0 and 1 indicating complexity.
    """
    # Simple parser to identify the type of problem
    # This is highly simplistic and may not work for complex natural language inputs
    # Ideally, NLP techniques would be used to parse the problem type

    # Example: "Solve the integral of x^2 dx."
    try:
        if "integral" in input_query.lower():
            # Extract the integrand
            match = re.search(r"integral of (.+?) dx", input_query.lower())
            if match:
                integrand = match.group(1)
                expr = sympy.sympify(integrand)
                integral = sympy.integrate(expr, sympy.Symbol("x"))
                # SymPy does not provide step counts; this is illustrative
                steps = (
                    len(sympy.integrate(expr, sympy.Symbol("x"), manual=True))
                    if hasattr(
                        sympy.integrate(expr, sympy.Symbol("x"), manual=True), "__len__"
                    )
                    else 1
                )
                print(f"[Theorem Proving] Integral result: {integral}, Steps: {steps}")
                with open(save_path, "a") as f:
                    f.write(
                        f"[Theorem Proving] Integral result: {integral}, Steps: {steps}"
                    )
                score = min(steps / step_threshold, 1.0)
                return score
    except Exception as e:
        print(f"[Theorem Proving] Error: {e}")
        with open(save_path, "a") as f:
            f.write(f"[Theorem Proving] Error: {e}")
        return 0.0

    # If the problem type is not recognized, return 0
    print("[Theorem Proving] Problem type not recognized. Score: 0.0")
    with open(save_path, "a") as f:
        f.write("[Theorem Proving] Problem type not recognized. Score: 0.0")
    return 0.0


# 18. Computational Linguistics Complexity Measures
def calculate_entropy(text: str) -> float:
    """
    Calculates the Shannon entropy of the text.

    Args:
        text (str): The text to analyze.

    Returns:
        float: Entropy value.
    """
    tokens = word_tokenize(text.lower())
    total_tokens = len(tokens)
    counts = Counter(tokens)
    probabilities = [count / total_tokens for count in counts.values()]
    entropy = -sum(p * math.log2(p) for p in probabilities if p > 0)
    return entropy


def is_complex_entropy(input_query: str, entropy_threshold: float = 5.0) -> float:
    """
    Determines complexity based on entropy.

    Args:
        input_query (str): The problem to solve.
        entropy_threshold (float): Entropy value to consider complex.

    Returns:
        float: Score between 0 and 1 indicating complexity.
    """
    entropy = calculate_entropy(input_query)
    score = min(entropy / entropy_threshold, 1.0)
    print(f"[Entropy Measure] Entropy: {entropy}, Score: {score}")
    with open(save_path, "a") as f:
        f.write(f"[Entropy Measure] Entropy: {entropy}, Score: {score}")
    return score


# 19. Temporal Sequence Analysis
def is_complex_temporal(
    input_query: str, temporal_keywords: list = None, sequence_threshold: int = 3
) -> float:
    """
    Determines complexity based on temporal sequence analysis.

    Args:
        input_query (str): The problem to solve.
        temporal_keywords (list): List of temporal connectors.
        sequence_threshold (int): Number of sequences to consider complex.

    Returns:
        float: Score between 0 and 1 indicating complexity.
    """
    if temporal_keywords is None:
        temporal_keywords = [
            "before",
            "after",
            "simultaneously",
            "then",
            "first",
            "next",
            "finally",
            "subsequently",
        ]

    input_lower = input_query.lower()
    sequences = 0
    for kw in temporal_keywords:
        matches = re.findall(r"\b" + re.escape(kw) + r"\b", input_lower)
        sequences += len(matches)
        print(
            f"[Temporal Analysis] Temporal keyword '{kw}' occurrences: {len(matches)}"
        )
        with open(save_path, "a") as f:
            f.write(
                f"[Temporal Analysis] Temporal keyword '{kw}' occurrences: {len(matches)}"
            )

    # Normalize score
    score = min(sequences / sequence_threshold, 1.0)
    print(
        f"[Temporal Analysis] Total Temporal Sequences Detected: {sequences}, Score: {score}"
    )
    with open(save_path, "a") as f:
        f.write(
            f"[Temporal Analysis] Total Temporal Sequences Detected: {sequences}, Score: {score}"
        )
    return score


# Function to combine all methods


def is_complex_final(
    input_query: str, output_full_score: bool = False
) -> Tuple[bool, Plan]:
    """
    Determines complexity using a comprehensive hybrid approach.

    Args:
        input_query (str): The problem to solve.

    Returns:
        bool: True if complex, False otherwise.
    """
    # Method 1: NLP Dependency Parsing
    score_nlp = is_complex_nlp_dependency(input_query)
    with open(save_path, "a") as f:
        f.write(f"\n[NLP Dependency Parsing] Score: {score_nlp}\n\n")

    # Method 2: Semantic Role Labeling
    # score_srl = is_complex_srl(input_query)
    score_srl = is_complex_spacy_srl(input_query)
    with open(save_path, "a") as f:
        f.write(f"[Semantic Role Labeling] Score: {score_srl}\n\n")
    # Method 3: Machine Learning Classification
    score_ml = is_complex_ml(input_query)
    with open(save_path, "a") as f:
        f.write(f"[Machine Learning Classification] Score: {score_ml}\n\n")

    # Method 4: Language Model-Based Analysis
    score_llm, plan = is_complex_llm(input_query)
    with open(save_path, "a") as f:
        f.write(f"[Language Model] Score: {score_llm}\n\n")

    # Method 6: Graph-Based Approach
    score_graph = is_complex_graph(input_query)
    with open(save_path, "a") as f:
        f.write(f"[Graph-Based Approach] Score: {score_graph}\n\n")

    # Method 7: Recursive Task Decomposition
    score_recursive = is_complex_recursive(input_query)
    with open(save_path, "a") as f:
        f.write(f"[Recursive Task Decomposition] Score: {score_recursive}\n\n")

    # Method 8: Ontological Mapping
    score_ontology = is_complex_ontology(input_query)
    with open(save_path, "a") as f:
        f.write(f"[Ontological Mapping] Score: {score_ontology}\n\n")

    # Method 9: Cognitive Complexity Metrics
    score_cognitive = is_complex_cognitive(input_query)
    with open(save_path, "a") as f:
        f.write(f"[Cognitive Complexity Metrics] Score: {score_cognitive}\n\n")

    # Method 10: AST Generation
    score_ast = is_complex_ast(input_query)
    with open(save_path, "a") as f:
        f.write(f"[AST Generation] Score: {score_ast}\n\n")

    # Method 11: Statistical Analysis of Historical Data
    score_stat = is_complex_statistical(input_query)
    with open(save_path, "a") as f:
        f.write(f"[Statistical Analysis] Score: {score_stat}\n\n")

    # Method 13: Interactive Query Expansion
    score_query_expansion = is_complex_query_expansion(input_query)
    with open(save_path, "a") as f:
        f.write(f"[Interactive Query Expansion] Score: {score_query_expansion}\n\n")

    # Method 14: Psycholinguistic Metrics
    score_psycholinguistic = is_complex_psycholinguistic(input_query)
    with open(save_path, "a") as f:
        f.write(f"[Psycholinguistic Metrics] Score: {score_psycholinguistic}\n\n")

    # Method 16: Emotional Sentiment Analysis
    score_sentiment = is_complex_sentiment(input_query)
    with open(save_path, "a") as f:
        f.write(f"[Sentiment Analysis] Score: {score_sentiment}\n\n")

    # Method 17: Automated Theorem Proving Techniques
    score_theorem = is_complex_theorem_proving(input_query)
    with open(save_path, "a") as f:
        f.write(f"[Theorem Proving] Score: {score_theorem}\n\n")

    # Method 18: Computational Linguistics Complexity Measures
    score_entropy = is_complex_entropy(input_query)
    with open(save_path, "a") as f:
        f.write(f"[Entropy Measure] Score: {score_entropy}\n\n")

    # Method 19: Temporal Sequence Analysis
    score_temporal = is_complex_temporal(input_query)
    with open(save_path, "a") as f:
        f.write(f"[Temporal Analysis] Score: {score_temporal}\n\n")

    # average score_nlp, score_llm, and score_entropy
    avg_nlp_llm_ent = (score_nlp + score_llm + score_entropy) / 3
    print(f"\n[Average NLP, LLM, Entropy] Average Score: {avg_nlp_llm_ent}\n\n")
    with open(save_path, "a") as f:
        f.write(f"\n[Average NLP, LLM, Entropy] Average Score: {avg_nlp_llm_ent}\n\n")

    # Assign weights to each method
    weights = {
        "nlp": 0.1,
        "srl": 0.05,
        "ml": 0.10,
        "llm": 0.4,
        "graph": 0.1,
        "recursive": 0.00,
        "ontology": 0.00,
        "cognitive": 0.000,
        "ast": 0.1,
        "stat": 0.00,
        "query_expansion": 0.05,
        "psycholinguistic": 0.000,
        "sentiment": 0.00,
        "theorem": 0.000,
        "entropy": 0.10,
        "temporal": 0.000,
    }

    # Calculate total weighted score
    total_score = (
        score_nlp * weights["nlp"]
        + score_srl * weights["srl"]
        + score_ml * weights["ml"]
        + score_llm * weights["llm"]
        + score_graph * weights["graph"]
        + score_recursive * weights["recursive"]
        + score_ontology * weights["ontology"]
        + score_cognitive * weights["cognitive"]
        + score_ast * weights["ast"]
        + score_stat * weights["stat"]
        + score_query_expansion * weights["query_expansion"]
        + score_psycholinguistic * weights["psycholinguistic"]
        + score_sentiment * weights["sentiment"]
        + score_theorem * weights["theorem"]
        + score_entropy * weights["entropy"]
        + score_temporal * weights["temporal"]
    )

    print(f"[Final Assessment] Total Weighted Score: {total_score}")
    with open(save_path, "a") as f:
        f.write(f"[Final Assessment] Total Weighted Score: {total_score}")

    # Define a threshold for complexity
    complexity_threshold = 0.5
    return (
        (total_score > complexity_threshold, plan)
        if not output_full_score
        else (total_score, plan)
    )


# Example Usage and Testing
if __name__ == "__main__":

    example_queries = [
        "What is the capital of France?",
        "Explain the process of photosynthesis.",
        "List the steps to bake a cake, including preparing the ingredients, mixing, baking, and decorating.",
        "Solve the integral of x^2 * sin(x) dx.",
        "Describe how to set up a machine learning pipeline involving data collection, preprocessing, model training, and evaluation.",
        "Provide a comprehensive guide to setting up and deploying a full-stack web application, including frontend design, backend implementation, database configuration, and server deployment.",
    ]

    for idx, query in enumerate(example_queries, 1):
        print(f"\n---\nQuery {idx}: {query}\n---")
        complexity, plan = is_complex_final(query)
        print(f"Is the query complex? {'Yes' if complexity else 'No'}\n")
        with open(save_path, "a") as f:
            f.write(f"\n---\nQuery {idx}: {query}\n---")
            assert isinstance(plan, Plan)
            for step in plan.steps:
                assert isinstance(step, PlanStep)
                f.write(f"{step.step_number}. {step.step_name}\n")
                f.write(f"    Description: {step.step_description}\n")
                f.write(f"    Full Text: {step.step_full_text}\n")
                f.write(f"    PlanStep Explanation: {step.step_explanation}\n")
                f.write(f"    Expected Output: {step.step_output}\n")
                f.write(f"    Subtasks: \n")
                for subtask in step.subtasks:
                    assert isinstance(subtask, Subtask)
                    f.write(
                        f"        {subtask.subtask_number}. {subtask.subtask_name}\n"
                    )
                    f.write(f"            Description: {subtask.subtask_description}\n")
                    f.write(f"            Full Text: {subtask.subtask_full_text}\n")
                    f.write(
                        f"            Subtask Explanation: {subtask.subtask_explanation}\n"
                    )
                    f.write(f"            Expected Output: {subtask.subtask_output}\n")
            f.write(f"Is the query complex? {'Yes' if complexity else 'No'}\n\n")
