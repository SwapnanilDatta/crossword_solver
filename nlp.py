import nltk
import spacy
from nltk.corpus import wordnet
from word2vec_model import get_similar_words
from rapidfuzz import process
import re

nltk.download("wordnet")
nlp = spacy.load("en_core_web_sm")

def extract_keywords(clue):
    """Extract important keywords from the clue using NLP."""
    doc = nlp(clue)
    keywords = [token.lemma_ for token in doc if token.pos_ in ["NOUN", "VERB", "ADJ"]]
    return keywords

def get_synonyms(word):
    """Fetch synonyms of a word using WordNet."""
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name().replace("_", " "))
    return list(synonyms)

def filter_matching_words(words, pattern):
    """Filter words to match the given pattern (e.g., h__t)."""
    regex = "^" + pattern.replace("_", ".") + "$"
    return [word for word in words if re.match(regex, word, re.IGNORECASE)]

def find_related_words(clue, word_list, pattern=""):
    """Find the closest matching words from the word list using NLP and Word2Vec."""
    keywords = extract_keywords(clue)
    candidate_words = set()

    for keyword in keywords:
        candidate_words.update(get_synonyms(keyword))
        candidate_words.update(get_similar_words(keyword))  # Get Word2Vec suggestions

    # Rank words using fuzzy matching
    # process.extract returns a list of tuples (word, score, <additional_values>)
    matched_words = process.extract(clue, word_list, limit=10)
    # Only take the word from each match tuple
    best_matches = [match[0] for match in matched_words]

    # Combine candidates from WordNet, Word2Vec, and Fuzzy Matching
    all_candidates = list(candidate_words) + best_matches

    # Filter only words matching the crossword pattern
    if pattern:
        all_candidates = filter_matching_words(all_candidates, pattern)

    return all_candidates