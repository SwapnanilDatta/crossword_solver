from flask import Flask, request, render_template, jsonify
import re
from nlp import find_related_words
from trie import Trie
from word2vec_model import get_similarity_score

app = Flask(__name__)

# Load words from file
def load_words():
    try:
        with open("words.txt", encoding='utf-8') as f:
            return set(word.strip().lower() for word in f.readlines())
    except FileNotFoundError:
        print("Warning: words.txt not found")
        return set()

word_list = load_words()

# Initialize Trie Data Structure
trie = Trie()
for word in word_list:
    trie.insert(word)

def find_matching_words(pattern, clue=""):
    """Find words that match the pattern and rank by clue relevance."""
    if not pattern:
        return []
        
    matched_words = trie.search_pattern(pattern.lower())
    
    if not matched_words:
        return []

    if clue:
        # Combine pattern matching with semantic relevance
        related_words = find_related_words(clue, word_list, pattern)
        matched_set = set(matched_words)
        combined_matches = list(matched_set.union(set(related_words)))
        
        # Score and sort all matches
        scored_words = [(word, get_similarity_score(word, clue)) for word in combined_matches]
            
        scored_words.sort(key=lambda x: x[1], reverse=True)
        return [word for word, score in scored_words]
    
    return matched_words

@app.route("/", methods=["GET", "POST"])
def index():
    solutions = []
    pattern = ""
    clue = ""
    solutions_with_scores = []  # Initialize the variable

    if request.method == "POST":
        pattern = request.form.get("pattern", "").strip()
        clue = request.form.get("clue", "").strip()

        if pattern:
            # Get words and scores
            scored_words = [(word, get_similarity_score(word, clue)) 
                          for word in find_matching_words(pattern, clue)[:20]]
            # Sort by score
            scored_words.sort(key=lambda x: x[1], reverse=True)
            # Separate words and scores for template
            solutions = [word for word, _ in scored_words]
            solutions_with_scores = scored_words

    return render_template("index.html", solutions=solutions, pattern=pattern, clue=clue, solutions_with_scores=solutions_with_scores)

@app.route("/api/search", methods=["POST"])
def api_search():
    data = request.get_json()
    pattern = data.get("pattern", "").strip()
    clue = data.get("clue", "").strip()
    
    solutions = find_matching_words(pattern, clue) if pattern else []
    return jsonify({"solutions": solutions[:20]})

if __name__ == "__main__":
    app.run(debug=True)