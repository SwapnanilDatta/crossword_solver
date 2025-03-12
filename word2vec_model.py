from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("all-MiniLM-L6-v2")

def get_sentence_similarity(word, clue):
    """Calculate similarity score between a word and clue using Sentence Transformers."""
    word_embedding = model.encode(word, convert_to_tensor=True)
    clue_embedding = model.encode(clue, convert_to_tensor=True)
    return util.pytorch_cos_sim(word_embedding, clue_embedding).item()

def get_similar_words(word, n=10):
    """Get similar words using sentence transformers."""
    # Return empty list if word is empty
    if not word:
        return []
    try:
        # Since we're using sentence transformers now, we'll return an empty list
        # as the similar words will be handled by the similarity scoring
        return []
    except Exception as e:
        print(f"Error getting similar words for {word}: {e}")
        return []

# Replace the old get_similarity_score function
get_similarity_score = get_sentence_similarity