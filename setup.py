import nltk
import spacy

def download_dependencies():
    # Download NLTK data
    nltk.download('wordnet')
    nltk.download('words')
    
    # Download spaCy model
    spacy.cli.download("en_core_web_sm")

if __name__ == "__main__":
    download_dependencies()