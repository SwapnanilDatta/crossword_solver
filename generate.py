import nltk
from nltk.corpus import words

nltk.download('words')

with open("words.txt", "w") as f:
    for word in words.words():
        f.write(word + "\n")

print("Word list saved to words.txt!")
