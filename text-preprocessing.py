import nltk
import re

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download required NLTK dataset
nltk.download('stopwords')

# Input text
text = """
Information Retrieval is an important field of computer science.
Search engines retrieve relevant documents from large collections.
Processing documents improves searching and retrieving information.
"""

print("Original Text:\n", text)

# Step 1: Convert text to lowercase
text = text.lower()

# Step 2: Tokenization
words = re.findall(r'\b[a-z]+\b', text)

print("\nTokenized Words:")
print(words)

# Step 3: Remove stop words
stop_words = set(stopwords.words('english'))

filtered_words = [
    word for word in words
    if word not in stop_words
]

print("\nAfter Stop Word Removal:")
print(filtered_words)

# Step 4: Apply Porter Stemming
ps = PorterStemmer()

stemmed_words = [
    ps.stem(word)
    for word in filtered_words
]

print("\nAfter Stemming:")
print(stemmed_words)

# Step 5: Display final processed text
processed_text = " ".join(stemmed_words)

print("\nFinal Processed Text:")
print(processed_text)