import re
from collections import defaultdict

# Step 1: Define documents
documents = {
    "Document 1": "The quick brown fox jumped over the lazy dog.",
    "Document 2": "The lazy dog slept in the sun.",
    "Document 3": "The fox is quick and the dog is lazy.",
    "Document 4": "Information retrieval uses inverted index."
}

# Step 2: Create an empty inverted index
inverted_index = defaultdict(list)

# Step 3: Tokenize documents and build index
for doc_id, text in documents.items():

    words = re.findall(r'\b[a-z]+\b', text.lower())

    for position, word in enumerate(words, start=1):

        inverted_index[word].append((doc_id, position))

# Step 4: Display inverted index
print("INVERTED INDEX")
print("-" * 50)

for word, postings in sorted(inverted_index.items()):

    print(word, "->", postings)

# Step 5: Search documents
query = input("\nEnter search query: ").lower()

query_words = re.findall(r'\b[a-z]+\b', query)

# Retrieve documents containing all query terms
if query_words:

    matching_docs = set(
        doc_id for doc_id, position
        in inverted_index.get(query_words[0], [])
    )

    for word in query_words[1:]:

        current_docs = set(
            doc_id for doc_id, position
            in inverted_index.get(word, [])
        )

        matching_docs = matching_docs.intersection(current_docs)

    print("\nSearch Results:")

    if matching_docs:

        for doc_id in sorted(matching_docs):
            print(doc_id, ":", documents[doc_id])

    else:
        print("No matching documents found.")

else:
    print("Please enter a valid search query.")