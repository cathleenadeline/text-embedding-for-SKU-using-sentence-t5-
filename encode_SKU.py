import json
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('sentence-transformers/sentence-t5-base')

def encode_sentences(sentences):
    embeddings = model.encode(sentences)
    return embeddings

def write_index(sentences, embeddings, path="/tmp/index.json"):
    index = {}
    for i, (s, e) in enumerate(zip(sentences, embeddings)):
        index[i] = {
            "sentence": s,
            "embedding": e.tolist()
        }
    with open(path, 'w') as f:
        json.dump(index, f)

# Read sentences from a text file
text_file_path = '/tmp/sku.txt'
with open(text_file_path, 'r') as file:
    sentences = file.readlines()

# Encode sentences
embeddings = encode_sentences(sentences)

# Write index to JSON file
write_index(sentences, embeddings, "/tmp/myindex.json")
