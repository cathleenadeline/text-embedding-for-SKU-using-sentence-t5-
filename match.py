from sentence_transformers import SentenceTransformer
import json
import numpy as np

model = SentenceTransformer('sentence-transformers/sentence-t5-base')

def load_index(json_file_path):
    with open(json_file_path, 'r') as file:
        data = json.load(file)
    
    index = []
    for outer_index, (outer_key, inner_dict) in enumerate(data.items()):
        sentence = inner_dict.get("sentence")
        embedding = np.array(inner_dict.get("embedding"))
        index.append({"sentence": sentence, "embedding": embedding})
    
    return index

def encode(text):
    embedding = model.encode(text)
    return embedding

def distance(embedding1, embedding2):
    return np.linalg.norm(embedding1 - embedding2)

def match(query, index, top_k=5):
    results = []
    source_embedding = encode(query)
    
    for idx, record in enumerate(index):
        target_embedding = record["embedding"]
        dist = distance(source_embedding, target_embedding)
        results.append((idx, dist))
    
    results.sort(key=lambda x: x[1])
    
    # Return only the top k results
    return results[:top_k]

json_file_path = '/tmp/myindex.json' 
embeddings = load_index(json_file_path)

query = "Selada Keriting 500 Gram"
results = match(query, embeddings, top_k=5)
print(results)
