import pandas as pd
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import os

# Load CSV dataset
csv_file = "chewy_scraper_sample.csv"
df = pd.read_csv(csv_file, encoding='utf-8')

# Handle missing values
df.fillna({
    'name': 'Unknown Product',
    'description': 'No description available.',
    'Price': 0.0,
    'category_1': 'Unknown',
    'category_2': '',
    'category_3': '',
    'availability': 'Unknown',
    'ingredients': 'No ingredients listed.',
    'nutrition_analysis': 'No nutrition info.',
    'average_rating': 0.0,
    'reviews_count': 0,
    'url': ''  # Handle missing url (product URL column)
}, inplace=True)

# Create knowledge base
knowledge_base = {
    "products": [],
    "store_info": {
        "name": "Chewy Pet Shop",
        "contact": "support@chewy.com",
        "hours": "24/7 Online"
    }
}

# Prepare documents for RAG
documents = []
doc_metadata = []
for idx, row in df.iterrows():
    product = {
        "id": row['uniq_id'],
        "name": row['name'],
        "category": f"{row['category_1']} > {row['category_2']} > {row['category_3']}".strip(' > '),
        "Price": float(row['Price'].replace('$', '') if isinstance(row['Price'], str) else row['Price']),
        "description": row['description'],
        "availability": row['availability'],
        "ingredients": row['ingredients'],
        "nutrition_analysis": row['nutrition_analysis'],
        "average_rating": float(row['average_rating']),
        "reviews_count": int(row['reviews_count']),
        "url": row['url']  # Add product URL from 'url' column
    }
    knowledge_base["products"].append(product)
    
    # Create document for retrieval, including product URL
    doc = (
        f"Product: {row['name']} (${product['Price']}): {row['description']} "
        f"(Category: {product['category']}, Availability: {row['availability']}, "
        f"Rating: {row['average_rating']} ({row['reviews_count']} reviews), "
        f"Product URL: {row['url']})"
    )
    documents.append(doc)
    doc_metadata.append({"type": "product", "id": row['uniq_id']})

# Save knowledge base
with open("knowledge_base.json", "w", encoding='utf-8') as f:
    json.dump(knowledge_base, f, ensure_ascii=False, indent=2)

# Initialize SentenceTransformer
embedder = SentenceTransformer('all-distilroberta-v1')

# Create FAISS index
embeddings = embedder.encode(documents)
dimension = embeddings.shape[1]
print(f"Embedding dimension: {dimension}")  # Debug log
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)
faiss.write_index(index, "chewy_index.faiss")

# Save metadata
with open("doc_metadata.json", "w", encoding='utf-8') as f:
    json.dump(doc_metadata, f, ensure_ascii=False, indent=2)

print("Preprocessing complete. Generated knowledge_base.json, chewy_index.faiss, and doc_metadata.json.")