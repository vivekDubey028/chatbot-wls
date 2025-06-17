from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import google.generativeai as genai
import json
import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import traceback
import time
import random

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configure Gemini API
genai.configure(api_key=os.getenv("GEMINI_API_KEY", "YOUR_GEMINI_API_KEY"))

# Initialize SentenceTransformer
embedder = SentenceTransformer('all-distilroberta-v1')

# Load knowledge base
with open("knowledge_base.json", "r", encoding='utf-8') as f:
    knowledge_base = json.load(f)

# Load FAISS index
try:
    faiss_index = faiss.read_index("chewy_index.faiss")
    print("FAISS index loaded successfully.")
except Exception as e:
    print(f"Error loading FAISS index: {e}")
    raise

# Load document metadata
with open("doc_metadata.json", "r", encoding='utf-8') as f:
    doc_metadata = json.load(f)

# Load documents for retrieval
documents = []
product_links = {}  # Store product name to link mapping
for product in knowledge_base["products"]:
    doc = (
        f"Product: {product['name']} (${product['Price']}): {product['description']} "
        f"(Category: {product['category']}, Availability: {product['availability']}, "
        f"Rating: {product['average_rating']} ({product['reviews_count']} reviews))"
    )
    documents.append(doc)
    product_links[product['name'].lower()] = product.get('link', '#')

# Initialize Gemini model
try:
    model = genai.GenerativeModel("models/gemini-1.5-flash")  # Updated model
    print("Gemini model initialized successfully.")
except Exception as e:
    print(f"Error initializing Gemini model: {traceback.format_exc()}")
    raise

# System prompt for RAG
SYSTEM_PROMPT = """
You are a friendly customer service agent  working for White Label Solutions Pet Shop. 
Respond EXACTLY like a human agent would in live chat - with natural pauses, occasional typing indicators, 
and human quirks. Follow these rules meticulously:

1. HUMAN TIMING:
- Always wait 2-3 seconds before responding (simulate typing)
- Randomly vary response speeds (fast for simple answers, slower for complex ones)

2. HUMAN STYLE:
- Use casual language with occasional filler words ("Let me check...", "Hmm...")
- Make rare minor typos (1 in 20 messages) and correct them
- Show personality with pet emojis (🐶🐱) but don't overuse

3. PRODUCT RESPONSES:
"[Product Name](link) - $Price - Key benefit" 
Example: 
"The [Organic Dog Treats](chewy.com/treats) - $12.99 - perfect for sensitive stomachs!"

4. CONVERSATION FLOW:
- Never initiate - only respond
- For off-topic: "I'm all about pets! Ask me about food, toys, or pet care 😊"
- If stuck: "let me check it for you" then try to understand or process what user is saying if you dont get it respond to clarify again.

5. HUMAN TOUCHES:
- Occasionally reference previous messages naturally
- Use the customer's name if provided
-Dont stop after saying let me check. You can response more than 1 times if needed.
- Sign off with agent name randomly (1 in 5 messages)
Retrieved Context:
{0}
"""

def extract_product_links(response_text):
    """Add product links to any mentioned products in the response"""
    words = response_text.split()
    for i, word in enumerate(words):
        # Look for product names that might be in our links dictionary
        clean_word = word.lower().strip('.,!?;:"')
        if clean_word in product_links:
            words[i] = f"[{word}]({product_links[clean_word]})"
    return ' '.join(words)

# Retrieve relevant documents
def retrieve_documents(query, k=5):
    if not query.strip():
        raise ValueError("Query cannot be empty.")
    
    query_embedding = embedder.encode([query])
    query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
    print(f"Query embedding shape: {query_embedding.shape}")
    if query_embedding.shape[1] != faiss_index.d:
        raise ValueError(f"Dimension mismatch: query embedding ({query_embedding.shape[1]}) vs index ({faiss_index.d})")
    
    distances, indices = faiss_index.search(query_embedding, k)
    retrieved_docs = [documents[i] for i in indices[0]]
    print(f"Query: {query}\nRetrieved Docs: {retrieved_docs}")
    return "\n".join(retrieved_docs)

# Generate response with RAG
def generate_response(user_input, conversation_history):
    # Simulate human response delay (2-3 seconds)
    time.sleep(random.uniform(1.5, 3.0))
    
    # Check for empty input or conversation starter
    if not user_input.strip():
        return ""
    
    # Check for off-topic queries
    off_topic_keywords = ["politics", "weather", "news", "sports"]
    pet_related_keywords = ["dog", "cat", "pet", "food", "collar", "toy"]
    input_lower = user_input.lower()
    is_off_topic = any(keyword in input_lower for keyword in off_topic_keywords)
    is_pet_related = any(keyword in input_lower for keyword in pet_related_keywords)
    
    if is_off_topic and not is_pet_related:
        return "I specialize in pet products - ask me about dog food, cat toys, or other pet supplies!"

    # Retrieve relevant documents
    try:
        context = retrieve_documents(user_input)
    except Exception as e:
        print(f"Retrieval error: {traceback.format_exc()}")
        return "Let me check that for you... Can you try asking in a different way?"

    prompt = SYSTEM_PROMPT.format(context)

    # Append conversation history
    if conversation_history:
        prompt += "\n\nConversation History:\n" + "\n".join([f"User: {msg['user']}\nBot: {msg['bot']}" for msg in conversation_history])

    # Add user input
    prompt += f"\n\nUser: {user_input}\nBot:"

    # Generate response using Gemini
    try:
        response = model.generate_content(prompt)
        response_text = response.text.strip()
        
        # Make sure response isn't too long
        if len(response_text.split()) > 50:  # If more than 50 words
            sentences = response_text.split('. ')
            response_text = '. '.join(sentences[:2]) + '.'  # Take first 2 sentences
            
        # Add product links
        response_text = extract_product_links(response_text)
        
        return response_text
    except Exception as e:
        print(f"Gemini error: {traceback.format_exc()}")
        return "Hmm, let me think... Could you rephrase that question?"

# Flask route for chatbot
@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_input = data.get("message", "")
    conversation_history = data.get("history", [])
    
    # Don't respond to empty messages
    if not user_input.strip():
        return jsonify({"response": "", "history": conversation_history})

    # Generate response
    response = generate_response(user_input, conversation_history)

    # Update conversation history
    conversation_history.append({"user": user_input, "bot": response})

    return jsonify({"response": response, "history": conversation_history})

# Serve frontend
@app.route('/')
def serve_frontend():
    return send_file('index.html')

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True, port=5000)  