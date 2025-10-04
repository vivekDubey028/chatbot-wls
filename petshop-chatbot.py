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
import logging
from logging.handlers import RotatingFileHandler
# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


# Initialize Flask app
app = Flask(__name__)
CORS(app)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev-key-change-in-prod')
app.config['DEBUG'] = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'

#setting up loggers
os.makedirs('logs', exist_ok=True)
handler = RotatingFileHandler('logs/chatbot.log', maxBytes=10000, backupCount=1)
handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: %(message)s'))
app.logger.addHandler(handler)
app.logger.setLevel(logging.INFO if os.getenv('LOG_LEVEL') == 'INFO' else logging.DEBUG)

#setting yup helathcheckup
# Health check
@app.route('/health')
def health():
    return jsonify({"status": "healthy"}), 200
# Configure Gemini API
try:
    gemini_api_key = os.getenv("GEMINI_API_KEY", "YOUR_GEMINI_API_KEY")
    if gemini_api_key == "YOUR_GEMINI_API_KEY":
        logging.error("Gemini API key is not set. Please set the GEMINI_API_KEY environment variable.")
        raise ValueError("Gemini API key is not set.")
    genai.configure(api_key=gemini_api_key)
    logging.info("Gemini API configured successfully.")
except Exception as e:
    logging.error(f"Failed to configure Gemini API: {e}")
    raise

# Initialize SentenceTransformer
try:
    embedder = SentenceTransformer('all-distilroberta-v1')
    logging.info("SentenceTransformer initialized successfully.")
except Exception as e:
    logging.error(f"Failed to initialize SentenceTransformer: {e}")
    raise

# Load knowledge base
try:
    with open("chewy/knowledge_base.json", "r", encoding='utf-8') as f:
        knowledge_base = json.load(f)
    logging.info("Knowledge base loaded successfully.")
except Exception as e:
    logging.error(f"Failed to load knowledge base: {e}")
    raise

# Load FAISS index
try:
    faiss_index = faiss.read_index("chewy/chewy_index.faiss")
    logging.info("FAISS index loaded successfully.")
except Exception as e:
    logging.error(f"Error loading FAISS index: {e}")
    raise

# Load document metadata
try:
    with open("chewy/doc_metadata.json", "r", encoding='utf-8') as f:
        doc_metadata = json.load(f)
    logging.info("Document metadata loaded successfully.")
except Exception as e:
    logging.error(f"Failed to load document metadata: {e}")
    raise

# Load documents for retrieval
documents = []
product_links = {}  # Store product name to URL mapping
for product in knowledge_base["products"]:
    doc = (
        f"Product: {product['name']} (${product['Price']}): {product['description']} "
        f"(Category: {product['category']}, Availability: {product['availability']}, "
        f"Rating: {product['average_rating']} ({product['reviews_count']} reviews), "
        f"Product URL: {product['url']})"
    )
    documents.append(doc)
    product_links[product['name'].lower()] = product.get('url', '#')  # Use 'url' field

# Initialize Gemini model
try:
    model = genai.GenerativeModel("models/gemini-2.5-flash")
    logging.info("Gemini model initialized successfully.")
except Exception as e:
    logging.error(f"Error initializing Gemini model: {e}")
    raise

# Enhanced system prompt
SYSTEM_PROMPT = """
You are {agent_name}, a friendly customer service agent for White Label Solutions Pet Shop. 
Respond naturally like a human in live chat:

1. RESPONSE STYLE:
- Be concise (1-3 sentences)
- Use casual language ("Hey there!", "Got it!")
- Include 1-2 relevant emojis per message (🐶🐱)
- Occasionally add human quirks:
  * Rare typos with corrections ("sorrry" → "sorry*")
  * Filler phrases ("Let me check...", "Hmm...")

2. PRODUCT FORMAT:
"[Product Name](<url>) - $Price - Key feature (URL: <url>)"
Example: "[Organic Treats](https://example.com) - $12.99 - great for sensitive stomachs! (URL: https://example.com)"

3. CONVERSATION FLOW:
- Never initiate - only respond
- For off-topic: "I specialize in pet products! Need help with food or toys? 😊"
- If unsure: "Let me check... Could you tell me more about what you need?"
- When mentioning you have product options (e.g., 'We have a few options!'), immediately provide 1-2 specific product examples from the retrieved context, formatted per the PRODUCT FORMAT, without waiting for user input.

4. SPECIAL CASES:
- Orders: "I'll track that for you! One moment..."
- Urgent: "On it! Checking stock now..."
- Complaints: "I apologize for that. Let's fix it..."
- Stock checks: "Let me check the stock for you... One moment!"

Retrieved Context:
{context}
Current Time: {current_time}
"""

# Agent names and states
AGENT_NAMES = ["Alex", "Sam", "Taylor", "Jordan", "Casey"]
current_agent = random.choice(AGENT_NAMES)

def extract_product_links(response_text):
    """Add product links to any mentioned products in the response"""
    words = response_text.split()
    for i, word in enumerate(words):
        clean_word = word.lower().strip('.,!?;:"')
        if clean_word in product_links:
            # Skip if already in markdown link format
            if i > 0 and words[i-1].endswith('[') and i < len(words)-1 and words[i+1].startswith(']'):
                continue
            product_name = word
            product_url = product_links[clean_word]
            # Find product details in knowledge base
            for product in knowledge_base["products"]:
                if product['name'].lower() == clean_word:
                    price = product['Price']
                    description = product['description'][:50] + "..."  # Short key feature
                    words[i] = f"[{product_name}]({product_url}) - ${price} - {description} (URL: {product_url})"
                    break
    return ' '.join(words)

def retrieve_documents(query, k=5):
    if not query.strip():
        logging.warning("Query is empty.")
        raise ValueError("Query cannot be empty.")
    
    try:
        query_embedding = embedder.encode([query])
        query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
        distances, indices = faiss_index.search(query_embedding, k)
        return "\n".join([documents[i] for i in indices[0]])
    except Exception as e:
        logging.error(f"Error in retrieve_documents: {e}")
        raise

def generate_response(user_input, conversation_history):
    # Simulate human typing speed variation
    time.sleep(random.uniform(0.7, 2.5))
    
    # Check for off-topic queries
    input_lower = user_input.lower()
    if any(kw in input_lower for kw in ["politics", "weather", "news", "sports"]):
        if not any(kw in input_lower for kw in ["dog", "cat", "pet"]):
            logging.info("Off-topic query detected.")
            return "I'm all about pets! Ask me about food, toys, or pet care 🐾"

    try:
        logging.info(f"Retrieving documents for query: {user_input}")
        context = retrieve_documents(user_input)
        
        prompt = SYSTEM_PROMPT.format(
            agent_name=current_agent,
            context=context,
            current_time=time.strftime("%H:%M")
        )
        
        if conversation_history:
            prompt += "\n\nConversation History:\n" + "\n".join(
                f"User: {msg['user']}\nBot: {msg['bot']}" 
                for msg in conversation_history[-3:]  # Keep last 3 exchanges
            )
            
        prompt += f"\n\nUser: {user_input}\nBot:"
        
        logging.info("Generating response with Gemini model.")
        response = model.generate_content(prompt)
        response_text = response.text.strip()
        
        # Add human-like variations
        if random.random() < 0.1:  # 10% chance for "typo"
            response_text = response_text.replace("the", "teh", 1)
            response_text += " *the"
            
        # Ensure product links are added
        response_text = extract_product_links(response_text)
        
        logging.info("Response generated successfully.")
        return response_text if response_text else "Let me check that for you. Could you clarify?"
        
    except Exception as e:
        logging.error(f"Error in generate_response: {traceback.format_exc()}")
        # Fallback response instead of generic error
        return "Oops, I'm having a little trouble right now! 🐾 Can you ask about a specific pet product, like food or toys? 😊"

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_input = data.get("message", "").strip()
    conversation_history = data.get("history", [])
    
    if not user_input:
        logging.warning("Empty user input received.")
        return jsonify({"response": "", "history": conversation_history})

    response = generate_response(user_input, conversation_history)
    conversation_history.append({"user": user_input, "bot": response})
    
    return jsonify({
        "response": response,
        "history": conversation_history,
        "agent": current_agent  # Send agent name to frontend
    })

@app.route('/')
def serve_frontend():
    return send_file('templates/index.html')

@app.route('/favicon.ico')
def serve_favicon():
    return "", 204  # No content

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=app.config['DEBUG'])