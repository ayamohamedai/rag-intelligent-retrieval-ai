import os
from openai import OpenAI
from sentence_transformers import SentenceTransformer

# Load OpenAI Client if API key exists
api_key = os.environ.get("OPENAI_API_KEY")
client = OpenAI(api_key=api_key) if api_key else None

# Local embeddings fallback
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def get_embedding(text):
    if client:
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding
    else:
        return embedding_model.encode(text)

def chat_with_ai(prompt, history=[]):
    if client:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": "You are a helpful assistant."}] + history + [{"role": "user", "content": prompt}],
            temperature=0.7
        )
        return response.choices[0].message.content
    else:
        return f"[LOCAL MODE] No API Key set. Input received: {prompt[:50]}..."
