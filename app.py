import os
from typing import List, Dict, Any
import streamlit as st
from pinecone import Pinecone
from google import genai
from openai import OpenAI

# Constants
PINECONE_INDEX_NAME = "ukmegashop-faq"
EMBEDDING_MODEL = "text-embedding-3-small"
GEMINI_MODEL = "gemini-2.0-flash"

# Load environment variables
def load_config():
    """Load configuration from environment variables."""
    return {
        "pinecone_api_key": os.getenv("PINECONE_API_KEY"),
        "google_api_key": os.getenv("GOOGLE_API_KEY"),
        "openai_api_key": os.getenv("OPENAI_API_KEY"),
    }

# Initialize clients
def initialize_clients(config: Dict[str, str]) -> tuple[Pinecone, genai.Client, OpenAI]:
    """Initialize Pinecone, Google Gemini, and OpenAI clients."""
    pinecone_client = Pinecone(api_key=config["pinecone_api_key"])
    google_client = genai.Client(api_key=config["google_api_key"])
    openai_client = OpenAI(api_key=config["openai_api_key"])
    return pinecone_client, google_client, openai_client

# Embedding generation
def get_embeddings(text: str, client: OpenAI, model: str = EMBEDDING_MODEL) -> List[float]:
    """Generate embeddings for the given text using OpenAI."""
    try:
        response = client.embeddings.create(input=[text], model=model)
        return response.data[0].embedding
    except Exception as e:
        st.error(f"Error generating embeddings: {e}")
        return []

# Query Pinecone index
def query_pinecone(index: Any, query_embedding: List[float], top_k: int = 2) -> Dict[str, Any]:
    """Query Pinecone index with the given embedding."""
    try:
        return index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
    except Exception as e:
        st.error(f"Error querying Pinecone: {e}")
        return {"matches": []}

# Generate response using Gemini
def generate_response(google_client: genai.Client, input_text: str, context: Dict[str, Any]) -> str:
    """Generate a response using the Google Gemini model."""
    prompt = (
        f"You are a customer support chatbot for UK Mega Shop. "
        f"User query: {input_text}, Retrieved Context: {context['matches']}. "
        f"Now answer the user query using the context."
    )
    try:
        response = google_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt
        )
        return response.text
    except Exception as e:
        st.error(f"Error generating response: {e}")
        return "Sorry, I couldn't process your request at this time."

# Main app logic
def main():
    """Main function to run the Streamlit app."""

    # Load configuration and initialize clients
    config = load_config()
    if not all(config.values()):
        st.error("Missing API keys. Please set them in your environment variables.")
        return

    pinecone_client, google_client, openai_client = initialize_clients(config)
    index = pinecone_client.Index(PINECONE_INDEX_NAME)

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.messages.append({"role": "human", "content": "Does the website offer Cash on Delivery (COD)"})
        st.session_state.messages.append({"role": "ai", "content": "Thanks for your query! Currently, we do not offer Cash on Delivery (COD) on our website. However, we are considering implementing it in the future."})

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"], unsafe_allow_html=True)

    # Handle user input
    if user_input := st.chat_input("How can I help you today?"):
        # Display user message
        with st.chat_message("human"):
            st.markdown(user_input)

        # Add to chat history
        st.session_state.messages.append({"role": "human", "content": user_input})

        # Create a placeholder for the loading animation and response
        with st.chat_message("ai"):
            loading_placeholder = st.empty()
            
            # Show "Fetching Context" animation
            loading_placeholder.markdown("""
                <div style='text-align: left'>
                    Fetching Context<span>.</span><span>.</span><span>.</span>
                </div>
                <style>
                    @keyframes blink {
                        0% { opacity: 0; }
                        50% { opacity: 1; }
                        100% { opacity: 0; }
                    }
                    span {
                        animation: blink 1s infinite;
                    }
                    span:nth-child(2) { animation-delay: 0.2s; }
                    span:nth-child(3) { animation-delay: 0.4s; }
                </style>
            """, unsafe_allow_html=True)

            # Generate embedding and query Pinecone
            query_embedding = get_embeddings(user_input, openai_client)
            if query_embedding:
                query_result = query_pinecone(index, query_embedding)

                # Update to "Synthesizing Context" animation
                loading_placeholder.markdown("""
                    <div style='text-align: left'>
                        Synthesizing Context<span>.</span><span>.</span><span>.</span>
                    </div>
                    <style>
                        @keyframes blink {
                            0% { opacity: 0; }
                            50% { opacity: 1; }
                            100% { opacity: 0; }
                        }
                        span {
                            animation: blink 1s infinite;
                        }
                        span:nth-child(2) { animation-delay: 0.2s; }
                        span:nth-child(3) { animation-delay: 0.4s; }
                    </style>
                """, unsafe_allow_html=True)

                # Generate AI response
                response = generate_response(google_client, user_input, query_result)
                
                # Replace loading animation with actual response
                loading_placeholder.markdown(response, unsafe_allow_html=True)

                # Add to chat history
                st.session_state.messages.append({"role": "ai", "content": response})

if __name__ == "__main__":
    main()
