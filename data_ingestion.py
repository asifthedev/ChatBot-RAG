import os
import pandas as pd
from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI
from typing import List

# Initialize API clients using environment variables for security
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
pinecone_client = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Constants for configuration
EMBEDDING_MODEL = "text-embedding-3-small"
INDEX_NAME = "ukmegashop-faq"
EMBEDDING_DIMENSION = 1536  # Dimension for text-embedding-3-small
METRIC = "cosine"
CLOUD_PROVIDER = "aws"
REGION = "us-east-1"

def get_embeddings(text: str, model: str = EMBEDDING_MODEL) -> List[float]:
    """
    Generate embeddings for a given text using OpenAI's embedding model.

    Args:
        text (str): The input text to embed.
        model (str): The OpenAI embedding model to use (default: text-embedding-3-small).

    Returns:
        List[float]: The embedding vector as a list of floats.
    """
    response = openai_client.embeddings.create(input=[text], model=model)
    return response.data[0].embedding

def create_pinecone_index() -> None:
    """Create a Pinecone index if it doesn't already exist."""
    if INDEX_NAME not in pinecone_client.list_indexes().names():
        pinecone_client.create_index(
            name=INDEX_NAME,
            dimension=EMBEDDING_DIMENSION,
            metric=METRIC,
            spec=ServerlessSpec(cloud=CLOUD_PROVIDER, region=REGION)
        )
        print(f"Created Pinecone index: {INDEX_NAME}")
    else:
        print(f"Index {INDEX_NAME} already exists.")

def load_and_process_data(file_path: str) -> pd.DataFrame:
    """
    Load FAQ data from a CSV file and return a DataFrame.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: DataFrame containing the FAQ data.
    """
    try:
        df = pd.read_csv(file_path)
        print("Data preview:")
        print(df.head())
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"CSV file not found at: {file_path}")
    except Exception as e:
        raise Exception(f"Error loading CSV file: {str(e)}")

def prepare_vectors(df: pd.DataFrame) -> List[dict]:
    """
    Prepare vectors with embeddings and metadata for upserting into Pinecone.

    Args:
        df (pd.DataFrame): DataFrame with 'Question' and 'Answer' columns.

    Returns:
        List[dict]: List of vectors with IDs, embeddings, and metadata.
    """
    vectors = []
    for idx, row in df.iterrows():
        question = row["Question"]
        answer = row["Answer"]
        embedding = get_embeddings(text=question)
        vector_id = f"qa_{idx}"

        vector = {
            "id": vector_id,
            "values": embedding,
            "metadata": {"question": question, "answer": answer}
        }
        vectors.append(vector)
    return vectors

def main():
    """Main function to orchestrate the process."""
    # Define file path
    file_path = "./dataset/faq.csv"

    # Load data
    df = load_and_process_data(file_path)

    # Create Pinecone index if it doesn't exist
    create_pinecone_index()

    # Prepare vectors for upsert
    vectors = prepare_vectors(df)

    # Connect to the Pinecone index and upsert vectors
    pinecone_index = pinecone_client.Index(INDEX_NAME)
    pinecone_index.upsert(vectors=vectors)
    print(f"Successfully upserted {len(vectors)} vectors into {INDEX_NAME}.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"An error occurred: {str(e)}")