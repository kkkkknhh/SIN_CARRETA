"""
Example usage of the embedding model.
"""
import numpy as np
from embedding_model import create_embedding_model

def cosine_similarity(v1, v2):
    """Calculate cosine similarity between two vectors."""
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    return dot_product / (norm_v1 * norm_v2)

def main():
    """Demonstrate the embedding model with examples."""
    print("Initializing embedding model...")
    try:
        model = create_embedding_model()
        print(f"Model loaded successfully: {model.get_model_info()}")
        
        # Example texts in Spanish and English
        texts = [
            "La inteligencia artificial está cambiando el mundo",
            "Artificial intelligence is changing the world",
            "Los modelos de lenguaje son cada vez más avanzados",
            "Me gusta la programación y la ciencia de datos",
            "I enjoy programming and data science"
        ]
        
        print("\nCreating embeddings for example texts...")
        embeddings = model.embed(texts)
        
        print(f"Created {len(embeddings)} embeddings of dimension {embeddings.shape[1]}")
        
        # Compare similarity between Spanish and English similar sentences
        similarity = cosine_similarity(embeddings[0], embeddings[1])
        print(f"\nSimilarity between '{texts[0]}' and '{texts[1]}': {similarity:.4f}")
        
        # Compare unrelated sentences
        similarity = cosine_similarity(embeddings[0], embeddings[3])
        print(f"Similarity between '{texts[0]}' and '{texts[3]}': {similarity:.4f}")
        
        # Compare similarity between same language similar topics
        similarity = cosine_similarity(embeddings[3], embeddings[4])
        print(f"Similarity between '{texts[3]}' and '{texts[4]}': {similarity:.4f}")
        
    except Exception as e:
        print(f"Error demonstrating embedding model: {e}")

if __name__ == "__main__":
    main()
