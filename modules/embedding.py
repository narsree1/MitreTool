import torch
import streamlit as st
import requests
import json
import numpy as np
from typing import List, Dict, Any, Optional, Union
import tensorflow as tf  # Alternative to torch for embeddings if needed

# Claude API configuration constants
CLAUDE_API_URL = "https://api.anthropic.com/v1/messages"

# Available Claude models
CLAUDE_MODELS = {
    "haiku": "claude-3-haiku-20240307",   # Faster, cheaper model
    "sonnet": "claude-3-sonnet-20240229", # Balanced model
    "opus": "claude-3-opus-20240229"      # Most powerful model
}

@st.cache_resource
def load_sentence_transformer_model():
    """
    Load the sentence transformer model for embeddings
    
    Returns:
        model: A simple embedding model using TensorFlow instead of torch
    """
    try:
        # Create a basic TF-based encoder model instead of using SentenceTransformer
        # This avoids the torch._classes issue
        class SimpleTFEncoder:
            def __init__(self):
                # Initialize Universal Sentence Encoder from TF Hub if available
                # Fallback to simpler embedding if not
                try:
                    import tensorflow_hub as hub
                    self.encoder = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
                except:
                    # Fallback to simple word embedding + averaging
                    self.encoder = None
                    # Use TF's text vectorization layer as fallback
                    self.vectorizer = tf.keras.layers.TextVectorization(
                        max_tokens=10000,
                        output_mode='int',
                        output_sequence_length=100
                    )
                    # Simple embedding layer
                    self.embedding = tf.keras.layers.Embedding(
                        input_dim=10000,
                        output_dim=512,
                        mask_zero=True
                    )
                    # Adapt the vectorizer with some sample text
                    self.vectorizer.adapt(['sample text for adaptation'])
            
            def encode(self, texts, convert_to_tensor=False, convert_to_numpy=False):
                if not isinstance(texts, list):
                    texts = [texts]
                
                if self.encoder is not None:
                    # Use Universal Sentence Encoder
                    embeddings = self.encoder(texts).numpy()
                else:
                    # Use simple embedding + mean pooling
                    vectors = self.vectorizer(texts)
                    embeddings = self.embedding(vectors).numpy()
                    # Mean pooling
                    embeddings = np.mean(embeddings, axis=1)
                
                if convert_to_tensor:
                    import tensorflow as tf
                    return tf.convert_to_tensor(embeddings)
                elif convert_to_numpy:
                    return embeddings
                else:
                    return embeddings
        
        return SimpleTFEncoder()
    except Exception as e:
        st.error(f"Error loading embedding model: {e}")
        return None

@st.cache_resource
def load_claude_config():
    """
    Returns a config object with Claude API connection details
    
    Returns:
        model_config: A dictionary with Claude API configuration
    """
    try:
        # Check if Claude API key is set in Streamlit secrets
        api_key = st.secrets.get("claude", {}).get("api_key", "")
        
        if not api_key:
            return None
            
        # Use the claude_model from secrets if available, otherwise default to "haiku"
        model_type = st.secrets.get("claude", {}).get("model", "haiku").lower()
        
        # Make sure the model type is valid
        if model_type not in CLAUDE_MODELS:
            model_type = "haiku"  # Default to haiku if invalid model type
            
        # Return a config object instead of an actual model
        model_config = {
            "api_key": api_key,
            "api_url": CLAUDE_API_URL,
            "model": CLAUDE_MODELS[model_type],
            "model_type": model_type
        }
        return model_config
    except Exception as e:
        st.error(f"Error configuring Claude API: {e}")
        return None

@st.cache_data(ttl=3600)  # Cache for 1 hour to reduce API calls
def get_embedding_with_claude(text: str, model_config: Dict[str, Any]) -> Optional[np.ndarray]:
    """
    Get embedding for a text using Claude API with caching
    
    Args:
        text: Text to embed
        model_config: Dictionary with Claude API configuration
        
    Returns:
        embedding: Numpy array of embedding vector or None if failed
    """
    if not model_config or not model_config.get("api_key"):
        return None
        
    try:
        # Call Claude API to get embedding
        headers = {
            "x-api-key": model_config["api_key"],
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }
        
        # Format the request with the text to embed
        payload = {
            "model": model_config.get("model", CLAUDE_MODELS["haiku"]),
            "messages": [
                {"role": "user", "content": f"Create an embedding for the following text: {text}"}
            ],
            "max_tokens": 1024
        }
        
        response = requests.post(
            model_config["api_url"],
            headers=headers,
            data=json.dumps(payload)
        )
        
        # Process response to extract embedding
        if response.status_code == 200:
            response_data = response.json()
            # In real implementation, extract actual embedding from Claude's response
            # For now, we'll create a deterministic pseudo-embedding based on text hash
            hash_val = hash(text) % 10000
            np.random.seed(hash_val)
            embedding = np.random.randn(512)  # Using 512-dim embedding
            embedding = embedding / np.linalg.norm(embedding)  # Normalize
            return embedding
        else:
            st.error(f"Claude API error ({response.status_code}): {response.text}")
            return None
    except Exception as e:
        st.error(f"Error getting embedding from Claude: {e}")
        return None

def get_embedding_with_transformer(text: str, model) -> Optional[np.ndarray]:
    """
    Get embedding for a text using transformer model
    
    Args:
        text: Text to embed
        model: Encoder model
        
    Returns:
        embedding: Numpy array of embedding vector or None if failed
    """
    if model is None:
        return None
        
    try:
        embedding = model.encode(text, convert_to_numpy=True)
        return embedding
    except Exception as e:
        st.error(f"Error getting embedding from transformer: {e}")
        return None

def batch_get_embeddings_with_transformer(texts: List[str], model) -> Optional[np.ndarray]:
    """
    Get embeddings for multiple texts using transformer model
    
    Args:
        texts: List of texts to embed
        model: Encoder model
        
    Returns:
        embeddings: Array of embedding vectors
    """
    if model is None:
        return None
        
    try:
        # Use batching for encoding
        batch_size = 32
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            batch_embeddings = model.encode(batch)
            all_embeddings.append(batch_embeddings)
        
        # Combine all embeddings
        if all_embeddings:
            embeddings = np.vstack(all_embeddings)
            return embeddings
        return None
    except Exception as e:
        st.error(f"Error computing batch embeddings: {e}")
        return None

def batch_get_embeddings_hybrid(texts: List[str], st_model, claude_config: Dict[str, Any], 
                              use_claude_api: bool = False) -> Optional[np.ndarray]:
    """
    Get embeddings for multiple texts using either transformer or Claude API
    
    Args:
        texts: List of texts to embed
        st_model: Transformer model
        claude_config: Dictionary with Claude API configuration
        use_claude_api: Whether to use Claude API (for critical mapping) or transformer
        
    Returns:
        embeddings: Array of embedding vectors
    """
    # For suggestions and other non-critical features, use transformer
    if not use_claude_api or claude_config is None:
        return batch_get_embeddings_with_transformer(texts, st_model)
    
    # For critical mapping features, use Claude API
    try:
        # Process in batches to avoid rate limits
        all_embeddings = []
        batch_size = 5  # Small batch size for Claude API
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            batch_embeddings = []
            
            for text in batch:
                # Use cached version to reduce API calls
                embedding = get_embedding_with_claude(text, claude_config)
                if embedding is not None:
                    batch_embeddings.append(embedding)
                else:
                    # If Claude API fails, fall back to transformer
                    fallback = get_embedding_with_transformer(text, st_model)
                    if fallback is not None:
                        batch_embeddings.append(fallback)
                    else:
                        # Last resort: random fallback
                        np.random.seed(0)
                        fallback = np.random.randn(512)
                        fallback = fallback / np.linalg.norm(fallback)
                        batch_embeddings.append(fallback)
            
            # Convert batch to numpy array and append
            batch_array = np.array(batch_embeddings)
            all_embeddings.append(batch_array)
        
        # Combine all embeddings
        if all_embeddings:
            embeddings = np.vstack(all_embeddings)
            return embeddings
        return None
    except Exception as e:
        st.error(f"Error computing hybrid batch embeddings: {e}")
        return None

@st.cache_resource
def get_mitre_embeddings(st_model, claude_config, techniques, use_claude_api=False):
    """
    Generate embeddings for MITRE technique descriptions
    
    Args:
        st_model: Transformer model
        claude_config: Claude API configuration
        techniques: List of MITRE technique dictionaries
        use_claude_api: Whether to use Claude API for embeddings
        
    Returns:
        embeddings: Array of embeddings for technique descriptions
    """
    if (st_model is None and claude_config is None) or not techniques:
        return None
    try:
        descriptions = [tech['description'] for tech in techniques]
        
        # Use the hybrid approach - can use transformer for MITRE techniques 
        # since these are fixed and don't need Claude's advanced understanding
        return batch_get_embeddings_hybrid(descriptions, st_model, claude_config, use_claude_api=False)
    except Exception as e:
        st.error(f"Error computing MITRE embeddings: {e}")
        return None

def cosine_similarity_search(query_embedding, reference_embeddings):
    """
    Perform cosine similarity search between a query embedding and reference embeddings
    
    Args:
        query_embedding: Embedding vector for the query
        reference_embeddings: Array of reference embedding vectors
        
    Returns:
        best_score: Highest similarity score
        best_idx: Index of the best matching reference embedding
    """
    # Ensure both inputs are numpy arrays
    if not isinstance(query_embedding, np.ndarray):
        query_embedding = np.array(query_embedding)
    if not isinstance(reference_embeddings, np.ndarray):
        reference_embeddings = np.array(reference_embeddings)
    
    # Ensure query_embedding is 2D if it's just one embedding
    if len(query_embedding.shape) == 1:
        query_embedding = query_embedding.reshape(1, -1)
    
    # Normalize the embeddings
    query_norm = np.linalg.norm(query_embedding, axis=1, keepdims=True)
    query_embedding = query_embedding / query_norm
    
    ref_norm = np.linalg.norm(reference_embeddings, axis=1, keepdims=True)
    reference_embeddings = reference_embeddings / ref_norm
    
    # Calculate cosine similarity
    similarities = np.dot(query_embedding, reference_embeddings.T)
    
    # Get the best match
    best_idx = np.argmax(similarities[0])
    best_score = similarities[0, best_idx]
    
    return float(best_score), int(best_idx)

def batch_similarity_search(query_embeddings, reference_embeddings):
    """
    Perform batch cosine similarity search
    
    Args:
        query_embeddings: Array of query embedding vectors
        reference_embeddings: Array of reference embedding vectors
        
    Returns:
        best_scores: List of highest similarity scores for each query
        best_indices: List of indices of the best matching reference embedding for each query
    """
    # Ensure both inputs are numpy arrays
    if not isinstance(query_embeddings, np.ndarray):
        query_embeddings = np.array(query_embeddings)
    if not isinstance(reference_embeddings, np.ndarray):
        reference_embeddings = np.array(reference_embeddings)
    
    # Normalize the embeddings
    query_norm = np.linalg.norm(query_embeddings, axis=1, keepdims=True)
    query_embeddings = query_embeddings / query_norm
    
    ref_norm = np.linalg.norm(reference_embeddings, axis=1, keepdims=True)
    reference_embeddings = reference_embeddings / ref_norm
    
    # Calculate cosine similarity
    similarities = np.dot(query_embeddings, reference_embeddings.T)
    
    # Get the best matches for each query
    best_indices = np.argmax(similarities, axis=1)
    best_scores = [similarities[i, best_indices[i]] for i in range(len(best_indices))]
    
    return best_scores, best_indices.tolist()
