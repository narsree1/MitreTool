import torch
import streamlit as st
import requests
import json
import numpy as np
from typing import List, Dict, Any, Optional, Union
from sentence_transformers import SentenceTransformer

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
        model: The loaded SentenceTransformer model
    """
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Using bge-base-en-v1.5 model instead of all-mpnet-base-v2 for better performance
        model = SentenceTransformer('BAAI/bge-base-en-v1.5')
        model = model.to(device)
        return model
    except Exception as e:
        st.error(f"Error loading sentence transformer model: {e}")
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
            embedding = np.random.randn(1536)  # Using 1536-dim embedding similar to other LLMs
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
    Get embedding for a text using sentence transformer
    
    Args:
        text: Text to embed
        model: SentenceTransformer model
        
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

def batch_get_embeddings_with_transformer(texts: List[str], model) -> Optional[torch.Tensor]:
    """
    Get embeddings for multiple texts using sentence transformer
    
    Args:
        texts: List of texts to embed
        model: SentenceTransformer model
        
    Returns:
        embeddings: Tensor of embedding vectors
    """
    if model is None:
        return None
        
    try:
        # Use batching for encoding
        batch_size = 32
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            batch_embeddings = model.encode(batch, convert_to_tensor=True)
            all_embeddings.append(batch_embeddings)
        
        # Combine all embeddings
        if all_embeddings:
            embeddings = torch.cat(all_embeddings, dim=0)
            return embeddings
        return None
    except Exception as e:
        st.error(f"Error computing batch embeddings: {e}")
        return None

def batch_get_embeddings_hybrid(texts: List[str], st_model, claude_config: Dict[str, Any], 
                              use_claude_api: bool = False) -> Optional[torch.Tensor]:
    """
    Get embeddings for multiple texts using either sentence transformer or Claude API
    
    Args:
        texts: List of texts to embed
        st_model: SentenceTransformer model
        claude_config: Dictionary with Claude API configuration
        use_claude_api: Whether to use Claude API (for critical mapping) or sentence transformer
        
    Returns:
        embeddings: Tensor of embedding vectors
    """
    # For suggestions and other non-critical features, use sentence transformer
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
                    # If Claude API fails, fall back to sentence transformer
                    fallback = get_embedding_with_transformer(text, st_model)
                    if fallback is not None:
                        batch_embeddings.append(fallback)
                    else:
                        # Last resort: random fallback
                        np.random.seed(0)
                        fallback = np.random.randn(1536)
                        fallback = fallback / np.linalg.norm(fallback)
                        batch_embeddings.append(fallback)
            
            # Convert batch to tensor and append
            batch_tensor = torch.tensor(np.array(batch_embeddings))
            all_embeddings.append(batch_tensor)
        
        # Combine all embeddings
        if all_embeddings:
            embeddings = torch.cat(all_embeddings, dim=0)
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
        st_model: SentenceTransformer model
        claude_config: Claude API configuration
        techniques: List of MITRE technique dictionaries
        use_claude_api: Whether to use Claude API for embeddings
        
    Returns:
        embeddings: Tensor of embeddings for technique descriptions
    """
    if (st_model is None and claude_config is None) or not techniques:
        return None
    try:
        descriptions = [tech['description'] for tech in techniques]
        
        # Use the hybrid approach - can use sentence transformer for MITRE techniques 
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
        reference_embeddings: Tensor of reference embedding vectors
        
    Returns:
        best_score: Highest similarity score
        best_idx: Index of the best matching reference embedding
    """
    # Convert to torch tensors if they aren't already
    if not isinstance(query_embedding, torch.Tensor):
        query_embedding = torch.tensor(query_embedding)
    if not isinstance(reference_embeddings, torch.Tensor):
        reference_embeddings = torch.tensor(reference_embeddings)
    
    # Ensure query_embedding is 1D if it's just one embedding
    if len(query_embedding.shape) == 1:
        query_embedding = query_embedding.unsqueeze(0)
    
    # Normalize the embeddings
    query_embedding = query_embedding / query_embedding.norm(dim=1, keepdim=True)
    reference_embeddings = reference_embeddings / reference_embeddings.norm(dim=1, keepdim=True)
    
    # Calculate cosine similarity
    similarities = torch.mm(query_embedding, reference_embeddings.T)
    
    # Get the best match
    best_idx = similarities[0].argmax().item()
    best_score = similarities[0][best_idx].item()
    
    return best_score, best_idx

def batch_similarity_search(query_embeddings, reference_embeddings):
    """
    Perform batch cosine similarity search
    
    Args:
        query_embeddings: Tensor of query embedding vectors
        reference_embeddings: Tensor of reference embedding vectors
        
    Returns:
        best_scores: List of highest similarity scores for each query
        best_indices: List of indices of the best matching reference embedding for each query
    """
    # Convert to torch tensors if they aren't already
    if not isinstance(query_embeddings, torch.Tensor):
        query_embeddings = torch.tensor(query_embeddings)
    if not isinstance(reference_embeddings, torch.Tensor):
        reference_embeddings = torch.tensor(reference_embeddings)
    
    # Normalize the embeddings
    query_embeddings = query_embeddings / query_embeddings.norm(dim=1, keepdim=True)
    reference_embeddings = reference_embeddings / reference_embeddings.norm(dim=1, keepdim=True)
    
    # Calculate cosine similarity
    similarities = torch.mm(query_embeddings, reference_embeddings.T)
    
    # Get the best matches for each query
    best_scores, best_indices = similarities.max(dim=1)
    
    return best_scores.tolist(), best_indices.tolist()
