import os
import json
import traceback
import pandas as pd
import requests
import streamlit as st
import numpy as np

def create_local_mitre_data_cache():
    """
    Create a minimal local cache of MITRE ATT&CK data that can be used
    when the online sources are unavailable.
    """
    # Check if we already have the cache file
    cache_file = 'assets/mitre_cache.json'
    
    # Ensure the assets directory exists
    os.makedirs('assets', exist_ok=True)
    
    if os.path.exists(cache_file):
        return cache_file
    
    # Create a minimal version of MITRE ATT&CK data with common techniques
    minimal_attack_data = {
        "objects": [
            # Tactics
            {
                "type": "x-mitre-tactic",
                "name": "Initial Access",
                "external_references": [{"external_id": "TA0001"}]
            },
            # ... rest of the data is unchanged ...
        ]
    }
    
    # Save the minimal data to a local cache file
    with open(cache_file, 'w') as f:
        json.dump(minimal_attack_data, f)
    
    return cache_file

@st.cache_data
def load_mitre_data():
    """
    Load MITRE ATT&CK data from online sources or local cache.
    Returns:
        techniques: List of technique dictionaries
        tactic_mapping: Dictionary mapping tactic names to IDs
        tactics_list: List of tactic names
    """
    try:
        # Add headers and timeout to make the request more robust
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        # Try the primary URL first
        try:
            response = requests.get(
                "https://raw.githubusercontent.com/mitre-attack/attack-stix-data/master/enterprise-attack/enterprise-attack.json",
                headers=headers,
                timeout=30
            )
            response.raise_for_status()  # Raise an exception for 4XX/5XX responses
            attack_data = response.json()
        except Exception as primary_error:
            # If primary URL fails, try a fallback URL
            try:
                # Fallback to a different version/branch or a cached copy
                response = requests.get(
                    "https://raw.githubusercontent.com/mitre/cti/master/enterprise-attack/enterprise-attack.json",
                    headers=headers,
                    timeout=30
                )
                response.raise_for_status()
                attack_data = response.json()
            except Exception as fallback_error:
                # If both online sources fail, use the local cache
                st.warning(f"MITRE data sources unavailable. Using local cache...")
                
                # Create and load local cache
                cache_file = create_local_mitre_data_cache()
                with open(cache_file, 'r') as f:
                    attack_data = json.load(f)
                st.info("Using locally cached MITRE data - limited technique coverage available")
        
        # Process the MITRE ATT&CK data
        techniques = []
        tactic_mapping = {}
        tactics_list = []

        # Extract tactics
        for obj in attack_data['objects']:
            if obj.get('type') == 'x-mitre-tactic':
                tactic_id = obj.get('external_references', [{}])[0].get('external_id', 'N/A')
                tactic_name = obj.get('name', 'N/A')
                tactic_mapping[tactic_name] = tactic_id
                tactics_list.append(tactic_name)

        # Extract techniques (skip sub-techniques)
        for obj in attack_data['objects']:
            if obj.get('type') == 'attack-pattern':
                ext_refs = obj.get('external_references', [{}])
                if not ext_refs:
                    continue
                    
                tech_id = ext_refs[0].get('external_id', 'N/A')
                if '.' in tech_id:
                    continue  # Skip sub-techniques
                
                # Get tactics for this technique
                tactic_names = []
                for phase in obj.get('kill_chain_phases', []):
                    tactic_names.append(phase.get('phase_name', ''))
                
                techniques.append({
                    'id': tech_id,
                    'name': obj.get('name', 'N/A'),
                    'description': obj.get('description', ''),
                    'tactic': ', '.join(tactic_names),
                    'tactics_list': tactic_names,
                    'url': ext_refs[0].get('url', '')
                })
        
        return techniques, tactic_mapping, tactics_list
    
    except Exception as e:
        # More detailed error reporting
        error_details = traceback.format_exc()
        st.error(f"Error loading MITRE data: {e}")
        st.error(f"Detailed error: {error_details}")
        
        # Return empty data structures as fallback
        st.warning("Using empty MITRE data as fallback. Mapping will be limited.")
        return [], {}, []

@st.cache_data
def load_library_data_with_embeddings(st_model, claude_config=None, use_claude_for_library=False):
    """
    Load library data and compute embeddings
    
    Args:
        st_model: TensorFlow-based encoder model
        claude_config: Claude API configuration dict (optional)
        use_claude_for_library: Whether to use Claude API for library embeddings
        
    Returns:
        library_df: DataFrame containing library data
        embeddings: Array of embeddings for library descriptions
    """
    try:
        # Read library.csv file
        try:
            library_df = pd.read_csv("library.csv")
        except Exception as e:
            # Modify warning to be more user friendly
            st.warning(f"No library.csv file found. You can create one to enable library matching and suggestions.")
            # Create an empty DataFrame with required columns
            library_df = pd.DataFrame(columns=['Use Case Name', 'Description', 'Log Source', 
                                              'Mapped MITRE Tactic(s)', 'Mapped MITRE Technique(s)', 
                                              'Reference Resource(s)', 'Search'])
        
        if library_df.empty:
            return None, None
        
        # Fill NaN values with placeholders
        for col in library_df.columns:
            if library_df[col].dtype == 'object':
                library_df[col] = library_df[col].fillna("N/A")
        
        # Check if embedding models are available
        if st_model is None and (claude_config is None or not claude_config.get("api_key")):
            st.warning("No embedding models available. Library matching may be unavailable.")
            return library_df, None
        
        # Import the batch_get_embeddings_hybrid function for embeddings
        from modules.embedding import batch_get_embeddings_hybrid
        
        # Precompute embeddings for all library entries
        descriptions = []
        for desc in library_df['Description'].tolist():
            if pd.isna(desc) or isinstance(desc, float):
                descriptions.append("No description available")  # Safe fallback
            else:
                descriptions.append(str(desc))  # Ensure it's a string
        
        # Use the hybrid approach - normally use transformer for library 
        # to save on Claude API credits
        embeddings = batch_get_embeddings_hybrid(
            descriptions, 
            st_model, 
            claude_config, 
            use_claude_api=use_claude_for_library
        )
        
        if embeddings is not None:
            return library_df, embeddings
        else:
            st.warning("Failed to generate embeddings for library data. Library matching may not work correctly.")
            return library_df, None
        
    except Exception as e:
        st.warning(f"Warning: Could not process library data: {e}")
        return None, None
