from typing import List, Dict, Tuple, Any, Optional
import pandas as pd
from sentence_transformers import SentenceTransformer
import torch

from modules.embedding import cosine_similarity_search, batch_similarity_search

def batch_check_library_matches(descriptions: List[str], 
                              library_df: pd.DataFrame,
                              library_embeddings: torch.Tensor,
                              _model: SentenceTransformer,
                              batch_size: int = 16,  # Smaller batch size for BGE models
                              similarity_threshold: float = 0.85) -> List[Tuple]:  # BGE models tend to have higher similarity scores
    """
    Check for matches in the library in batches for better performance.
    
    Args:
        descriptions: List of use case descriptions to match
        library_df: DataFrame containing library data
        library_embeddings: Tensor of embeddings for library descriptions
        _model: SentenceTransformer model for creating embeddings
        batch_size: Number of descriptions to process at once
        similarity_threshold: Minimum similarity score to consider a match
        
    Returns:
        List of tuples: (matched_row, score, match_message)
    """
    if library_df is None or library_df.empty or library_embeddings is None:
        return [(None, 0.0, "No library data available") for _ in descriptions]
    
    results = []
    
    # First try exact matches (fast text comparison)
    exact_matches = {}
    for i, desc in enumerate(descriptions):
        # Handle NaN, None or float values
        if pd.isna(desc) or desc is None or isinstance(desc, float):
            exact_matches[i] = (None, 0.0, "Invalid description (None or numeric value)")
            continue
            
        # Convert to lowercase for case-insensitive matching
        try:
            lower_desc = str(desc).lower()
            
            # Check if there's an exact match in library
            matches = library_df[library_df['Description'].str.lower() == lower_desc]
            if not matches.empty:
                exact_matches[i] = (matches.iloc[0], 1.0, "Exact match found in library")
        except Exception as e:
            # Handle any errors in string operations
            exact_matches[i] = (None, 0.0, f"Error processing description: {str(e)}")
    
    # Process descriptions in batches for embeddings
    query_embeddings_list = []
    
    # Process only the descriptions that didn't have exact matches
    remaining_indices = [i for i in range(len(descriptions)) if i not in exact_matches]
    
    # Validate remaining descriptions for encoding
    valid_indices = []
    valid_descriptions = []
    
    for idx in remaining_indices:
        desc = descriptions[idx]
        # Skip None or non-string values
        if pd.isna(desc) or desc is None or isinstance(desc, float):
            results.append((idx, (None, 0.0, "Invalid description (None or numeric value)")))
        else:
            valid_indices.append(idx)
            valid_descriptions.append(str(desc))  # Convert to string to be safe
    
    # Skip if no valid descriptions remain
    if not valid_descriptions:
        return [exact_matches.get(i, (None, 0.0, "No match found in library")) for i in range(len(descriptions))]
    
    # Encode in batches
    for i in range(0, len(valid_descriptions), batch_size):
        batch = valid_descriptions[i:i+batch_size]
        try:
            # BGE models work best with normalize_embeddings=True
            batch_embeddings = _model.encode(batch, convert_to_tensor=True, normalize_embeddings=True)
            
            # Perform search for this batch using PyTorch
            for j, query_embedding in enumerate(batch_embeddings):
                best_score, best_idx = cosine_similarity_search(query_embedding, library_embeddings)
                
                orig_idx = valid_indices[i + j]
                
                if best_score >= similarity_threshold:
                    results.append((orig_idx, (library_df.iloc[best_idx], best_score, 
                                f"Similar match found in library (score: {best_score:.2f})")))
                else:
                    results.append((orig_idx, (None, 0.0, "No match found in library")))
        except Exception as e:
            # Handle encoding errors
            for j in range(len(batch)):
                if i+j < len(valid_indices):
                    orig_idx = valid_indices[i + j]
                    results.append((orig_idx, (None, 0.0, f"Error during embedding: {str(e)}")))
    
    # Combine exact matches and embedding-based matches
    all_results = []
    for i in range(len(descriptions)):
        if i in exact_matches:
            all_results.append(exact_matches[i])
        else:
            # Find the result for this index
            result_found = False
            for idx, result in results:
                if idx == i:
                    all_results.append(result)
                    result_found = True
                    break
            if not result_found:
                all_results.append((None, 0.0, "No match found in library"))
    
    return all_results

def batch_map_to_mitre(descriptions: List[str], 
                      _model: SentenceTransformer, 
                      mitre_techniques: List[Dict], 
                      mitre_embeddings: torch.Tensor, 
                      batch_size: int = 16) -> List[Tuple]:  # Smaller batch size for BGE models
    """
    Map a batch of descriptions to MITRE ATT&CK techniques for better performance
    
    Args:
        descriptions: List of use case descriptions to map
        _model: SentenceTransformer model for creating embeddings
        mitre_techniques: List of MITRE technique dictionaries
        mitre_embeddings: Tensor of embeddings for technique descriptions
        batch_size: Number of descriptions to process at once
        
    Returns:
        List of tuples: (tactic, technique, url, tactics_list, score)
    """
    if _model is None or mitre_embeddings is None:
        return [("N/A", "N/A", "N/A", [], 0.0) for _ in descriptions]
    
    results = []
    
    # Process in batches
    for i in range(0, len(descriptions), batch_size):
        batch = descriptions[i:i+batch_size]
        
        try:
            # Encode query batch with normalize_embeddings=True for BGE models
            query_embeddings = _model.encode(batch, convert_to_tensor=True, normalize_embeddings=True)
            
            # Get best matches using batch similarity search
            best_scores, best_indices = batch_similarity_search(query_embeddings, mitre_embeddings)
            
            # Process results
            for j, (score, idx) in enumerate(zip(best_scores, best_indices)):
                best_tech = mitre_techniques[idx]
                
                # FIX: Convert tactic to Title Case
                tactic_proper = ', '.join([t.title() for t in best_tech['tactics_list']])
                
                # FIX: Extract just the technique name instead of "ID - Name"
                tech_name = best_tech['name']
                
                results.append((
                    tactic_proper,  # Use properly capitalized tactic
                    tech_name,      # Use just the technique name
                    best_tech['url'], 
                    [t.title() for t in best_tech['tactics_list']],  # Capitalize tactics in list
                    score
                ))
                
        except Exception as e:
            # Handle errors
            print(f"Error mapping batch to MITRE: {e}")
            # Fill with error values for this batch
            for _ in range(len(batch)):
                results.append(("Error", "Error", "Error", [], 0.0))
    
    return results

def process_mappings(df, _model, mitre_techniques, mitre_embeddings, library_df, library_embeddings):
    """
    Main function to process mappings in an optimized way
    
    Args:
        df: DataFrame containing security use cases to map
        _model: SentenceTransformer model for creating embeddings
        mitre_techniques: List of MITRE technique dictionaries
        mitre_embeddings: Tensor of embeddings for technique descriptions
        library_df: DataFrame containing library data
        library_embeddings: Tensor of embeddings for library descriptions
        
    Returns:
        df: Updated DataFrame with mapping results
        techniques_count: Dictionary counting occurrences of each technique
    """
    # Adjusted similarity threshold for BGE models - they tend to have higher scores
    similarity_threshold = 0.85
    
    # Get all descriptions at once and validate them
    descriptions = []
    for desc in df['Description'].tolist():
        if pd.isna(desc) or desc is None or isinstance(desc, float):
            descriptions.append("No description available")
        else:
            descriptions.append(str(desc))  # Convert to string to ensure it's a string
    
    # First batch check library matches
    library_match_results = batch_check_library_matches(
        descriptions, library_df, library_embeddings, _model, similarity_threshold=similarity_threshold
    )
    
    # Prepare lists for rows that need model mapping
    model_map_indices = []
    model_map_descriptions = []
    
    # Process results and collect cases needing model mapping
    tactics = []
    techniques = []
    references = []
    all_tactics_lists = []
    confidence_scores = []
    match_sources = []
    match_scores = []
    techniques_count = {}
    
    # Make sure all lists have entries for each row in the dataframe
    for _ in range(len(df)):
        tactics.append("N/A")
        techniques.append("N/A")
        references.append("N/A")
        all_tactics_lists.append([])
        confidence_scores.append(0)
        match_sources.append("N/A")
        match_scores.append(0)
    
    for i, library_match in enumerate(library_match_results):
        matched_row, match_score, match_source = library_match
        
        if matched_row is not None:
            # Use library match
            
            # FIX: Ensure tactic names use Title Case
            tactic_str = matched_row.get('Mapped MITRE Tactic(s)', 'N/A')
            if tactic_str != 'N/A':
                # Split by comma, capitalize each tactic, then rejoin
                tactic_parts = [part.strip().title() for part in tactic_str.split(',')]
                tactic = ', '.join(tactic_parts)
            else:
                tactic = 'N/A'
                
            # FIX: Extract just the technique name if it's in "ID - Name" format
            technique_str = matched_row.get('Mapped MITRE Technique(s)', 'N/A')
            if technique_str != 'N/A' and ' - ' in technique_str:
                # Extract just the name part after the ID
                technique = technique_str.split(' - ', 1)[1]
            else:
                technique = technique_str
                
            reference = matched_row.get('Reference Resource(s)', 'N/A')
            tactics_list = tactic.split(', ') if tactic != 'N/A' else []
            confidence = match_score
            
            # Store results
            tactics[i] = tactic
            techniques[i] = technique
            references[i] = reference
            all_tactics_lists[i] = tactics_list
            confidence_scores[i] = round(confidence * 100, 2)
            match_sources[i] = match_source
            match_scores[i] = round(match_score * 100, 2)
            
            # Count techniques (using normalized technique name)
            techniques_count[technique] = techniques_count.get(technique, 0) + 1
        else:
            # Make sure we're not trying to map invalid descriptions
            if not (descriptions[i] == "No description available" or pd.isna(descriptions[i])):
                # Mark for model mapping
                model_map_indices.append(i)
                model_map_descriptions.append(descriptions[i])
            else:
                # Invalid description placeholders are already set by default
                match_sources[i] = "Invalid description"
    
    # Batch map remaining cases using model
    if model_map_descriptions:
        model_results = batch_map_to_mitre(
            model_map_descriptions, _model, mitre_techniques, mitre_embeddings
        )
        
        # Process model results and insert at the correct positions
        for i, idx in enumerate(model_map_indices):
            if i < len(model_results):
                tactic, technique, reference, tactics_list, confidence = model_results[i]
                
                # Insert at the correct position
                tactics[idx] = tactic
                techniques[idx] = technique
                references[idx] = reference
                all_tactics_lists[idx] = tactics_list
                confidence_scores[idx] = round(confidence * 100, 2)
                match_sources[idx] = "BGE-large-en-v1.5 Embedding Model"  # Update to indicate the high-quality model
                match_scores[idx] = 0  # No library match score
                
                # Count techniques
                techniques_count[technique] = techniques_count.get(technique, 0) + 1
 
    # Add results to dataframe
    df['Mapped MITRE Tactic(s)'] = tactics
    df['Mapped MITRE Technique(s)'] = techniques
    df['Reference Resource(s)'] = references
    df['Confidence Score (%)'] = confidence_scores
    df['Match Source'] = match_sources
    df['Library Match Score (%)'] = match_scores
    
    return df, techniques_count

def get_suggested_use_cases(uploaded_df, library_df):
    """
    Find use cases from the library that match log sources in the uploaded data
    but aren't already present in the uploaded data.
    
    Args:
        uploaded_df: DataFrame containing user's uploaded use cases
        library_df: DataFrame containing library data
        
    Returns:
        DataFrame with suggested use cases
    """
    if uploaded_df is None or library_df is None or library_df.empty:
        return pd.DataFrame()
    
    # Step 1: Extract unique log sources from uploaded data
    user_log_sources = set()
    if 'Log Source' in uploaded_df.columns:
        # Handle multi-value log sources (comma separated)
        for log_source in uploaded_df['Log Source'].fillna('').astype(str):
            if log_source and log_source != 'N/A':
                for source in log_source.split(','):
                    user_log_sources.add(source.strip())
    
    # Filter out empty or N/A sources
    user_log_sources = {src for src in user_log_sources if src and src != 'N/A'}
    
    if not user_log_sources:
        return pd.DataFrame()  # No valid log sources found
    
    # Step 2: Find matching use cases in the library based on log sources
    matching_use_cases = []
    
    # Get set of existing use case descriptions for deduplication
    existing_descriptions = set()
    if 'Description' in uploaded_df.columns:
        existing_descriptions = set(uploaded_df['Description'].fillna('').astype(str).str.lower())
    
    # For each library entry, check if its log source matches any user log source
    for _, lib_row in library_df.iterrows():
        lib_log_source = str(lib_row.get('Log Source', ''))
        lib_description = str(lib_row.get('Description', '')).lower()
        
        # Check if any user log source matches this library entry's log source
        if any(user_source.lower() in lib_log_source.lower() for user_source in user_log_sources):
            # Check if this use case is already in the user's data (by description)
            if lib_description not in existing_descriptions:
                matching_use_cases.append(lib_row)
    
    # If we have matches, convert to DataFrame
    if matching_use_cases:
        suggestions_df = pd.DataFrame(matching_use_cases)
        
        # Add a relevance score column based on exact log source match
        suggestions_df['Relevance'] = suggestions_df.apply(
            lambda row: sum(1 for src in user_log_sources 
                          if src.lower() in str(row.get('Log Source', '')).lower()),
            axis=1
        )
        
        # Sort by relevance (highest first)
        suggestions_df = suggestions_df.sort_values('Relevance', ascending=False)
        
        # Include only relevant columns and rename for clarity
        needed_columns = ['Use Case Name', 'Description', 'Log Source', 
                          'Mapped MITRE Tactic(s)', 'Mapped MITRE Technique(s)',
                          'Reference Resource(s)', 'Search', 'Relevance']
        
        # Filter columns that exist
        actual_columns = [col for col in needed_columns if col in suggestions_df.columns]
        return suggestions_df[actual_columns]
    
    return pd.DataFrame()  # No suggestions found
