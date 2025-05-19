# analytics.py - MITRE ATT&CK Analytics Module

import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

def normalize_tactic(tactic):
    """Normalize tactic names to a standard format"""
    if not tactic or tactic == 'N/A':
        return tactic
    
    # Convert to lowercase for comparison
    tactic_lower = tactic.lower()
    
    # Define canonical forms
    if "command" in tactic_lower and ("control" in tactic_lower or "and" in tactic_lower):
        return "Command and Control"
    if "persistence" in tactic_lower:
        return "Persistence"
    if "discovery" in tactic_lower:
        return "Discovery"
    if "execution" in tactic_lower:
        return "Execution"
    if "privilege" in tactic_lower and "escalation" in tactic_lower:
        return "Privilege Escalation"
    if "defense" in tactic_lower and "evasion" in tactic_lower:
        return "Defense Evasion"
    if "credential" in tactic_lower and "access" in tactic_lower:
        return "Credential Access"
    if "lateral" in tactic_lower and "movement" in tactic_lower:
        return "Lateral Movement"
    if "collection" in tactic_lower:
        return "Collection"
    if "exfiltration" in tactic_lower:
        return "Exfiltration"
    if "impact" in tactic_lower:
        return "Impact"
    if "initial" in tactic_lower and "access" in tactic_lower:
        return "Initial Access"
    
    # Default capitalize first letter
    return tactic[0].upper() + tactic[1:] if len(tactic) > 0 else tactic

def render_analytics_page(mitre_techniques):
    """
    Analytics page with techniques and tactics visualization
    """
    st.markdown("# 📈 Coverage Analytics")
    
    if st.session_state.mapping_complete and st.session_state.processed_data is not None:
        df = st.session_state.processed_data
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        # Create a techniques_count dictionary here instead of relying on session state
        # This will correctly process the data with our fixes
        processed_techniques_count = {}
        technique_name_mapping = {}
        
        # Create technique ID to name mapping from mitre_techniques list
        for tech in mitre_techniques:
            technique_name_mapping[tech['id']] = tech['name']
        
        # Process each row in the dataframe to properly handle comma-separated techniques
        for _, row in df.iterrows():
            technique_str = row.get('Mapped MITRE Technique(s)', '')
            if pd.isna(technique_str) or technique_str == 'N/A':
                continue
                
            # Handle comma-separated techniques by splitting
            techniques = [t.strip() for t in technique_str.split(',')]
            
            for technique in techniques:
                if not technique:
                    continue
                
                # Extract ID if in "T1234 - Name" format
                if ' - ' in technique and technique.startswith('T'):
                    tech_id = technique.split(' - ')[0].strip()
                else:
                    # For techniques without ID format, look up by name
                    tech_name = technique
                    tech_id = None
                    
                    # Try to find the ID by name
                    for tid, tname in technique_name_mapping.items():
                        if tname.lower() == tech_name.lower():
                            tech_id = tid
                            break
                    
                    # If still not found, use the name as ID
                    if not tech_id:
                        tech_id = tech_name
                
                # Count this technique
                processed_techniques_count[tech_id] = processed_techniques_count.get(tech_id, 0) + 1
                
        # Update session state for other pages
        st.session_state.techniques_count = processed_techniques_count
        
        total_techniques = 203  # Total number of MITRE techniques
        covered_techniques = len(processed_techniques_count.keys())
        coverage_percent = round((covered_techniques / total_techniques) * 100, 2)
        
        # Count library matches vs model matches - handle NaN values safely
        library_matches = df[df['Match Source'].fillna('Unknown').astype(str).str.contains('library', case=False, na=False)].shape[0]
        claude_matches = df[df['Match Source'].fillna('Unknown').astype(str).str.contains('Claude', case=False, na=False)].shape[0]
        
        # Count use cases with queries (if Query column exists)
        has_query_col = 'Query' in df.columns
        query_count = 0
        if has_query_col:
            query_count = df[df['Query'].fillna('').astype(str).str.strip() != ''].shape[0]
        
        with col1:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-value">{}</div>
                <div class="metric-label">Security Use Cases</div>
            </div>
            """.format(len(df)), unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-value">{}</div>
                <div class="metric-label">Mapped Techniques</div>
            </div>
            """.format(covered_techniques), unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-value">{}%</div>
                <div class="metric-label">Framework Coverage</div>
            </div>
            """.format(coverage_percent), unsafe_allow_html=True)
        
        with col4:
            if has_query_col:
                st.markdown("""
                <div class="metric-card">
                    <div class="metric-value">{}</div>
                    <div class="metric-label">Use Cases with Queries</div>
                </div>
                """.format(query_count), unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="metric-card">
                    <div class="metric-value">{} / {}</div>
                    <div class="metric-label">Library Matches / Claude AI Matches</div>
                </div>
                """.format(library_matches, claude_matches), unsafe_allow_html=True)
        
        # Additional metrics row if Query column exists
        if has_query_col:
            col5, col6, col7, col8 = st.columns(4)
            
            with col5:
                st.markdown("""
                <div class="metric-card">
                    <div class="metric-value">{} / {}</div>
                    <div class="metric-label">Library / Claude Matches</div>
                </div>
                """.format(library_matches, claude_matches), unsafe_allow_html=True)
            
            with col6:
                query_percentage = round((query_count / len(df)) * 100, 1) if len(df) > 0 else 0
                st.markdown("""
                <div class="metric-card">
                    <div class="metric-value">{}%</div>
                    <div class="metric-label">Use Cases with Queries</div>
                </div>
                """.format(query_percentage), unsafe_allow_html=True)
            
            with col7:
                # Count use cases mapped via query analysis (this would be a subset of Claude matches)
                query_mapped = df[(df['Match Source'].fillna('Unknown').astype(str).str.contains('Claude', case=False, na=False)) & 
                                (df['Query'].fillna('').astype(str).str.strip() != '')].shape[0]
                st.markdown("""
                <div class="metric-card">
                    <div class="metric-value">{}</div>
                    <div class="metric-label">Query-Enhanced Mappings</div>
                </div>
                """.format(query_mapped), unsafe_allow_html=True)
            
            with col8:
                # Average confidence score
                avg_confidence = round(df['Confidence Score (%)'].fillna(0).mean(), 1)
                st.markdown("""
                <div class="metric-card">
                    <div class="metric-value">{}%</div>
                    <div class="metric-label">Avg Confidence Score</div>
                </div>
                """.format(avg_confidence), unsafe_allow_html=True)
        
        # Match source chart
        st.markdown("### Mapping Source Distribution")
        
        # Handle empty or all-NaN columns
        if not df['Match Source'].isna().all():
            match_source_counts = df['Match Source'].fillna('Unknown').value_counts().reset_index()
            match_source_counts.columns = ['Source', 'Count']
            
            # Create chart only if there's data
            if not match_source_counts.empty:
                fig_source = px.pie(
                    match_source_counts, 
                    values='Count', 
                    names='Source',
                    title="Distribution of Mapping Sources",
                    hole=0.5,
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                
                fig_source.update_layout(
                    legend=dict(orientation="h", yanchor="bottom", y=-0.2)
                )
                st.plotly_chart(fig_source, use_container_width=True)
            else:
                st.info("No mapping source data available for visualization.")
        else:
            st.info("No mapping source data available for visualization.")
        
        # Coverage by Tactic - Doughnut Chart with better color scheme
        st.markdown("### Coverage by Tactic")
        
        # Create data for tactic coverage - with normalization to fix duplicates
        tactic_counts = {}
        for _, row in df.iterrows():
            tactic_str = row.get('Mapped MITRE Tactic(s)', '')
            if pd.isna(tactic_str) or tactic_str == 'N/A':
                continue
                
            # Split and normalize each tactic to avoid duplicates
            for tactic in str(tactic_str).split(','):
                tactic = tactic.strip()
                if tactic and tactic != 'N/A':
                    # Normalize the tactic name to avoid duplicates
                    normalized_tactic = normalize_tactic(tactic)
                    tactic_counts[normalized_tactic] = tactic_counts.get(normalized_tactic, 0) + 1
        
        # Transform to dataframe for visualization
        tactic_df = pd.DataFrame({
            'Tactic': list(tactic_counts.keys()),
            'Use Cases': list(tactic_counts.values())
        }).sort_values('Use Cases', ascending=False)
        
        if not tactic_df.empty:
            # Create doughnut chart for tactic coverage with better colors
            fig_tactic = go.Figure(data=[go.Pie(
                labels=tactic_df['Tactic'],
                values=tactic_df['Use Cases'],
                hole=.5,
                textposition='outside',  # Modified: This ensures all labels are outside
                textinfo='label+percent',
                marker=dict(colors=px.colors.qualitative.Dark24)  # Using Dark24 for better contrast
            )])
            
            fig_tactic.update_layout(
                title="Security Use Cases by MITRE Tactic",
                showlegend=False,  # Remove legend to prevent overlap
                margin=dict(t=50, b=50, l=100, r=100)  # Added: Margin for external labels
            )
            
            st.plotly_chart(fig_tactic, use_container_width=True)
        else:
            st.info("No tactic data available for visualization.")
        
        # Coverage by Technique - Doughnut Chart with better naming
        st.markdown("### Coverage by Technique")
        
        if processed_techniques_count:
            # Get top techniques for the chart (limiting to top 10 for readability)
            technique_ids = list(processed_techniques_count.keys())
            technique_counts = list(processed_techniques_count.values())
            
            # Get technique names - with improved formatting
            technique_names = []
            for tech_id in technique_ids:
                # Find the technique name using the mapping
                tech_name = technique_name_mapping.get(tech_id, tech_id)
                technique_names.append(tech_name)
            
            technique_df = pd.DataFrame({
                'Technique': technique_names,
                'Count': technique_counts
            }).sort_values('Count', ascending=False).head(10)
            
            # Create doughnut chart for technique coverage with better colors
            fig_tech = go.Figure(data=[go.Pie(
                labels=technique_df['Technique'],
                values=technique_df['Count'],
                hole=.5,
                textposition='outside',  # Modified: This ensures all labels are outside
                textinfo='label+percent',
                marker=dict(colors=px.colors.qualitative.Bold)  # Using Bold color scheme for better contrast
            )])
            
            fig_tech.update_layout(
                title="Top 10 MITRE Techniques in Security Use Cases",
                showlegend=False,  # Remove legend to prevent overlap
                margin=dict(t=50, b=50, l=100, r=100)  # Added: Margin for external labels
            )
            
            st.plotly_chart(fig_tech, use_container_width=True)
        else:
            st.info("No technique data available for visualization.")
        
        # Query Analysis (if Query column exists)
        if has_query_col and query_count > 0:
            st.markdown("### Query Analysis")
            
            # Create tabs for different query insights
            tab1, tab2 = st.tabs(["Query Types", "Query vs Description Mapping"])
            
            with tab1:
                # Analyze query types based on keywords
                query_types = {}
                for _, row in df.iterrows():
                    query = str(row.get('Query', '')).lower()
                    if not query or query == 'nan' or query == 'n/a':
                        continue
                    
                    # Simple keyword-based classification
                    if any(keyword in query for keyword in ['select', 'from', 'where']):
                        query_types['SQL Query'] = query_types.get('SQL Query', 0) + 1
                    elif any(keyword in query for keyword in ['event.code', 'winlog', 'sysmon']):
                        query_types['Windows Event Query'] = query_types.get('Windows Event Query', 0) + 1
                    elif any(keyword in query for keyword in ['process.name', 'file.path', 'network']):
                        query_types['Process/Network Query'] = query_types.get('Process/Network Query', 0) + 1
                    elif any(keyword in query for keyword in ['sourcetype', 'index', 'search']):
                        query_types['Splunk Query'] = query_types.get('Splunk Query', 0) + 1
                    elif any(keyword in query for keyword in ['query', 'filter', 'rule']):
                        query_types['Generic Rule'] = query_types.get('Generic Rule', 0) + 1
                    else:
                        query_types['Other'] = query_types.get('Other', 0) + 1
                
                if query_types:
                    query_type_df = pd.DataFrame({
                        'Query Type': list(query_types.keys()),
                        'Count': list(query_types.values())
                    })
                    
                    fig_query_types = px.bar(
                        query_type_df, 
                        x='Query Type', 
                        y='Count',
                        title="Distribution of Query Types",
                        color='Count',
                        color_continuous_scale='Blues'
                    )
                    
                    st.plotly_chart(fig_query_types, use_container_width=True)
                else:
                    st.info("No query type patterns detected.")
            
            with tab2:
                # Compare mapping sources for use cases with vs without queries
                mapping_comparison = {}
                
                # Use cases with queries
                with_queries = df[df['Query'].fillna('').astype(str).str.strip() != '']
                without_queries = df[df['Query'].fillna('').astype(str).str.strip() == '']
                
                for source_type, subset in [('With Queries', with_queries), ('Without Queries', without_queries)]:
                    if not subset.empty:
                        source_counts = subset['Match Source'].fillna('Unknown').value_counts()
                        for source, count in source_counts.items():
                            if source not in mapping_comparison:
                                mapping_comparison[source] = {}
                            mapping_comparison[source][source_type] = count
                
                if mapping_comparison:
                    comparison_df = pd.DataFrame(mapping_comparison).fillna(0).T
                    comparison_df.reset_index(inplace=True)
                    comparison_df.rename(columns={'index': 'Match Source'}, inplace=True)
                    
                    fig_comparison = px.bar(
                        comparison_df, 
                        x='Match Source',
                        y=['With Queries', 'Without Queries'],
                        title="Mapping Sources: Use Cases With vs Without Queries",
                        barmode='group',
                        color_discrete_sequence=['#2E8B57', '#4682B4']
                    )
                    
                    st.plotly_chart(fig_comparison, use_container_width=True)
                    
                    # Display summary statistics
                    st.markdown("#### Summary Statistics")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Use Cases with Queries", len(with_queries))
                        if len(with_queries) > 0:
                            avg_conf_with = round(with_queries['Confidence Score (%)'].mean(), 1)
                            st.metric("Avg Confidence (with queries)", f"{avg_conf_with}%")
                    
                    with col2:
                        st.metric("Use Cases without Queries", len(without_queries))
                        if len(without_queries) > 0:
                            avg_conf_without = round(without_queries['Confidence Score (%)'].mean(), 1)
                            st.metric("Avg Confidence (without queries)", f"{avg_conf_without}%")
                else:
                    st.info("Insufficient data for comparison analysis.")
    
    else:
        st.info("No analytics data available. Please upload a CSV file on the Home page and complete the mapping process.")
        
        # Add a button to navigate back to home
        if st.button("Go to Home"):
            st.session_state.page = "home"
            st.experimental_rerun()
