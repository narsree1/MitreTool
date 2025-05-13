import datetime
import json
import uuid
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import re

# Function to standardize tactic names 
def standardize_tactic_name(tactic):
    """
    Standardize tactic names to ensure consistent format.
    This handles variations like 'command and control', 'Command-And-Control', etc.
    
    Args:
        tactic: Original tactic name
        
    Returns:
        Standardized tactic name
    """
    if not tactic or tactic == 'N/A':
        return 'N/A'
    
    # Convert to lowercase for comparison
    tactic_lower = tactic.lower()
    
    # Dictionary of standardized names
    standard_names = {
        'command and control': 'Command And Control',
        'command-and-control': 'Command And Control',
        'command&control': 'Command And Control',
        'command_and_control': 'Command And Control',
        'command-and-control': 'Command And Control',
        'commandand control': 'Command And Control',
        'command & control': 'Command And Control',
        'commandcontrol': 'Command And Control',
        
        'discovery': 'Discovery',
        
        'execution': 'Execution',
        
        'privilege escalation': 'Privilege Escalation',
        'privilegeescalation': 'Privilege Escalation',
        'privilege_escalation': 'Privilege Escalation',
        
        'initial access': 'Initial Access',
        'initialaccess': 'Initial Access',
        'initial-access': 'Initial Access',
        'initial_access': 'Initial Access',
        
        'persistence': 'Persistence',
        
        'credential access': 'Credential Access',
        'credentialaccess': 'Credential Access',
        'credential-access': 'Credential Access',
        'credential_access': 'Credential Access',
        
        'lateral movement': 'Lateral Movement',
        'lateralmovement': 'Lateral Movement',
        'lateral-movement': 'Lateral Movement',
        'lateral_movement': 'Lateral Movement',
        
        'defense evasion': 'Defense Evasion',
        'defenseevasion': 'Defense Evasion',
        'defense-evasion': 'Defense Evasion',
        'defense_evasion': 'Defense Evasion',
        
        'collection': 'Collection',
        
        'impact': 'Impact',
        
        'exfiltration': 'Exfiltration',
        
        'resource development': 'Resource Development',
        'resourcedevelopment': 'Resource Development',
        'resource-development': 'Resource Development',
        'resource_development': 'Resource Development',
        
        'reconnaissance': 'Reconnaissance'
    }
    
    # Try to match with standardized names
    for pattern, standard in standard_names.items():
        if tactic_lower == pattern:
            return standard
    
    # If not found in dictionary, use Title Case
    return ' '.join(word.capitalize() for word in tactic_lower.split())

# Function to standardize technique names
def standardize_technique_name(technique):
    """
    Standardize technique names to ensure consistent format.
    Handles variations in capitalization, spacing, and removes IDs.
    
    Args:
        technique: Original technique name
        
    Returns:
        Standardized technique name
    """
    if not technique or technique == 'N/A':
        return 'N/A'
    
    # If it has the format "T#### - Name", extract just the name
    if ' - ' in technique and technique.split(' - ')[0].startswith('T'):
        technique = technique.split(' - ', 1)[1]
    
    # Convert to lowercase for comparison
    technique_lower = technique.lower()
    
    # Dictionary of common technique name variations
    standard_techniques = {
        'brute force': 'Brute Force',
        'bruteforce': 'Brute Force',
        'brute-force': 'Brute Force',
        
        'powershell': 'PowerShell',
        'power shell': 'PowerShell',
        'power-shell': 'PowerShell',
        
        'command and scripting interpreter': 'Command and Scripting Interpreter',
        'command & scripting interpreter': 'Command and Scripting Interpreter',
        
        'modify registry': 'Modify Registry',
        'registry modification': 'Modify Registry',
        
        'credential dumping': 'Credential Dumping',
        'credential dump': 'Credential Dumping',
        
        'phishing': 'Phishing',
        'spear phishing': 'Phishing',
        
        'data exfiltration': 'Data Exfiltration',
        'exfiltration over alternative protocol': 'Exfiltration Over Alternative Protocol',
        'dns exfiltration': 'Exfiltration Over Alternative Protocol: DNS',
        'dns tunneling': 'Exfiltration Over Alternative Protocol: DNS',
        'exfiltration over dns': 'Exfiltration Over Alternative Protocol: DNS',
        
        'execution through api': 'Execution Through API',
        'api execution': 'Execution Through API',
        
        'scheduled task': 'Scheduled Task/Job',
        'scheduled tasks': 'Scheduled Task/Job',
        'scheduled job': 'Scheduled Task/Job',
        'scheduled task/job': 'Scheduled Task/Job',
        
        'valid accounts': 'Valid Accounts',
        'valid account': 'Valid Accounts',
        
        'pass the hash': 'Pass the Hash',
        'passthehash': 'Pass the Hash',
        'pass-the-hash': 'Pass the Hash',
        
        'multi-hop proxy': 'Multi-hop Proxy',
        'multihop proxy': 'Multi-hop Proxy',
        'multi hop proxy': 'Multi-hop Proxy',
        
        'office application startup': 'Office Application Startup',
        
        'process injection': 'Process Injection',
    }
    
    # Try to match with standardized names
    for pattern, standard in standard_techniques.items():
        if technique_lower == pattern:
            return standard
    
    # If not found in dictionary, use the original with proper Title Case
    return ' '.join(word.capitalize() if word.lower() not in ['and', 'or', 'the', 'in', 'on', 'at', 'to'] 
                   else word.lower() for word in technique.split())

def create_metrics_display(num_use_cases, covered_techniques, coverage_percent, library_matches, model_matches):
    """
    Create HTML for metrics display cards
    
    Args:
        num_use_cases: Number of security use cases
        covered_techniques: Number of MITRE techniques covered
        coverage_percent: Percentage of framework coverage
        library_matches: Number of library matches
        model_matches: Number of model-based matches
    """
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{num_use_cases}</div>
            <div class="metric-label">Security Use Cases</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{covered_techniques}</div>
            <div class="metric-label">Mapped Techniques</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{coverage_percent}%</div>
            <div class="metric-label">Framework Coverage</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{library_matches} / {model_matches}</div>
            <div class="metric-label">Library Matches / Model Matches</div>
        </div>
        """, unsafe_allow_html=True)

def create_source_chart(df):
    """
    Create a pie chart showing distribution of mapping sources
    
    Args:
        df: DataFrame containing mapping results
        
    Returns:
        fig: Plotly figure object or None if no data
    """
    # Handle empty or all-NaN columns
    if df is None or 'Match Source' not in df.columns or df['Match Source'].isna().all():
        return None
        
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
        return fig_source
    
    return None

def create_tactic_chart(df):
    """
    Create a doughnut chart showing coverage by tactic
    
    Args:
        df: DataFrame containing mapping results
        
    Returns:
        fig: Plotly figure object or None if no data
    """
    if df is None or 'Mapped MITRE Tactic(s)' not in df.columns:
        return None
        
    # Create data for tactic coverage with standardization
    tactic_counts = {}
    for _, row in df.iterrows():
        tactic_str = row.get('Mapped MITRE Tactic(s)', '')
        if pd.isna(tactic_str):
            continue
            
        for tactic in str(tactic_str).split(', '):
            if tactic and tactic != 'N/A':
                # Standardize tactic name before counting
                standard_tactic = standardize_tactic_name(tactic)
                tactic_counts[standard_tactic] = tactic_counts.get(standard_tactic, 0) + 1
    
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
            textposition='outside',  # This ensures all labels are outside
            textinfo='label+percent',
            marker=dict(colors=px.colors.qualitative.Dark24)  # Using Dark24 for better contrast
        )])
        
        fig_tactic.update_layout(
            title="Security Use Cases by MITRE Tactic",
            showlegend=False,  # Remove legend to prevent overlap
            margin=dict(t=50, b=50, l=100, r=100)  # Margin for external labels
        )
        
        return fig_tactic
    
    return None

def create_technique_chart(df, techniques_count, mitre_techniques):
    """
    Create a doughnut chart showing coverage by technique
    
    Args:
        df: DataFrame containing mapping results
        techniques_count: Dictionary counting occurrences of each technique
        mitre_techniques: List of MITRE technique dictionaries
        
    Returns:
        fig: Plotly figure object or None if no data
    """
    if not techniques_count:
        return None
        
    # Standardize technique names in the count dictionary
    standardized_counts = {}
    for tech_name, count in techniques_count.items():
        standard_name = standardize_technique_name(tech_name)
        standardized_counts[standard_name] = standardized_counts.get(standard_name, 0) + count
        
    # Get top techniques for the chart (limiting to top 10 for readability)
    technique_names = list(standardized_counts.keys())
    technique_counts = list(standardized_counts.values())
    
    # Create dataframe for visualization
    technique_df = pd.DataFrame({
        'Technique': technique_names,
        'Count': technique_counts
    }).sort_values('Count', ascending=False).head(10)
    
    # Create doughnut chart for technique coverage with better colors
    fig_tech = go.Figure(data=[go.Pie(
        labels=technique_df['Technique'],
        values=technique_df['Count'],
        hole=.5,
        textposition='outside',  # This ensures all labels are outside
        textinfo='label+percent',
        marker=dict(colors=px.colors.qualitative.Bold)  # Using Bold color scheme for better contrast
    )])
    
    fig_tech.update_layout(
        title="Top 10 MITRE Techniques in Security Use Cases",
        showlegend=False,  # Remove legend to prevent overlap
        margin=dict(t=50, b=50, l=100, r=100)  # Margin for external labels
    )
    
    return fig_tech

def create_navigator_layer(techniques_count):
    """
    Create a MITRE ATT&CK Navigator layer from technique count data
    
    Args:
        techniques_count: Dictionary counting occurrences of each technique
        
    Returns:
        layer_json: JSON string of Navigator layer
        layer_id: Unique ID for the layer
    """
    try:
        # Standardize technique names in the count dictionary
        standardized_counts = {}
        for tech_name, count in techniques_count.items():
            standard_name = standardize_technique_name(tech_name)
            standardized_counts[standard_name] = standardized_counts.get(standard_name, 0) + count
            
        # Common technique names to IDs mapping
        # This is a simplified mapping - in production you would have a complete mapping
        technique_to_id = {
            'Brute Force': 'T1110',
            'PowerShell': 'T1059.001',
            'Command and Scripting Interpreter': 'T1059',
            'Modify Registry': 'T1112',
            'Credential Dumping': 'T1003',
            'Phishing': 'T1566',
            'Exfiltration Over Alternative Protocol': 'T1048',
            'Exfiltration Over Alternative Protocol: DNS': 'T1048.003',
            'Execution Through API': 'T1106',
            'Scheduled Task/Job': 'T1053',
            'Valid Accounts': 'T1078',
            'Pass the Hash': 'T1550.002',
            'Multi-hop Proxy': 'T1090.003',
            'Office Application Startup': 'T1137',
            'Process Injection': 'T1055'
        }
        
        techniques_data = []
        for tech_name, count in standardized_counts.items():
            # Try to find technique ID
            tech_id = technique_to_id.get(tech_name)
            
            # If it's already in ID format, use it directly
            if not tech_id and tech_name.startswith('T') and tech_name[1:].isdigit():
                tech_id = tech_name
                
            # If we still don't have an ID, use a placeholder
            if not tech_id:
                # For now, use a generic ID as placeholder
                tech_id = "T9999"
                
            techniques_data.append({
                "techniqueID": tech_id,
                "score": count,
                "color": "",
                "comment": f"{tech_name}: {count}",
                "enabled": True,
                "metadata": [],
                "links": [],
                "showSubtechniques": False
            })
        
        current_date = datetime.datetime.now().strftime("%Y-%m-%d")
        layer_id = str(uuid.uuid4())
        
        layer = {
            "name": f"Security Use Cases Mapping - {current_date}",
            "versions": {
                "attack": "17",
                "navigator": "4.8.1",
                "layer": "4.4"
            },
            "domain": "enterprise-attack",
            "description": f"Mapping of security use cases to MITRE ATT&CK techniques, generated on {current_date}",
            "filters": {
                "platforms": ["Linux", "macOS", "Windows", "Network", "PRE", "Containers", "Office 365", "SaaS", "IaaS", "Google Workspace", "Azure AD"]
            },
            "sorting": 0,
            "layout": {
                "layout": "side",
                "aggregateFunction": "max",
                "showID": True,
                "showName": True,
                "showAggregateScores": True,
                "countUnscored": False
            },
            "hideDisabled": False,
            "techniques": techniques_data,
            "gradient": {
                "colors": ["#ffffff", "#66b1ff", "#0d4a90"],
                "minValue": 0,
                "maxValue": max(standardized_counts.values()) if standardized_counts else 1
            },
            "legendItems": [],
            "metadata": [],
            "links": [],
            "showTacticRowBackground": True,
            "tacticRowBackground": "#dddddd",
            "selectTechniquesAcrossTactics": True,
            "selectSubtechniquesWithParent": False
        }
        
        return json.dumps(layer, indent=2), layer_id
    except Exception as e:
        st.error(f"Error creating Navigator layer: {e}")
        return "{}", ""
