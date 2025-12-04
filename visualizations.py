









import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import networkx as nx
from pyvis.network import Network
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import sqlite3
from datetime import datetime, timedelta
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def render_visualizations(training_df, history, DB_PATH):
    """
    Ultimate AI Medical Recommendation System Visualizations
    Arguments:
        training_df: pandas DataFrame containing symptom-disease training data
        history: list of past predictions stored in session_state
        DB_PATH: path to the SQLite database for all users history
    """
    # ========== CUSTOM CSS FOR SEAMLESS DESIGN ==========
    st.markdown("""
        <style>
        /* Remove default Streamlit padding and margins */
        .block-container {
            padding-top: 2rem;
            padding-bottom: 0rem;
        }
        /* Custom container styling */
        .viz-container {
            background: white;
            border-radius: 20px;
            padding: 2em;
            margin-bottom: 2em;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        /* Metric cards styling */
        .stMetric {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 1em;
            border-radius: 12px;
            color: white;
        }
        /* Plotly chart container */
        .js-plotly-plot {
            border-radius: 15px;
            overflow: hidden;
        }
        </style>
    """, unsafe_allow_html=True)
    st.markdown("""
        <h1 style='
            text-align: center;
            background: linear-gradient(90deg, #4ade80 0%, #38bdf8 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 3em;
            font-weight: bold;
            margin-bottom: 0.5em;
        '>
        🌈 AI-Driven Health Insights Dashboard
        </h1>
    """, unsafe_allow_html=True)
    # Sidebar controls
    with st.sidebar:
        st.markdown("### 🎨 Dashboard Controls")
        color_scheme = st.selectbox("Color Theme", 
                                    ["Electric Blue", "Medical Green", "Royal Purple", "Sunset Orange", "Deep Ocean"])
        animation_speed = st.slider("Animation Speed (ms)", 500, 3000, 1500, step=100)
        show_theory = st.checkbox("Show Theory Sections", value=True)
    # Color scheme definitions
    color_themes = {
        "Electric Blue": {
            "primary": "#3b82f6", "secondary": "#06b6d4", "accent": "#8b5cf6",
            "gradient1": "#667eea", "gradient2": "#764ba2", "success": "#10b981", "danger": "#ef4444"
        },
        "Medical Green": {
            "primary": "#059669", "secondary": "#10b981", "accent": "#34d399",
            "gradient1": "#0891b2", "gradient2": "#059669", "success": "#22c55e", "danger": "#f87171"
        },
        "Royal Purple": {
            "primary": "#7c3aed", "secondary": "#a855f7", "accent": "#c084fc",
            "gradient1": "#7c3aed", "gradient2": "#ec4899", "success": "#10b981", "danger": "#f43f5e"
        },
        "Sunset Orange": {
            "primary": "#f59e0b", "secondary": "#fb923c", "accent": "#fbbf24",
            "gradient1": "#f59e0b", "gradient2": "#ef4444", "success": "#10b981", "danger": "#dc2626"
        },
        "Deep Ocean": {
            "primary": "#0284c7", "secondary": "#0891b2", "accent": "#06b6d4",
            "gradient1": "#0ea5e9", "gradient2": "#6366f1", "success": "#14b8a6", "danger": "#f43f5e"
        }
    }
    colors = color_themes[color_scheme]



    # ========== 1. ENHANCED SYMPTOM CO-OCCURRENCE HEATMAP (WHITE BACKGROUND) ==========
    # Only one version, with unique keys and Streamlit formatting and all theory blocks.
    if training_df is not None and not training_df.empty:
        st.markdown(f"""
            <h2 style='color: {colors['primary']}; font-size: 2em; margin-bottom: 0.5em;'>
            🔥 Symptom Interplay Matrix
            </h2>
        """, unsafe_allow_html=True)
        if show_theory:
            st.markdown(f"""
                <div style='background: linear-gradient(135deg, {colors['gradient1']}15 0%, {colors['gradient2']}15 100%); 
                padding: 1.5em; border-radius: 15px; margin-bottom: 1.5em; border-left: 5px solid {colors['primary']};'>
                <h4 style='color: {colors['primary']};'>🔬 Scientific Theory & Clinical Application</h4>
                <p><b>Mathematical Foundation:</b> This visualization uses Pearson correlation coefficients (ρ) to measure 
                linear relationships between symptom pairs. The correlation matrix C is computed as:</p>
                <p style='text-align: center; background: #f0f9ff; color: #1e3a8a; padding: 0.5em 1em; border-radius: 8px; font-family: monospace; font-size: 1.1em;'>
                ρ(X,Y) = Cov(X,Y) / (σ_X × σ_Y)</p>
                <p><b>📊 Interpretation Guide:</b></p>
                <ul>
                    <li><b style='color: #dc2626;'>Deep Red (0.7 to 1.0):</b> Strong positive correlation - symptoms frequently co-occur, 
                    suggesting shared pathophysiological mechanisms or common disease syndromes</li>
                    <li><b style='color: #3b82f6;'>Deep Blue (-1.0 to -0.7):</b> Strong negative correlation - symptoms rarely occur together, 
                    indicating mutually exclusive conditions or diagnostic differentiation points</li>
                    <li><b style='color: #6b7280;'>Gray (Near 0):</b> No significant relationship - symptoms are independent</li>
                </ul>
                <p><b>💡 Clinical Significance:</b></p>
                <ul>
                    <li><b>Differential Diagnosis:</b> Identify distinguishing symptom patterns</li>
                    <li><b>Syndrome Recognition:</b> Discover symptom clusters indicating specific diseases</li>
                    <li><b>Diagnostic Accuracy:</b> Understand which symptoms provide complementary information</li>
                    <li><b>Medical Education:</b> Visual tool for teaching symptom relationships</li>
                </ul>
                </div>
            """, unsafe_allow_html=True)
        # Prepare data
        symptom_cols = training_df.columns[:-1]
        freq = training_df[symptom_cols].sum().sort_values(ascending=False)[:15]
        top_symptoms = list(freq.index)
        corr_matrix = training_df[top_symptoms].corr()
        # Create enhanced Plotly heatmap
        fig_heatmap = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=[s.replace('_', ' ').title() for s in top_symptoms],
            y=[s.replace('_', ' ').title() for s in top_symptoms],
            colorscale=[
                [0.0, '#0c4a6e'],
                [0.25, '#3b82f6'],
                [0.5, '#f3f4f6'],
                [0.75, '#fb923c'],
                [1.0, '#dc2626']
            ],
            zmid=0,
            zmin=-1,
            zmax=1,
            text=corr_matrix.values,
            texttemplate='%{z:.2f}',
            textfont={"size": 11, "color": "black"},
            hovertemplate='<b>%{y}</b> ↔ <b>%{x}</b><br>' +
                         'Correlation: <b>%{z:.3f}</b><br>' +
                         '<extra></extra>',
            colorbar=dict(
                title="Correlation<br>Coefficient",
                tickmode="linear",
                tick0=-1,
                dtick=0.5,
                thickness=20,
                len=0.7,
                x=1.02
            )
        ))
        fig_heatmap.update_layout(
            title=dict(
                text='<b>Symptom Correlation Matrix with Hierarchical Ordering</b>',
                font=dict(
                    size=22,
                    family="Inter, sans-serif",
                    color=colors['primary']
                ),
                x=0.5,
                xanchor='center'
            ),
            xaxis=dict(
                tickangle=45,
                tickfont=dict(size=13, family="Inter, sans-serif", color='#1f2937'),
                side='bottom',
                showgrid=False
            ),
            yaxis=dict(
                tickfont=dict(size=13, family="Inter, sans-serif", color='#1f2937'),
                showgrid=False
            ),
            width=1200,
            height=800,
            paper_bgcolor='white',
            plot_bgcolor='white',
            font=dict(family="Inter, sans-serif"),
            margin=dict(l=200, r=150, t=100, b=200)
        )
        st.plotly_chart(fig_heatmap, use_container_width=True, key="heatmap_chart")
        # Top Symptoms Frequency Bar Chart
        st.markdown(f"""
            <h2 style='color: {colors['primary']}; font-size: 2em; margin-bottom: 0.5em;'>
            🌟 Symptom Occurrence Leaderboard
            </h2>
        """, unsafe_allow_html=True)
        symptom_counts = training_df[symptom_cols].sum().sort_values(ascending=False)[:15]
        fig_top_symptoms = go.Figure()
        fig_top_symptoms.add_trace(go.Bar(
            x=[s.replace('_', ' ').title() for s in symptom_counts.index],
            y=symptom_counts.values,
            marker_color=colors['primary'],
            text=symptom_counts.values,
            textposition='auto',
            hovertemplate='<b>%{x}</b><br>Frequency: %{y}<extra></extra>'
        ))
        fig_top_symptoms.update_layout(
            title='Top 15 Symptoms by Occurrence',
            xaxis_title='Symptoms',
            yaxis_title='Frequency',
            paper_bgcolor='white',
            plot_bgcolor='white',
            font=dict(family="Inter, sans-serif", size=14),
            height=450
        )
        st.plotly_chart(fig_top_symptoms, use_container_width=True, key="top_symptoms_chart")
        # Correlation strength distribution
        st.markdown(f"<h4 style='color: {colors['primary']};'>📊 Correlation Distribution Analysis</h4>", 
                   unsafe_allow_html=True)
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        upper_triangle = corr_matrix.where(mask)
        corr_values = upper_triangle.values.flatten()
        corr_values = corr_values[~np.isnan(corr_values)]
        fig_dist = go.Figure()
        fig_dist.add_trace(go.Histogram(
            x=corr_values,
            nbinsx=30,
            marker=dict(
                color=corr_values,
                colorscale='RdBu',
                line=dict(color='white', width=1)
            ),
            hovertemplate='Correlation Range: %{x}<br>Count: %{y}<extra></extra>'
        ))
        fig_dist.update_layout(
            title='Distribution of Symptom Correlations',
            xaxis_title='Correlation Coefficient',
            yaxis_title='Frequency',
            paper_bgcolor='white',
            plot_bgcolor='white',
            font=dict(family="Inter, sans-serif", size=14),
            height=400
        )
        st.plotly_chart(fig_dist, use_container_width=True, key="distribution_chart")
        strong_positive = (corr_values > 0.7).sum()
        strong_negative = (corr_values < -0.7).sum()
        weak = ((corr_values >= -0.3) & (corr_values <= 0.3)).sum()
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"""
                <div style='background: linear-gradient(135deg, {colors['success']}20, {colors['success']}40); 
                padding: 1.5em; border-radius: 12px; text-align: center;'>
                <h3 style='color: {colors['success']}; margin: 0;'>{strong_positive}</h3>
                <p style='margin: 0.5em 0 0 0; color: #374151;'>Strong Positive</p>
                </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
                <div style='background: linear-gradient(135deg, {colors['danger']}20, {colors['danger']}40); 
                padding: 1.5em; border-radius: 12px; text-align: center;'>
                <h3 style='color: {colors['danger']}; margin: 0;'>{strong_negative}</h3>
                <p style='margin: 0.5em 0 0 0; color: #374151;'>Strong Negative</p>
                </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
                <div style='background: linear-gradient(135deg, #9ca3af20, #9ca3af40); 
                padding: 1.5em; border-radius: 12px; text-align: center;'>
                <h3 style='color: #6b7280; margin: 0;'>{weak}</h3>
                <p style='margin: 0.5em 0 0 0; color: #374151;'>Weak/No Correlation</p>
                </div>
            """, unsafe_allow_html=True)
        with col4:
            avg_corr = np.mean(np.abs(corr_values))
            st.markdown(f"""
                <div style='background: linear-gradient(135deg, {colors['primary']}20, {colors['primary']}40); 
                padding: 1.5em; border-radius: 12px; text-align: center;'>
                <h3 style='color: {colors['primary']}; margin: 0;'>{avg_corr:.3f}</h3>
                <p style='margin: 0.5em 0 0 0; color: #374151;'>Avg |Correlation|</p>
                </div>
            """, unsafe_allow_html=True)
    # ========== 2. SELF-ROTATING ANIMATED NETWORK GRAPH ==========
    # ========== 2. SELF-ROTATING ANIMATED NETWORK GRAPH ==========
    # Only one version, with unique key and all theory blocks.
    if training_df is not None and not training_df.empty:
        st.markdown(f"""
            <h2 style='color: {colors['primary']}; font-size: 2em; margin-bottom: 0.5em;'>
            🕸️ Symptom-Disease Connectivity Map
            </h2>
        """, unsafe_allow_html=True)
        if show_theory:
            st.markdown(f"""
                <div style='background: linear-gradient(135deg, {colors['gradient1']}15 0%, {colors['gradient2']}15 100%); 
                padding: 1.5em; border-radius: 15px; margin-bottom: 1.5em; border-left: 5px solid {colors['primary']};'>
                <h4 style='color: {colors['primary']};'>🔬 Graph Theory & Network Science</h4>
                <p><b>Mathematical Foundation:</b> This visualization represents a bipartite graph G = (S ∪ D, E) where:</p>
                <ul>
                    <li><b>S</b> = Set of symptom nodes</li>
                    <li><b>D</b> = Set of disease nodes</li>
                    <li><b>E</b> = Edges representing symptom-disease associations</li>
                </ul>
                <p><b>Network Physics:</b> Uses force-directed layout algorithm with:</p>
                <ul>
                    <li><b>Spring Forces:</b> Attract connected nodes (Hooke's Law: F = -kx)</li>
                    <li><b>Repulsive Forces:</b> Push nodes apart (Coulomb's Law: F = k(q₁q₂)/r²)</li>
                    <li><b>Gravity:</b> Prevents network from flying apart</li>
                    <li><b>Damping:</b> Reduces oscillations for stability</li>
                </ul>
                <p><b>🎮 Interactive Features:</b></p>
                <ul>
                    <li><b>Auto-Rotation:</b> Network continuously rotates for 360° exploration</li>
                    <li><b>Drag & Drop:</b> Click and drag nodes to explore connections</li>
                    <li><b>Hover Details:</b> Node information and connection strength</li>
                    <li><b>Physics Simulation:</b> Real-time force calculations</li>
                </ul>
                <p><b>💡 Clinical Applications:</b></p>
                <ul>
                    <li><b>Hub Identification:</b> Central symptoms affecting multiple diseases</li>
                    <li><b>Community Detection:</b> Disease clusters with similar symptom profiles</li>
                    <li><b>Path Analysis:</b> Symptom chains in disease progression</li>
                    <li><b>Diagnostic Trees:</b> Decision pathways for differential diagnosis</li>
                </ul>
                </div>
            """, unsafe_allow_html=True)
        # Build network
        G = nx.Graph()
        top_symptoms_net = training_df[symptom_cols].sum().sort_values(ascending=False).head(10).index.tolist()
        top_diseases_net = training_df['prognosis'].value_counts().head(8).index.tolist()
        symptom_freq = training_df[top_symptoms_net].sum()
        disease_freq = training_df['prognosis'].value_counts()
        for symptom in top_symptoms_net:
            G.add_node(symptom, 
                      type='symptom', 
                      freq=int(symptom_freq[symptom]),
                      label=symptom.replace('_', ' ').title())
        for disease in top_diseases_net:
            G.add_node(disease, 
                      type='disease', 
                      freq=int(disease_freq[disease]),
                      label=disease)
        for disease in top_diseases_net:
            disease_data = training_df[training_df['prognosis'] == disease]
            for symptom in top_symptoms_net:
                association = disease_data[symptom].mean()
                if association > 0.25:
                    G.add_edge(symptom, disease, weight=float(association))
        net = Network(
            height="800px",
            width="100%",
            bgcolor="white",
            font_color="#1f2937",
            directed=False,
        )
        pos = nx.circular_layout(G)
        for n in pos:
            pos[n] = pos[n] * 350
        symptom_color = colors['primary']
        disease_color = colors['secondary']
        for n, attr in G.nodes(data=True):
            n_color = symptom_color if attr['type'] == 'symptom' else disease_color
            freq_val = attr['freq']
            size = 30 + (freq_val / 3)
            n_title = f"""
            <div style='font-family: Inter, sans-serif; padding: 10px;'>
                <b style='font-size: 16px;'>{attr['label']}</b><br>
                <span style='color: #6b7280;'>Type:</span> {attr['type'].capitalize()}<br>
                <span style='color: #6b7280;'>Frequency:</span> {freq_val}<br>
                <span style='color: #6b7280;'>Connections:</span> {G.degree(n)}
            </div>
            """
            border_color = colors['accent'] if attr['type'] == 'symptom' else colors['success']
            x, y = float(pos[n][0]), float(pos[n][1])
            net.add_node(
                n,
                size=size,
                title=n_title,
                borderWidth=3,
                borderWidthSelected=6,
                color={'border': border_color, 'background': n_color, 'highlight': {'border': '#fbbf24', 'background': n_color}},
                font={'size': 16, 'color': '#1f2937', 'face': 'Inter, sans-serif', 'bold': True},
                x=x,
                y=y,
                label=attr['label'][:20]
            )
        for e in G.edges(data=True):
            weight = e[2].get('weight', 0.5)
            edge_width = weight * 6
            edge_color = f"rgba({int(colors['accent'][1:3], 16)}, {int(colors['accent'][3:5], 16)}, {int(colors['accent'][5:7], 16)}, {weight})"
            edge_title = f"""
            <div style='font-family: Inter, sans-serif; padding: 8px;'>
                <b>{e[0].replace('_', ' ').title()}</b> ↔ <b>{e[1]}</b><br>
                Association Strength: {weight:.2%}
            </div>
            """
            net.add_edge(
                e[0],
                e[1],
                value=edge_width,
                color=edge_color,
                title=edge_title,
            )
        net.set_options("""
        {
            "physics": {
                "enabled": true,
                "stabilization": {
                    "enabled": true,
                    "iterations": 100
                },
                "barnesHut": {
                    "gravitationalConstant": -8000,
                    "centralGravity": 0.3,
                    "springLength": 200,
                    "springConstant": 0.04,
                    "damping": 0.5,
                    "avoidOverlap": 0.2
                }
            },
            "interaction": {
                "hover": true,
                "dragNodes": true,
                "dragView": true,
                "zoomView": true,
                "tooltipDelay": 100,
                "navigationButtons": true,
                "keyboard": {
                    "enabled": true
                }
            },
            "nodes": {
                "shadow": {
                    "enabled": true,
                    "color": "rgba(0,0,0,0.2)",
                    "size": 10,
                    "x": 2,
                    "y": 2
                },
                "font": {
                    "size": 16,
                    "face": "Inter, sans-serif"
                }
            },
            "edges": {
                "smooth": {
                    "enabled": true,
                    "type": "continuous",
                    "roundness": 0.5
                },
                "shadow": {
                    "enabled": true,
                    "color": "rgba(0,0,0,0.1)",
                    "size": 5
                }
            }
        }
        """)
        net_html = net.generate_html(notebook=False)
        rotation_script = f"""
        <script type="text/javascript">
            (function() {{
                let angle = 0;
                let isRotating = true;
                let rotationSpeed = {animation_speed / 1000};
                setTimeout(function() {{
                    let network = window.network;
                    if (!network) return;
                    let controlDiv = document.createElement('div');
                    controlDiv.style.cssText = `
                        position: absolute;
                        top: 20px;
                        right: 20px;
                        z-index: 1000;
                        background: linear-gradient(135deg, {colors['gradient1']}, {colors['gradient2']});
                        color: white;
                        padding: 12px 24px;
                        border-radius: 25px;
                        cursor: pointer;
                        font-family: Inter, sans-serif;
                        font-weight: bold;
                        font-size: 14px;
                        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
                        transition: all 0.3s ease;
                    `;
                    controlDiv.innerHTML = '⏸️ Pause Rotation';
                    controlDiv.onmouseover = function() {{ this.style.transform = 'scale(1.05)'; }};
                    controlDiv.onmouseout = function() {{ this.style.transform = 'scale(1)'; }};
                    controlDiv.onclick = function() {{
                        isRotating = !isRotating;
                        this.innerHTML = isRotating ? '⏸️ Pause Rotation' : '▶️ Resume Rotation';
                    }};
                    document.getElementById('mynetwork').appendChild(controlDiv);
                    function rotate() {{
                        if (isRotating) {{
                            angle += 0.5;
                            let positions = network.getPositions();
                            let newPositions = {{}};
                            for (let nodeId in positions) {{
                                let pos = positions[nodeId];
                                let distance = Math.sqrt(pos.x * pos.x + pos.y * pos.y);
                                let currentAngle = Math.atan2(pos.y, pos.x);
                                let newAngle = currentAngle + (Math.PI / 180);
                                newPositions[nodeId] = {{
                                    x: distance * Math.cos(newAngle),
                                    y: distance * Math.sin(newAngle)
                                }};
                            }}
                            network.moveNode = function() {{}};
                            for (let nodeId in newPositions) {{
                                network.moveNode(nodeId, newPositions[nodeId].x, newPositions[nodeId].y);
                            }}
                        }}
                        setTimeout(rotate, rotationSpeed * 1000);
                    }}
                    rotate();
                }}, 1000);
            }})();
        </script>
        """
        full_html = net_html.replace("</body>", rotation_script + "</body>")
        legend_html = f"""
        <div style="position: absolute; bottom: 20px; left: 20px; background: white; 
        padding: 16px 20px; border-radius: 15px; box-shadow: 0 4px 12px rgba(0,0,0,0.15); z-index: 1000; 
        font-family: Inter, sans-serif; border: 2px solid {colors['primary']};">
            <b style="font-size: 16px; color: {colors['primary']};">Legend</b><br>
            <div style="margin-top: 10px;">
                <span style="display: inline-block; width: 16px; height: 16px; background: {colors['primary']}; 
                border-radius: 50%; margin-right: 8px; vertical-align: middle;"></span>
                <span style="color: #374151;">Symptoms</span>
            </div>
            <div style="margin-top: 8px;">
                <span style="display: inline-block; width: 16px; height: 16px; background: {colors['secondary']}; 
                border-radius: 50%; margin-right: 8px; vertical-align: middle;"></span>
                <span style="color: #374151;">Diseases</span>
            </div>
            <div style="margin-top: 8px;">
                <span style="color: #6b7280; font-size: 13px;">💡 Node size = Frequency<br>
                🔗 Edge thickness = Association strength</span>
            </div>
        </div>
        """
        full_html = full_html.replace("</body>", legend_html + "</body>")
        st.components.v1.html(full_html, height=850, scrolling=False)
        st.markdown(f"<h4 style='color: {colors['primary']}; margin-top: 2em;'>📊 Network Analytics</h4>", 
                   unsafe_allow_html=True)
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.markdown(f"""
                <div style='background: linear-gradient(135deg, {colors['primary']}20, {colors['primary']}40); 
                padding: 1.5em; border-radius: 12px; text-align: center;'>
                <h3 style='color: {colors['primary']}; margin: 0;'>{G.number_of_nodes()}</h3>
                <p style='margin: 0.5em 0 0 0; color: #374151; font-size: 14px;'>Total Nodes</p>
                </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
                <div style='background: linear-gradient(135deg, {colors['secondary']}20, {colors['secondary']}40); 
                padding: 1.5em; border-radius: 12px; text-align: center;'>
                <h3 style='color: {colors['secondary']}; margin: 0;'>{G.number_of_edges()}</h3>
                <p style='margin: 0.5em 0 0 0; color: #374151; font-size: 14px;'>Connections</p>
                </div>
            """, unsafe_allow_html=True)
        with col3:
            avg_degree = sum(dict(G.degree()).values()) / G.number_of_nodes()
            st.markdown(f"""
                <div style='background: linear-gradient(135deg, {colors['accent']}20, {colors['accent']}40); 
                padding: 1.5em; border-radius: 12px; text-align: center;'>
                <h3 style='color: {colors['accent']}; margin: 0;'>{avg_degree:.1f}</h3>
                <p style='margin: 0.5em 0 0 0; color: #374151; font-size: 14px;'>Avg Degree</p>
                </div>
            """, unsafe_allow_html=True)
        with col4:
            try:
                density = nx.density(G)
                st.markdown(f"""
                    <div style='background: linear-gradient(135deg, {colors['success']}20, {colors['success']}40); 
                    padding: 1.5em; border-radius: 12px; text-align: center;'>
                    <h3 style='color: {colors['success']}; margin: 0;'>{density:.3f}</h3>
                    <p style='margin: 0.5em 0 0 0; color: #374151; font-size: 14px;'>Density</p>
                    </div>
                """, unsafe_allow_html=True)
            except:
                st.markdown(f"""
                    <div style='background: linear-gradient(135deg, {colors['success']}20, {colors['success']}40); 
                    padding: 1.5em; border-radius: 12px; text-align: center;'>
                    <h3 style='color: {colors['success']}; margin: 0;'>N/A</h3>
                    <p style='margin: 0.5em 0 0 0; color: #374151; font-size: 14px;'>Density</p>
                    </div>
                """, unsafe_allow_html=True)
        with col5:
            try:
                diameter = nx.diameter(G) if nx.is_connected(G) else "N/A"
                st.markdown(f"""
                    <div style='background: linear-gradient(135deg, {colors['danger']}20, {colors['danger']}40); 
                    padding: 1.5em; border-radius: 12px; text-align: center;'>
                    <h3 style='color: {colors['danger']}; margin: 0;'>{diameter}</h3>
                    <p style='margin: 0.5em 0 0 0; color: #374151; font-size: 14px;'>Diameter</p>
                    </div>
                """, unsafe_allow_html=True)
            except:
                st.markdown(f"""
                    <div style='background: linear-gradient(135deg, {colors['danger']}20, {colors['danger']}40); 
                    padding: 1.5em; border-radius: 12px; text-align: center;'>
                    <h3 style='color: {colors['danger']}; margin: 0;'>N/A</h3>
                    <p style='margin: 0.5em 0 0 0; color: #374151; font-size: 14px;'>Diameter</p>
                    </div>
                """, unsafe_allow_html=True)
    # ========== 3. 3D PCA DISEASE SPACE WITH ANIMATION ==========
    if training_df is not None and not training_df.empty:
        st.markdown(f"""
            <h2 style='color: {colors['primary']}; font-size: 2em; margin-bottom: 0.5em;'>
            🌌 3D Disease Symptom Space
            </h2>
        """, unsafe_allow_html=True)
        if show_theory:
            st.markdown(f"""
                <div style='background: linear-gradient(135deg, {colors['gradient1']}15 0%, {colors['gradient2']}15 100%); 
                padding: 1.5em; border-radius: 15px; margin-bottom: 1.5em; border-left: 5px solid {colors['primary']};'>
                <h4 style='color: {colors['primary']};'>🔬 Principal Component Analysis Theory</h4>
                <p><b>Mathematical Foundation:</b> PCA reduces high-dimensional symptom data to 3 principal components 
                through eigenvalue decomposition of the covariance matrix:</p>
                <p style='text-align: center; background: #f0f9ff; color: #1e3a8a; padding: 0.5em 1em; border-radius: 8px; font-family: monospace; font-size: 1.1em;'>
                Cov(X) = V × Λ × V^T</p>
                <ul>
                    <li><b>V:</b> Eigenvectors (principal component directions)</li>
                    <li><b>Λ:</b> Eigenvalues (variance explained by each component)</li>
                    <li><b>Transformation:</b> Z = X × V (project data onto new axes)</li>
                </ul>
                
                <p><b>💡 Interpretation:</b></p>
                <ul>
                    <li><b>Proximity:</b> Diseases close together have similar symptom profiles</li>
                    <li><b>Clusters:</b> Natural disease families (e.g., respiratory, cardiovascular)</li>
                    <li><b>Outliers:</b> Unique diseases with distinctive symptoms</li>
                    <li><b>Variance Explained:</b> How much information each axis captures</li>
                </ul>
                </div>
            """, unsafe_allow_html=True)
        
        # Prepare data for PCA
        disease_symptom_matrix = training_df.groupby('prognosis')[symptom_cols].mean()
        
        # Apply PCA
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(disease_symptom_matrix)
        pca = PCA(n_components=3)
        pca_result = pca.fit_transform(scaled_data)
        
        # Create DataFrame
        pca_df = pd.DataFrame(data=pca_result, columns=['PC1', 'PC2', 'PC3'])
        pca_df['Disease'] = disease_symptom_matrix.index
        pca_df['Frequency'] = training_df['prognosis'].value_counts()[pca_df['Disease']].values
        
        # Categorize diseases
        def categorize_disease(disease_name):
            disease_lower = disease_name.lower()
            if any(x in disease_lower for x in ['infection', 'fever', 'flu', 'cold', 'pneumonia', 'tuberculosis']):
                return 'Infectious'
            elif any(x in disease_lower for x in ['diabetes', 'hypertension', 'heart', 'cardiac', 'blood pressure']):
                return 'Chronic/Metabolic'
            elif any(x in disease_lower for x in ['allergy', 'asthma', 'bronchial', 'respiratory']):
                return 'Respiratory'
            elif any(x in disease_lower for x in ['gastro', 'ulcer', 'hepatitis', 'jaundice', 'stomach']):
                return 'Digestive'
            elif any(x in disease_lower for x in ['arthritis', 'joint', 'muscle', 'cervical']):
                return 'Musculoskeletal'
            else:
                return 'Other'
        
        pca_df['Category'] = pca_df['Disease'].apply(categorize_disease)
        
        # Create 3D scatter with animation
        fig_3d = px.scatter_3d(
            pca_df, 
            x='PC1', 
            y='PC2', 
            z='PC3',
            color='Category',
            size='Frequency',
            hover_name='Disease',
            hover_data={'PC1': ':.3f', 'PC2': ':.3f', 'PC3': ':.3f', 'Frequency': True, 'Category': True},
            color_discrete_sequence=px.colors.qualitative.Bold,
            size_max=30,
            opacity=0.85
        )
        
        fig_3d.update_traces(
            marker=dict(
                line=dict(width=2, color='white'),
                symbol='circle'
            ),
            textposition='top center',
            hovertemplate='<b>%{hovertext}</b><br><br>' +
                         'PC1: %{x:.3f}<br>' +
                         'PC2: %{y:.3f}<br>' +
                         'PC3: %{z:.3f}<br>' +
                         'Category: %{customdata[4]}<br>' +
                         'Frequency: %{customdata[3]}<br>' +
                         '<extra></extra>'
        )
        
        fig_3d.update_layout(
            title=dict(
                text=f'🌌 3D Disease Symptom Space | Total Variance Explained: {sum(pca.explained_variance_ratio_)*100:.1f}%',
                font=dict(size=20, family="Inter, sans-serif", color=colors['primary']),
                x=0.5,
                xanchor='center'
            ),
            scene=dict(
                xaxis=dict(
                    title=dict(
                        text=f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)',
                        font=dict(size=14, color=colors['primary'])
                    ),
                    backgroundcolor="rgba(245,245,245,0.5)",
                    gridcolor="lightgray",
                    showbackground=True
                ),
                yaxis=dict(
                    title=dict(
                        text=f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)',
                        font=dict(size=14, color=colors['primary'])
                    ),
                    backgroundcolor="rgba(245,245,245,0.5)",
                    gridcolor="lightgray",
                    showbackground=True
                ),
                zaxis=dict(
                    title=dict(
                        text=f'PC3 ({pca.explained_variance_ratio_[2]*100:.1f}%)',
                        font=dict(size=14, color=colors['primary'])
                    ),
                    backgroundcolor="rgba(245,245,245,0.5)",
                    gridcolor="lightgray",
                    showbackground=True
                ),
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.3)
                )
            ),
            height=800,
            showlegend=True,
            legend=dict(
                x=0.02, 
                y=0.98,
                bgcolor='rgba(255,255,255,0.95)',
                bordercolor=colors['primary'],
                borderwidth=2,
                font=dict(size=13)
            ),
            paper_bgcolor='white',
            font=dict(family="Inter, sans-serif"),
            margin=dict(l=0, r=0, t=80, b=0)
        )
        
        st.plotly_chart(fig_3d, use_container_width=True, key="pca_3d_chart")
        
        # PCA Component Analysis
        st.markdown(f"<h4 style='color: {colors['primary']};'>📊 Component Analysis</h4>", 
                   unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Variance explained
            variance_df = pd.DataFrame({
                'Component': [f'PC{i+1}' for i in range(3)],
                'Variance Explained (%)': pca.explained_variance_ratio_ * 100,
                'Cumulative Variance (%)': np.cumsum(pca.explained_variance_ratio_) * 100
            })
            
            fig_var = go.Figure()
            fig_var.add_trace(go.Bar(
                x=variance_df['Component'],
                y=variance_df['Variance Explained (%)'],
                name='Individual',
                marker_color=colors['primary'],
                text=variance_df['Variance Explained (%)'].round(2),
                texttemplate='%{text}%',
                textposition='outside'
            ))
            fig_var.add_trace(go.Scatter(
                x=variance_df['Component'],
                y=variance_df['Cumulative Variance (%)'],
                name='Cumulative',
                mode='lines+markers',
                line=dict(color=colors['danger'], width=3),
                marker=dict(size=10),
                yaxis='y2'
            ))
            
            fig_var.update_layout(
                title='Variance Explained by Components',
                xaxis_title='Principal Component',
                yaxis_title='Individual Variance (%)',
                yaxis2=dict(
                    title='Cumulative Variance (%)',
                    overlaying='y',
                    side='right',
                    range=[0, 110]
                ),
                paper_bgcolor='white',
                plot_bgcolor='white',
                height=400,
                font=dict(family="Inter, sans-serif", size=12)
            )
            
            st.plotly_chart(fig_var, use_container_width=True, key="variance_chart")
        
        with col2:
            # Disease category distribution
            category_counts = pca_df['Category'].value_counts()
            
            fig_cat = go.Figure(data=[go.Pie(
                labels=category_counts.index,
                values=category_counts.values,
                hole=0.4,
                marker=dict(
                    colors=px.colors.qualitative.Bold,
                    line=dict(color='white', width=2)
                ),
                textinfo='label+percent',
                textfont=dict(size=13),
                hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
            )])
            
            fig_cat.update_layout(
                title='Disease Category Distribution',
                paper_bgcolor='white',
                height=400,
                font=dict(family="Inter, sans-serif", size=12),
                annotations=[dict(
                    text=f'{len(pca_df)}<br>Diseases',
                    x=0.5, y=0.5,
                    font_size=18,
                    showarrow=False
                )]
            )
            
            st.plotly_chart(fig_cat, use_container_width=True, key="category_chart")
    # ========== 4. HIERARCHICAL CLUSTERING DENDROGRAM ==========

    # ========== 4. HIERARCHICAL CLUSTERING DENDROGRAM ==========
    # --- Updated section: robust handling for NaNs/Infs, proper condensed matrix, and theory/formatting preserved ---
    if training_df is not None and not training_df.empty:
        st.markdown(f"""
            <h2 style='color: {colors['primary']}; font-size: 2em; margin-bottom: 0.5em;'>
            🌳 Symptom Hierarchy Dendrogram
            </h2>
        """, unsafe_allow_html=True)
        if show_theory:
            st.markdown(f"""
                <div style='background: linear-gradient(135deg, {colors['gradient1']}15 0%, {colors['gradient2']}15 100%); 
                padding: 1.5em; border-radius: 15px; margin-bottom: 1.5em; border-left: 5px solid {colors['primary']};'>
                <h4 style='color: {colors['primary']};'>🔬 Hierarchical Clustering Theory</h4>
                <p><b>Algorithm:</b> Agglomerative clustering using Ward's linkage criterion, which minimizes 
                the within-cluster variance at each merge:</p>
                <p style='text-align: center; background: #f0f9ff; color: #1e3a8a; padding: 0.5em 1em; border-radius: 8px; font-family: monospace; font-size: 1.1em;'>
                d(C_i, C_j) = Δ(SSE) = SSE(C_i ∪ C_j) - SSE(C_i) - SSE(C_j)</p>
                <p><b>💡 Reading the Dendrogram:</b></p>
                <ul>
                    <li><b>Vertical Lines:</b> Individual symptoms or clusters</li>
                    <li><b>Horizontal Lines:</b> Merge points (height = dissimilarity)</li>
                    <li><b>Lower Merges:</b> More similar symptoms (frequently co-occur)</li>
                    <li><b>Higher Merges:</b> Less related symptoms</li>
                    <li><b>Color Coding:</b> Identifies distinct symptom families</li>
                </ul>
                </div>
            """, unsafe_allow_html=True)
        # Select top symptoms, calculate correlation matrix
        from scipy.spatial.distance import squareform
        top_symptoms = training_df[symptom_cols].sum().sort_values(ascending=False).head(20).index.tolist()
        corr_matrix = training_df[top_symptoms].corr()
        # Handle NaNs/Infs robustly
        corr_matrix = corr_matrix.fillna(0)
        distance_matrix = 1 - corr_matrix
        distance_matrix = np.nan_to_num(distance_matrix, nan=0.0, posinf=1.0, neginf=1.0)
        # Convert to condensed distance matrix for linkage
        condensed_dist = squareform(np.array(distance_matrix, dtype=np.float64), checks=False)
        # Perform hierarchical clustering (Ward's method)
        linkage_matrix = linkage(condensed_dist, method='ward')
        # Plot dendrogram
        fig_dend, ax = plt.subplots(figsize=(18, 12))
        ax.set_facecolor('white')
        fig_dend.patch.set_facecolor('white')
        dendro = dendrogram(
            linkage_matrix,
            labels=[s.replace('_', ' ').title() for s in top_symptoms],
            ax=ax,
            orientation='right',
            color_threshold=0.7*max(linkage_matrix[:,2]),
            above_threshold_color='#9ca3af',
            leaf_font_size=15
        )
        ax.set_title('🌳 Symptom Hierarchy Dendrogram', 
                    fontsize=26, fontweight='bold', color=colors['primary'], pad=25)
        ax.set_xlabel('Distance (Dissimilarity)', fontsize=18, fontweight='bold', color='#374151')
        ax.set_ylabel('Symptoms', fontsize=18, fontweight='bold', color='#374151')
        ax.tick_params(labelsize=14, colors='#374151')
        ax.grid(axis='x', alpha=0.3, linestyle='--', linewidth=1)
        ax.axvline(x=0.7*max(linkage_matrix[:,2]), color=colors['danger'], linestyle='--', 
                  linewidth=2, label='Clustering Threshold', alpha=0.7)
        ax.legend(fontsize=14)
        for spine in ax.spines.values():
            spine.set_edgecolor(colors['primary'])
            spine.set_linewidth(2)
        plt.tight_layout()
        st.pyplot(fig_dend)
    # ========== 5. ANIMATED CONFIDENCE TIMELINE ==========

    # ========== 5. ANIMATED CONFIDENCE TIMELINE ==========
    if history and len(history) > 0:
        st.markdown(f"""
            <h2 style='color: {colors['primary']}; font-size: 2em; margin-bottom: 0.5em;'>
            ⏱️ Confidence Trend Over Time
            </h2>
        """, unsafe_allow_html=True)
        
        if show_theory:
            st.markdown(f"""
                <div style='background: linear-gradient(135deg, {colors['gradient1']}15 0%, {colors['gradient2']}15 100%); 
                padding: 1.5em; border-radius: 15px; margin-bottom: 1.5em; border-left: 5px solid {colors['primary']};'>
                <h4 style='color: {colors['primary']};'>🔬 Time Series Analysis & Quality Metrics</h4>
                <p><b>Statistical Measures:</b></p>
                <ul>
                    <li><b>Confidence Score:</b> Probability assigned by ML model (0-100%)</li>
                    <li><b>Moving Average:</b> Smoothed trend using rolling window</li>
                    <li><b>Quality Zones:</b> High (≥80%), Medium (60-80%), Low (<60%)</li>
                    <li><b>Outlier Detection:</b> Points deviating significantly from trend</li>
                </ul>
                </div>
            """, unsafe_allow_html=True)
        
        hist_df = pd.DataFrame(history)
        hist_df['ts'] = pd.to_datetime(hist_df['timestamp'])
        hist_df = hist_df.sort_values('ts')
        hist_df['confidence_pct'] = hist_df['confidence'] * 100
        hist_df['ma_5'] = hist_df['confidence_pct'].rolling(window=min(5, len(hist_df)), min_periods=1).mean()
        
        # Create animated timeline
        fig_timeline = go.Figure()
        
        # Confidence zones
        fig_timeline.add_hrect(y0=80, y1=100, fillcolor=colors['success'], opacity=0.15,
                              annotation_text="High Confidence Zone", annotation_position="top right",
                              annotation=dict(font_size=12))
        fig_timeline.add_hrect(y0=60, y1=80, fillcolor=colors['accent'], opacity=0.15,
                              annotation_text="Medium Confidence Zone", annotation_position="top right",
                              annotation=dict(font_size=12))
        fig_timeline.add_hrect(y0=0, y1=60, fillcolor=colors['danger'], opacity=0.15,
                              annotation_text="Low Confidence Zone", annotation_position="top right",
                              annotation=dict(font_size=12))
        
        # Main confidence line
        fig_timeline.add_trace(go.Scatter(
            x=hist_df['ts'],
            y=hist_df['confidence_pct'],
            mode='lines+markers',
            name='Confidence Score',
            line=dict(color=colors['primary'], width=3),
            marker=dict(
                size=10,
                color=hist_df['confidence_pct'],
                colorscale='RdYlGn',
                showscale=True,
                colorbar=dict(title="Score (%)", x=1.15),
                line=dict(width=2, color='white'),
                symbol='circle'
            ),
            hovertemplate='<b>Time:</b> %{x|%Y-%m-%d %H:%M:%S}<br>' +
                         '<b>Confidence:</b> %{y:.2f}%<br>' +
                         '<extra></extra>'
        ))
        
        # Moving average
        fig_timeline.add_trace(go.Scatter(
            x=hist_df['ts'],
            y=hist_df['ma_5'],
            mode='lines',
            name='Moving Average (5-point)',
            line=dict(color=colors['secondary'], width=2, dash='dash'),
            hovertemplate='<b>MA:</b> %{y:.2f}%<extra></extra>'
        ))
        
        # Highlight extremes
        max_idx = hist_df['confidence_pct'].idxmax()
        min_idx = hist_df['confidence_pct'].idxmin()
        
        fig_timeline.add_trace(go.Scatter(
            x=[hist_df.loc[max_idx, 'ts']],
            y=[hist_df.loc[max_idx, 'confidence_pct']],
            mode='markers+text',
            name='Peak',
            marker=dict(color='gold', size=20, symbol='star', line=dict(width=2, color='darkgoldenrod')),
            text=['🏆 Peak'],
            textposition='top center',
            textfont=dict(size=14, color='darkgoldenrod'),
            hovertemplate='<b>Peak Confidence</b><br>%{y:.2f}%<extra></extra>'
        ))
        
        fig_timeline.add_trace(go.Scatter(
            x=[hist_df.loc[min_idx, 'ts']],
            y=[hist_df.loc[min_idx, 'confidence_pct']],
            mode='markers+text',
            name='Lowest',
            marker=dict(color=colors['danger'], size=20, symbol='x', line=dict(width=2, color='darkred')),
            text=['⚠️ Low'],
            textposition='bottom center',
            textfont=dict(size=14, color='darkred'),
            hovertemplate='<b>Lowest Confidence</b><br>%{y:.2f}%<extra></extra>'
        ))
        
        fig_timeline.update_layout(
            title=dict(
                text='⏱️ Confidence Trend Over Time',
                font=dict(size=20, family="Inter, sans-serif", color=colors['primary']),
                x=0.5,
                xanchor='center'
            ),
            xaxis_title='Timeline',
            yaxis_title='Confidence Score (%)',
            hovermode='x unified',
            height=600,
            paper_bgcolor='white',
            plot_bgcolor='white',
            font=dict(family="Inter, sans-serif", size=14),
            showlegend=True,
            legend=dict(
                x=0.02, 
                y=0.98, 
                bgcolor='rgba(255,255,255,0.95)',
                bordercolor=colors['primary'],
                borderwidth=2
            ),
            yaxis=dict(range=[0, 105]),
            xaxis=dict(showgrid=True, gridcolor='lightgray'),
            yaxis2=dict(showgrid=True, gridcolor='lightgray')
        )
        
        st.plotly_chart(fig_timeline, use_container_width=True, key="confidence_timeline_chart")
        
        # Statistics
        col1, col2, col3, col4, col5 = st.columns(5)
        
        avg_conf = hist_df['confidence_pct'].mean()
        std_conf = hist_df['confidence_pct'].std()
        high_conf_pct = (hist_df['confidence_pct'] >= 80).sum() / len(hist_df) * 100
        
        with col1:
            st.markdown(f"""
                <div style='background: linear-gradient(135deg, {colors['primary']}20, {colors['primary']}40); 
                padding: 1.5em; border-radius: 12px; text-align: center;'>
                <h3 style='color: {colors['primary']}; margin: 0;'>{avg_conf:.1f}%</h3>
                <p style='margin: 0.5em 0 0 0; color: #374151; font-size: 14px;'>Mean Confidence</p>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
                <div style='background: linear-gradient(135deg, {colors['secondary']}20, {colors['secondary']}40); 
                padding: 1.5em; border-radius: 12px; text-align: center;'>
                <h3 style='color: {colors['secondary']}; margin: 0;'>{hist_df['confidence_pct'].median():.1f}%</h3>
                <p style='margin: 0.5em 0 0 0; color: #374151; font-size: 14px;'>Median</p>
                </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
                <div style='background: linear-gradient(135deg, {colors['accent']}20, {colors['accent']}40); 
                padding: 1.5em; border-radius: 12px; text-align: center;'>
                <h3 style='color: {colors['accent']}; margin: 0;'>{std_conf:.1f}%</h3>
                <p style='margin: 0.5em 0 0 0; color: #374151; font-size: 14px;'>Std Dev</p>
                </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
                <div style='background: linear-gradient(135deg, {colors['success']}20, {colors['success']}40); 
                padding: 1.5em; border-radius: 12px; text-align: center;'>
                <h3 style='color: {colors['success']}; margin: 0;'>{high_conf_pct:.1f}%</h3>
                <p style='margin: 0.5em 0 0 0; color: #374151; font-size: 14px;'>High Conf Rate</p>
                </div>
            """, unsafe_allow_html=True)
        
        with col5:
            st.markdown(f"""
                <div style='background: linear-gradient(135deg, {colors['danger']}20, {colors['danger']}40); 
                padding: 1.5em; border-radius: 12px; text-align: center;'>
                <h3 style='color: {colors['danger']}; margin: 0;'>{len(hist_df)}</h3>
                <p style='margin: 0.5em 0 0 0; color: #374151; font-size: 14px;'>Total Predictions</p>
                </div>
            """, unsafe_allow_html=True)
    # ========== 6. DISEASE PREDICTION COUNTS BY USER (SUNBURST CHART) ==========
    # Properly integrated Sunburst chart with DB_PATH and table existence check, wrapped in gradient/theory block.
    if DB_PATH:
        st.markdown(f"""
            <h2 style='color: {colors['primary']}; font-size: 2em; margin-bottom: 0.5em;'>
            🌐 User-wise Disease Prediction Distribution
            </h2>
        """, unsafe_allow_html=True)
        if show_theory:
            st.markdown(f"""
                <div style='background: linear-gradient(135deg, {colors['gradient1']}15 0%, {colors['gradient2']}15 100%);
                padding: 1.5em; border-radius: 15px; margin-bottom: 1.5em; border-left: 5px solid {colors['primary']};'>
                <h4 style='color: {colors['primary']};'>🔬 Sunburst Chart Theory & Clinical Insight</h4>
                <p><b>Mathematical Foundation:</b> Sunburst charts represent hierarchical relationships. Here, we model:
                <ul>
                    <li><b>Root:</b> All users</li>
                    <li><b>Level 1:</b> Individual users</li>
                    <li><b>Level 2:</b> Predicted diseases</li>
                </ul>
                Each sector's size represents the number of predictions.</p>
                <p><b>📊 Interpretation Guide:</b></p>
                <ul>
                    <li>Wider sector = More predictions for that disease/user</li>
                    <li>Hierarchy helps track which diseases are frequent per user</li>
                    <li>Color coding emphasizes distribution across categories</li>
                </ul>
                <p><b>💡 Clinical Applications:</b></p>
                <ul>
                    <li>Monitor individual user trends</li>
                    <li>Identify common diseases among all users</li>
                    <li>Assess model output distribution for fairness</li>
                </ul>
                </div>
            """, unsafe_allow_html=True)
        # Connect to SQLite and check for predictions table existence
        conn = sqlite3.connect(DB_PATH)
        try:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    user_id TEXT,
                    prediction TEXT,
                    confidence REAL,
                    timestamp TEXT,
                    prognosis TEXT
                )
            """)
            conn.commit()
            # Only proceed if table exists and has data
            user_df = pd.read_sql_query("SELECT user_id, prognosis FROM predictions", conn)
        except Exception:
            user_df = pd.DataFrame(columns=['user_id', 'prognosis'])
        finally:
            conn.close()
        if not user_df.empty:
            sunburst_df = user_df.groupby(['user_id', 'prognosis']).size().reset_index(name='count')
            fig_sunburst = px.sunburst(
                sunburst_df,
                path=['user_id', 'prognosis'],
                values='count',
                color='count',
                color_continuous_scale=px.colors.sequential.Viridis,
                hover_data={'user_id': True, 'prognosis': True, 'count': True}
            )
            fig_sunburst.update_layout(
                title='🌐 User-wise Disease Prediction Distribution',
                paper_bgcolor='white',
                plot_bgcolor='white',
                font=dict(family="Inter, sans-serif", size=14),
                height=600
            )
            st.plotly_chart(fig_sunburst, use_container_width=True, key="sunburst_chart")
    # ========== 6. TOP 10 PREDICTED DISEASES BAR CHART ==========
    # ========== 6. TOP 10 PREDICTED DISEASES BAR CHART ==========
    if history and len(history) > 0:
        st.markdown(f"""
            <h2 style='color: {colors['primary']}; font-size: 2em; margin-bottom: 0.5em;'>
            🏆 Most Frequently Predicted Diseases
            </h2>
        """, unsafe_allow_html=True)
        if show_theory:
            st.markdown(f"""
                <div style='background: linear-gradient(135deg, {colors['gradient1']}15 0%, {colors['gradient2']}15 100%); 
                padding: 1.5em; border-radius: 15px; margin-bottom: 1.5em; border-left: 5px solid {colors['primary']};'>
                <h4 style='color: {colors['primary']};'>🔬 Theoretical Basis & Clinical Utility</h4>
                <p>
                <b>Statistical Foundation:</b><br>
                This bar chart aggregates the frequency of each disease predicted across all user interactions or model runs.<br>
                Let D = set of diseases; the count for disease d ∈ D is:<br>
                <span style='background: #f0f9ff; color: #1e3a8a; padding: 0.2em 0.5em; border-radius: 6px; font-family: monospace; font-size: 1.1em;'>
                count(d) = Σ<sub>i=1</sub><sup>N</sup> 1[prediction<sub>i</sub> = d]
                </span><br>
                where 1[·] is the indicator function and N is the number of predictions.
                </p>
                <p>
                <b>Interpretation:</b>
                <ul>
                    <li>Bar height = how frequently the model predicts each disease</li>
                    <li>Highlights most common model outputs, possible bias, or population trends</li>
                    <li>Useful for tracking disease prevalence in the dataset or monitoring model drift</li>
                </ul>
                </p>
                <p>
                <b>Clinical Application:</b>
                <ul>
                    <li><b>Resource Allocation:</b> Focus on frequently predicted diseases for screening or education</li>
                    <li><b>Bias Detection:</b> Spot over/under-represented conditions</li>
                    <li><b>Feedback Loop:</b> Identify targets for model improvement or further data collection</li>
                </ul>
                </p>
                </div>
            """, unsafe_allow_html=True)
        # Prepare data
        hist_df = pd.DataFrame(history)
        pred_counts = hist_df['prediction'].value_counts().sort_values(ascending=False)[:10]
        pred_names = [str(d) for d in pred_counts.index]
        pred_freqs = pred_counts.values
        # Plotly bar chart
        fig_pred = go.Figure()
        fig_pred.add_trace(go.Bar(
            x=pred_names,
            y=pred_freqs,
            marker_color=colors['primary'],
            text=pred_freqs,
            textposition='auto',
            hovertemplate='<b>%{x}</b><br>Predicted: %{y} times<extra></extra>'
        ))
        fig_pred.update_layout(
            title='🏆 Most Frequently Predicted Diseases',
            xaxis_title='Disease',
            yaxis_title='Prediction Count',
            paper_bgcolor='white',
            plot_bgcolor='white',
            font=dict(family="Inter, sans-serif", size=14),
            height=450,
            margin=dict(l=40, r=40, t=60, b=80),
        )
        st.plotly_chart(fig_pred, use_container_width=True, key="top_predicted_diseases_chart")




# ========== EXTRA SECTION: TOP SYMPTOMS FREQUENCY & CONFIDENCE DISTRIBUTION ==========
# This section is appended and does not interfere with existing functions.
# It is intended for demonstration or expansion purposes.

def render_extra_visualizations(training_df, history):
    """
    Extra visualizations: Top Symptoms Frequency and Confidence Distribution Across All Models.
    Intended as a separate section, does not interfere with main dashboard.
    """
    import streamlit as st
    import plotly.graph_objects as go
    import pandas as pd
    import numpy as np
    import plotly.express as px

    st.markdown("---")
    st.header("🌟 Symptom Occurrence Leaderboard (Standalone)")
    if training_df is not None and not training_df.empty:
        symptom_cols = training_df.columns[:-1]
        symptom_counts = training_df[symptom_cols].sum().sort_values(ascending=False)[:15]
        fig_symptoms = go.Figure()
        fig_symptoms.add_trace(go.Bar(
            x=[s.replace('_', ' ').title() for s in symptom_counts.index],
            y=symptom_counts.values,
            marker_color="#3b82f6",
            text=symptom_counts.values,
            textposition='auto',
            hovertemplate='<b>%{x}</b><br>Frequency: %{y}<extra></extra>'
        ))
        fig_symptoms.update_layout(
            title='Top 15 Symptoms by Occurrence',
            xaxis_title='Symptoms',
            yaxis_title='Frequency',
            paper_bgcolor='white',
            plot_bgcolor='white',
            font=dict(family="Inter, sans-serif", size=14),
            height=450
        )
        st.plotly_chart(fig_symptoms, use_container_width=True)
    else:
        st.info("No training data available for symptom frequency visualization.")

    st.markdown("---")
    st.header("📈 Confidence Distribution Across All Models (Standalone)")
    if history and len(history) > 0:
        hist_df = pd.DataFrame(history)
        if 'confidence' in hist_df.columns:
            hist_df['confidence_pct'] = hist_df['confidence'] * 100
            fig_conf = px.histogram(
                hist_df, x='confidence_pct', nbins=20,
                color_discrete_sequence=['#06b6d4'],
                labels={'confidence_pct': 'Confidence (%)'},
                title='Distribution of Prediction Confidence Scores'
            )
            fig_conf.update_layout(
                paper_bgcolor='white',
                plot_bgcolor='white',
                font=dict(family="Inter, sans-serif", size=14),
                height=400,
                xaxis_title='Confidence (%)',
                yaxis_title='Frequency'
            )
            st.plotly_chart(fig_conf, use_container_width=True)
        else:
            st.info("No confidence scores found in history.")
    else:
        st.info("No prediction history available for confidence distribution.")