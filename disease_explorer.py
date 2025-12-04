def clean_dict(data_dict):
    """
    Remove NULL / None / empty values from dictionary lists
    """
    cleaned = {}
    for key, values in data_dict.items():
        if isinstance(values, list):
            cleaned[key] = [v for v in values if v not in (None, "NULL", "", "None")]
        else:
            cleaned[key] = values
    return cleaned
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import json

def render_disease_explorer(description, precautions, medications, diets, workout):
    """
    Premium Disease Explorer & Intelligent Comparison System
    """
    
    # ========== PREMIUM CSS STYLING ==========

    # Manually define disease_list for use throughout the function
    disease_list = sorted(description.keys())
    # ✅ GLOBAL DATA CLEANING (removes NULL everywhere once)
    precautions = clean_dict(precautions)
    medications = clean_dict(medications)
    diets = clean_dict(diets)
    workout = clean_dict(workout)
    st.markdown("""
        <style>
        /* Global Styles */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
        
        * {
            font-family: 'Inter', sans-serif;
        }
        
        /* Hero Header */
        .hero-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 3.5em 2em;
            border-radius: 30px;
            text-align: center;
            margin-bottom: 2.5em;
            box-shadow: 0 20px 60px rgba(102, 126, 234, 0.4);
            position: relative;
            overflow: hidden;
        }
        
        .hero-header::before {
            content: '';
            position: absolute;
            top: -50%;
            left: -50%;
            width: 200%;
            height: 200%;
            background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
            animation: pulse 4s ease-in-out infinite;
        }
        
        @keyframes pulse {
            0%, 100% { transform: scale(1); opacity: 0.5; }
            50% { transform: scale(1.1); opacity: 0.8; }
        }
        
        /* Disease Card */
        .disease-card-premium {
            background: white;
            border-radius: 25px;
            padding: 2.5em;
            margin: 1.5em 0;
            box-shadow: 0 10px 40px rgba(0,0,0,0.08);
            border: 1px solid rgba(102, 126, 234, 0.1);
            transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
            position: relative;
        }
        
        .disease-card-premium::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 6px;
            background: linear-gradient(90deg, #667eea, #764ba2, #f093fb);
            border-radius: 25px 25px 0 0;
        }
        
        .disease-card-premium:hover {
            transform: translateY(-8px);
            box-shadow: 0 20px 60px rgba(0,0,0,0.12);
        }
        
        /* Stats Grid */
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 1.5em;
            margin: 2em 0;
        }
        
        .stat-card {
            background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
            border-radius: 20px;
            padding: 2em 1.5em;
            text-align: center;
            box-shadow: 0 8px 30px rgba(0,0,0,0.06);
            border: 2px solid transparent;
            transition: all 0.3s ease;
            position: relative;
            overflow: hidden;
        }
        
        .stat-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: linear-gradient(135deg, var(--color-start), var(--color-end));
            opacity: 0;
            transition: opacity 0.3s ease;
            z-index: 0;
        }
        
        .stat-card:hover::before {
            opacity: 0.05;
        }
        
        .stat-card:hover {
            border-color: var(--color-start);
            transform: translateY(-5px);
            box-shadow: 0 15px 45px rgba(0,0,0,0.1);
        }
        
        .stat-card.blue { --color-start: #0ea5e9; --color-end: #0284c7; }
        .stat-card.green { --color-start: #10b981; --color-end: #059669; }
        .stat-card.orange { --color-start: #f59e0b; --color-end: #d97706; }
        .stat-card.pink { --color-start: #ec4899; --color-end: #db2777; }
        
        .stat-number {
            font-size: 3em;
            font-weight: 800;
            margin: 0;
            background: linear-gradient(135deg, var(--color-start), var(--color-end));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            position: relative;
            z-index: 1;
        }
        
        .stat-label {
            font-size: 0.95em;
            color: #64748b;
            font-weight: 600;
            margin-top: 0.5em;
            position: relative;
            z-index: 1;
        }
        
        .stat-icon {
            font-size: 2.5em;
            margin-bottom: 0.3em;
            opacity: 0.8;
        }
        
        /* Content Sections */
        .content-section {
            background: linear-gradient(135deg, #f8fafc 0%, #ffffff 100%);
            border-radius: 20px;
            padding: 2em;
            margin: 1.5em 0;
            border-left: 5px solid;
            box-shadow: 0 6px 25px rgba(0,0,0,0.04);
        }
        
        .section-header-premium {
            font-size: 1.8em;
            font-weight: 700;
            margin: 0 0 1.2em 0;
            display: flex;
            align-items: center;
            gap: 0.4em;
        }
        
        /* List Items */
        .premium-list-item {
            background: white;
            padding: 1.2em 1.5em;
            margin: 0.8em 0;
            border-radius: 15px;
            border-left: 4px solid;
            box-shadow: 0 4px 15px rgba(0,0,0,0.04);
            transition: all 0.3s ease;
            display: flex;
            align-items: center;
            gap: 1em;
        }
        
        .premium-list-item:hover {
            transform: translateX(8px);
            box-shadow: 0 8px 25px rgba(0,0,0,0.08);
        }
        
        .item-number {
            background: linear-gradient(135deg, var(--color-start), var(--color-end));
            color: white;
            width: 35px;
            height: 35px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 700;
            font-size: 0.9em;
            flex-shrink: 0;
        }
        
        /* Description Box */
        .description-premium {
            background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
            border-radius: 20px;
            padding: 2em;
            margin: 1.5em 0;
            border-left: 6px solid #0ea5e9;
            line-height: 1.8;
            font-size: 1.05em;
            color: #1e293b;
            box-shadow: 0 6px 25px rgba(14, 165, 233, 0.1);
        }
        
        /* Comparison Styles */
        .comparison-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
            padding: 3em 2em;
            border-radius: 30px;
            text-align: center;
            margin-bottom: 2.5em;
            box-shadow: 0 20px 60px rgba(102, 126, 234, 0.4);
        }
        
        .vs-divider {
            display: inline-block;
            background: rgba(255,255,255,0.3);
            padding: 0.5em 1.5em;
            border-radius: 50px;
            margin: 0 1em;
            font-weight: 600;
            font-size: 1.2em;
        }
        
        /* Common/Unique Badges */
        .badge-common {
            background: linear-gradient(135deg, #10b981, #059669);
            color: white;
            padding: 0.5em 1em;
            border-radius: 20px;
            font-size: 0.9em;
            font-weight: 600;
            display: inline-flex;
            align-items: center;
            gap: 0.4em;
            box-shadow: 0 4px 15px rgba(16, 185, 129, 0.3);
        }
        
        .badge-unique-1 {
            background: linear-gradient(135deg, #6366f1, #4f46e5);
            color: white;
            padding: 0.5em 1em;
            border-radius: 20px;
            font-size: 0.9em;
            font-weight: 600;
            display: inline-flex;
            align-items: center;
            gap: 0.4em;
            box-shadow: 0 4px 15px rgba(99, 102, 241, 0.3);
        }
        
        .badge-unique-2 {
            background: linear-gradient(135deg, #ec4899, #db2777);
            color: white;
            padding: 0.5em 1em;
            border-radius: 20px;
            font-size: 0.9em;
            font-weight: 600;
            display: inline-flex;
            align-items: center;
            gap: 0.4em;
            box-shadow: 0 4px 15px rgba(236, 72, 153, 0.3);
        }
        
        /* Similarity Score */
        .similarity-showcase {
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            padding: 3em 2em;
            border-radius: 25px;
            text-align: center;
            margin: 2em 0;
            box-shadow: 0 15px 50px rgba(102, 126, 234, 0.4);
            position: relative;
            overflow: hidden;
        }
        
        .similarity-showcase::before {
            content: '';
            position: absolute;
            top: -100px;
            right: -100px;
            width: 300px;
            height: 300px;
            background: radial-gradient(circle, rgba(255,255,255,0.1), transparent);
            border-radius: 50%;
        }
        
        .similarity-score {
            font-size: 5em;
            font-weight: 800;
            margin: 0.3em 0;
            text-shadow: 0 4px 20px rgba(0,0,0,0.2);
        }
        
        /* Alert Box */
        .alert-premium {
            background: linear-gradient(135deg, #fef3c7, #fde68a);
            border-left: 6px solid #f59e0b;
            border-radius: 20px;
            padding: 1.8em;
            margin: 1.5em 0;
            box-shadow: 0 8px 30px rgba(245, 158, 11, 0.15);
        }
        
        /* Footer */
        .footer-premium {
            background: linear-gradient(135deg, #1f2937, #111827);
            color: white;
            padding: 3em 2em;
            border-radius: 25px;
            text-align: center;
            margin-top: 4em;
        }
        
        /* Tabs Enhancement */
        .stTabs [data-baseweb="tab-list"] {
            gap: 10px;
            background: transparent;
        }
        
        .stTabs [data-baseweb="tab"] {
            background: #f1f5f9;
            border-radius: 15px 15px 0 0;
            padding: 1em 2em;
            font-weight: 600;
            border: none;
            transition: all 0.3s ease;
        }
        
        .stTabs [data-baseweb="tab"]:hover {
            background: #e2e8f0;
        }
        
        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white !important;
            box-shadow: 0 6px 20px rgba(102, 126, 234, 0.3);
        }
        
        /* Responsive Design */
        @media (max-width: 768px) {
            .stats-grid {
                grid-template-columns: repeat(2, 1fr);
            }
            .stat-number {
                font-size: 2.5em;
            }
        }

        /* ========= PERFECT PREMIUM MEDICAL UI THEME ========= */

        /* Global readable light text */
        h1, h2, h3, h4, h5,
        .hero-header p,
        .comparison-header div {
            color: #ecfeff !important;
        }

        /* Sidebar text */
        section[data-testid="stSidebar"] * {
            color: #ecfeff !important;
        }

        /* Tabs */
        .stTabs [data-baseweb="tab"] {
            color: #a5f3fc !important;
            font-weight: 600;
        }

        .stTabs [aria-selected="true"] {
            color: #ffffff !important;
        }

        /* Disease selectors */
        div[data-baseweb="select"] span {
            color: #ecfeff !important;
        }

        /* ===== Section Headers (Precautions / Meds / Diet / Workouts) ===== */
        .section-header-premium {
            font-size: 1.9em;
            font-weight: 700;
            letter-spacing: 0.3px;
            color: #67e8f9 !important;
            border-bottom: 1px solid rgba(103,232,249,0.35);
            padding-bottom: 0.4em;
            margin-bottom: 1.2em;
        }

        /* ===== Clean Card Lists ===== */
        .premium-list-item {
            background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
            padding: 1.4em 1.6em;
            margin: 0.9em 0;
            border-radius: 18px;
            border-left: 6px solid;
            display: flex;
            align-items: flex-start;
            gap: 1.1em;
            box-shadow: 0 8px 22px rgba(0,0,0,0.06);
            transition: all 0.25s ease;
        }

        .premium-list-item:hover {
            transform: translateY(-4px);
            box-shadow: 0 14px 30px rgba(0,0,0,0.1);
        }

        /* Number badge */
        .item-number {
            width: 40px;
            height: 40px;
            border-radius: 50%;
            font-size: 0.95em;
            font-weight: 800;
            color: white;
            display: flex;
            align-items: center;
            justify-content: center;
        }

        /* Text inside list */
        .premium-list-item div:last-child {
            color: #1e293b;
            font-size: 1.05em;
            line-height: 1.6;
        }

        /* ===== Tabs background harmony ===== */
        .stTabs [data-baseweb="tab-list"] {
            background: rgba(255,255,255,0.05);
            border-radius: 18px;
            padding: 10px;
        }
        /* ===== SIMPLE LIGHT BEAUTIFUL BOXES ===== */
        .light-box {
            background: #f8fafc;
            border-radius: 16px;
            padding: 14px 18px;
            margin: 10px 0;
            font-size: 16px;
            color: #1e293b;
            box-shadow: 0 6px 16px rgba(0,0,0,0.06);
            border-left: 5px solid var(--box-color);
            transition: all 0.25s ease;
        }

        .light-box:hover {
            background: #ffffff;
            transform: translateY(-2px);
            box-shadow: 0 10px 22px rgba(0,0,0,0.1);
        }
        /* 🔥 FINAL FIX — HIDE STREAMLIT NULL / DEBUG OUTPUT BLOCKS */
        div[data-testid="stJson"],
        div[data-testid="stObject"],
        div[data-testid="stTable"],
        div[data-testid="stDataFrame"] {
            display: none !important;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # ========== HELPER FUNCTIONS ==========
    
    def calculate_completeness(disease):
        """Calculate information richness (0-100)"""
        scores = []
        max_per_category = 5
        
        for data_dict in [precautions, medications, diets, workout]:
            items = data_dict.get(disease, [])
            score = min(len(items), max_per_category) / max_per_category * 100
            scores.append(score)
        
        return sum(scores) / len(scores)
    
    def get_overlap_analysis(list1, list2):
        """Analyze overlap between two lists"""
        set1, set2 = set(list1 or []), set(list2 or [])
        common = sorted(set1 & set2)
        unique1 = sorted(set1 - set2)
        unique2 = sorted(set2 - set1)
        return common, unique1, unique2
    
    def create_radar_chart(disease_name, data_counts):
        """Create professional radar chart with improved spacing, clarity, and interactivity"""
        categories = ['Precautions', 'Medications', 'Diet', 'Workouts']
        values = data_counts + data_counts[:1]  # Close the loop
        categories_loop = categories + categories[:1]
        # Prepare hover text for each category
        hover_text = [f"{cat}: {val}" for cat, val in zip(categories, data_counts)] + [f"{categories[0]}: {data_counts[0]}"]

        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories_loop,
            fill='toself',
            fillcolor='rgba(102, 126, 234, 0.18)',  # Slightly less opacity for fill
            line=dict(color='#667eea', width=2.5),  # Thinner line
            marker=dict(size=8, color='#764ba2', line=dict(color='white', width=2)),
            name=disease_name,
            hoverinfo='text',
            text=hover_text
        ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, max(values) + 3],  # More buffer above max value
                    showticklabels=True,
                    tickfont=dict(size=12),
                    gridcolor='#f1f5f9',  # Lighter grid
                    linecolor='#e2e8f0',
                    showline=True
                ),
                angularaxis=dict(
                    gridcolor='#f1f5f9',
                    tickfont=dict(size=13, family='Inter, sans-serif', color='#1e293b'),
                    linecolor='#e2e8f0',
                    showline=True
                ),
                bgcolor='rgba(248, 250, 252, 0.65)'
            ),
            title=dict(
                text=f'<b>Recommendation Profile</b>',
                font=dict(size=20, color='#1e293b'),
                x=0.5,
                xanchor='center'
            ),
            showlegend=False,
            height=500,
            paper_bgcolor='white',
            margin=dict(t=100, b=60, l=110, r=110)  # More margin to prevent label overlap
        )
        return fig
    
    def create_comparison_bars(d1, d2, data1, data2):
        """Create grouped bar comparison with improved spacing, coloring, and interactivity"""
        categories = ['Precautions', 'Medications', 'Diet', 'Workouts']
        # Use color gradients for bars
        colors1 = ['#667eea', '#5a67d8', '#818cf8', '#3730a3']
        colors2 = ['#ec4899', '#f472b6', '#f9a8d4', '#be185d']
        fig = go.Figure()
        fig.add_trace(go.Bar(
            name=d1,
            y=categories,
            x=data1,
            orientation='h',
            marker=dict(
                color=colors1,
                line=dict(color='white', width=2)
            ),
            text=data1,
            textposition='outside',
            textfont=dict(size=14, color='#667eea', weight='bold'),
            width=0.34,
            hovertemplate=f"<b>{d1}</b><br>%{{y}}: %{{x}}<extra></extra>",
            textangle=0
        ))
        fig.add_trace(go.Bar(
            name=d2,
            y=categories,
            x=data2,
            orientation='h',
            marker=dict(
                color=colors2,
                line=dict(color='white', width=2)
            ),
            text=data2,
            textposition='outside',
            textfont=dict(size=14, color='#ec4899', weight='bold'),
            width=0.34,
            hovertemplate=f"<b>{d2}</b><br>%{{y}}: %{{x}}<extra></extra>",
            textangle=0
        ))
        fig.update_layout(
            title=dict(
                text='<b>Head-to-Head Comparison</b>',
                font=dict(size=22, color='#1e293b'),
                x=0.5,
                xanchor='center'
            ),
            barmode='group',
            bargap=0.32,  # More gap between grouped bars
            height=450,
            paper_bgcolor='white',
            plot_bgcolor='rgba(248, 250, 252, 0.5)',
            xaxis=dict(
                title='<b>Count</b>',
                showgrid=True,
                gridcolor='#e2e8f0',
                tickfont=dict(size=12)
            ),
            yaxis=dict(
                tickfont=dict(size=13, weight='bold'),
                tickangle=-15  # Slight rotation for clarity
            ),
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=1.02,
                xanchor='center',
                x=0.5,
                font=dict(size=13, weight='bold'),
                bgcolor='rgba(255,255,255,0.9)',
                bordercolor='#cbd5e1',
                borderwidth=2
            ),
            margin=dict(t=100, b=60)
        )
        return fig
    
    def create_sunburst(d1, d2, data_dict):
        """Create hierarchical sunburst chart with better label placement, border, and hover info"""
        labels, parents, values, colors = [], [], [], []
        # Root
        labels.extend([d1, d2])
        parents.extend(['', ''])
        values.extend([sum(data_dict[d1]), sum(data_dict[d2])])
        colors.extend(['#667eea', '#ec4899'])
        # Categories
        category_colors = {'Precautions': '#0ea5e9', 'Medications': '#10b981',
                          'Diet': '#f59e0b', 'Workouts': '#a855f7'}
        for disease, parent_color in [(d1, '#667eea'), (d2, '#ec4899')]:
            for i, (cat, count) in enumerate(zip(['Precautions', 'Medications', 'Diet', 'Workouts'], data_dict[disease])):
                if count > 0:
                    labels.append(f"{cat}")
                    parents.append(disease)
                    values.append(count)
                    colors.append(category_colors[cat])
        fig = go.Figure(go.Sunburst(
            labels=labels,
            parents=parents,
            values=values,
            marker=dict(
                colors=colors,
                line=dict(color='white', width=4)  # Thicker white border for separation
            ),
            branchvalues='total',
            hovertemplate='<b>%{label}</b><br>Items: %{value}<extra></extra>',
            insidetextorientation='radial',
            textinfo='label+value',
            textfont=dict(size=17, color='black', family='Inter, sans-serif')  # Larger min text size
        ))
        fig.update_layout(
            title=dict(
                text='<b>Hierarchical Distribution</b>',
                font=dict(size=22, color='#1e293b'),
                x=0.5,
                xanchor='center'
            ),
            height=600,
            paper_bgcolor='white',
            margin=dict(t=80, b=20, l=20, r=20)
        )
        return fig
    
    # ========== MAIN UI ==========
    
    # Hero Header
    st.markdown(f"""
        <div class='hero-header'>
            <h1 style='color: white; font-size: 3.5em; margin: 0; text-shadow: 2px 2px 8px rgba(0,0,0,0.2); position: relative; z-index: 1;'>
                🔬 Disease Intelligence Hub
            </h1>
            <p style='color: rgba(255,255,255,0.95); font-size: 1.4em; margin-top: 1em; font-weight: 500; position: relative; z-index: 1;'>
                Advanced Medical Analytics & Comprehensive Disease Comparison
            </p>
            <div style='margin-top: 1.5em; display: flex; justify-content: center; gap: 2em; flex-wrap: wrap; position: relative; z-index: 1;'>
                <div style='background: rgba(255,255,255,0.2); padding: 0.8em 1.5em; border-radius: 50px;'>
                    <span style='font-size: 1.2em; font-weight: 600;'>📊 {len(description)} Diseases</span>
                </div>
                <div style='background: rgba(255,255,255,0.2); padding: 0.8em 1.5em; border-radius: 50px;'>
                    <span style='font-size: 1.2em; font-weight: 600;'>🔍 Deep Insights</span>
                </div>
                <div style='background: rgba(255,255,255,0.2); padding: 0.8em 1.5em; border-radius: 50px;'>
                    <span style='font-size: 1.2em; font-weight: 600;'>📈 Smart Analytics</span>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.markdown("""
        <div style='background: linear-gradient(135deg, #667eea, #764ba2); padding: 2.5em 1.5em; 
        border-radius: 20px; margin-bottom: 2em; text-align: center; box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);'>
            <h2 style='color: white; margin: 0; font-size: 1.8em;'>🎯 Disease Selector</h2>
            <p style='color: rgba(255,255,255,0.9); margin-top: 0.8em; font-size: 1em;'>
                Select 1-2 diseases for analysis
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Show all disease names first in the select box without descriptions
    selected = st.sidebar.multiselect(
        "🔍 Select Disease",
        options=disease_list,
        default=[],
        help="Select one or two diseases to view their risk and analysis"
    )
    
    # Sidebar Stats
    if disease_list:
        st.sidebar.markdown("### 📊 Database Overview")
        
        total_stats = {
            'Precautions': sum(len(precautions.get(d, [])) for d in disease_list),
            'Medications': sum(len(medications.get(d, [])) for d in disease_list),
            'Diets': sum(len(diets.get(d, [])) for d in disease_list),
            'Workouts': sum(len(workout.get(d, [])) for d in disease_list)
        }
        
        colors = ['#0ea5e9', '#10b981', '#f59e0b', '#ec4899']
        icons = ['⚠️', '💊', '🥗', '🏃']
        
        for (label, count), color, icon in zip(total_stats.items(), colors, icons):
            st.sidebar.markdown(f"""
                <div style='background: linear-gradient(135deg, {color}, {color}dd); color: white; 
                padding: 1.5em; border-radius: 15px; margin: 0.8em 0; text-align: center; 
                box-shadow: 0 6px 20px {color}33;'>
                    <div style='font-size: 2em; margin-bottom: 0.2em;'>{icon}</div>
                    <h3 style='margin: 0; font-size: 2em;'>{count}</h3>
                    <p style='margin: 0.3em 0 0 0; opacity: 0.95; font-size: 0.9em;'>{label}</p>
                </div>
            """, unsafe_allow_html=True)
    
    # Validation
    if not selected:
        st.markdown("""
            <div class='alert-premium'>
                <h3 style='margin-top: 0; color: #92400e; display: flex; align-items: center; gap: 0.5em;'>
                    <span style='font-size: 1.5em;'>⚠️</span>
                    No Disease Selected
                </h3>
                <p style='margin-bottom: 0; color: #78350f; font-size: 1.1em; line-height: 1.6;'>
                    Please select at least one disease from the sidebar to begin your exploration journey.
                </p>
            </div>
        """, unsafe_allow_html=True)
        return
    
    if len(selected) > 2:
        st.markdown(f"""
            <div class='alert-premium' style='background: linear-gradient(135deg, #fee2e2, #fecaca); border-left-color: #dc2626;'>
                <h3 style='margin-top: 0; color: #991b1b; display: flex; align-items: center; gap: 0.5em;'>
                    <span style='font-size: 1.5em;'>🚫</span>
                    Too Many Selections
                </h3>
                <p style='margin-bottom: 0; color: #7f1d1d; font-size: 1.1em; line-height: 1.6;'>
                    Please select a maximum of two diseases. Currently selected: <strong>{len(selected)}</strong>
                </p>
            </div>
        """, unsafe_allow_html=True)
        return
    
    # ========== SINGLE DISEASE VIEW ==========
    if len(selected) == 1:
        disease = selected[0]
        
        # Disease Header
        st.markdown(f"""
            <div class='disease-card-premium' style='background: linear-gradient(135deg, #667eea, #764ba2); 
            color: white; text-align: center; padding: 3em 2em;'>
                <h1 style='margin: 0; font-size: 3em; text-shadow: 2px 2px 8px rgba(0,0,0,0.2);'>
                    {disease}
                </h1>
                <p style='margin-top: 1em; font-size: 1.3em; opacity: 0.95;'>
                    Complete Medical Intelligence Profile
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        # Get data (cleaned to remove empty/NULL entries)
        prec = [p for p in precautions.get(disease, []) if p]
        meds = [m for m in medications.get(disease, []) if m]
        diet = [d for d in diets.get(disease, []) if d]
        work = [w for w in workout.get(disease, []) if w]
        
        # Stats Cards
        stats_data = [
            (len(prec), 'Precautions', '⚠️', 'blue'),
            (len(meds), 'Medications', '💊', 'green'),
            (len(diet), 'Diet Items', '🥗', 'orange'),
            (len(work), 'Workouts', '🏃', 'pink')
        ]
        
        cols = st.columns(4)
        for col, (count, label, icon, color) in zip(cols, stats_data):
            with col:
                st.markdown(f"""
                    <div class='stat-card {color}'>
                        <div class='stat-icon'>{icon}</div>
                        <h2 class='stat-number'>{count}</h2>
                        <p class='stat-label'>{label}</p>
                    </div>
                """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Completeness Score
        completeness = calculate_completeness(disease)
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=completeness,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "<b>Information Richness Score</b>", 'font': {'size': 24, 'color': '#1e293b'}},
            number={'font': {'size': 60, 'color': '#667eea'}, 'suffix': '%'},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 2, 'tickcolor': "#cbd5e1"},
                'bar': {'color': "#667eea", 'thickness': 0.75},
                'bgcolor': "white",
                'borderwidth': 3,
                'bordercolor': "#e2e8f0",
                'steps': [
                    {'range': [0, 33], 'color': '#fecaca'},
                    {'range': [33, 66], 'color': '#fde68a'},
                    {'range': [66, 100], 'color': '#bbf7d0'}
                ],
                'threshold': {
                    'line': {'color': "#667eea", 'width': 6},
                    'thickness': 0.8,
                    'value': completeness
                }
            }
        ))
        fig_gauge.update_layout(
            paper_bgcolor="white",
            height=400,
            margin=dict(l=40, r=40, t=80, b=40)
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

        # === Risk Assessment Visualization ===
        # Calculate risk score
        max_score = 20  # Example max
        risk_score = len(prec) * 0.4 + len(meds) * 0.3 + len(diet) * 0.2 + (5 - len(work)) * 0.1

        fig_risk = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=risk_score,
            domain={'x':[0,1],'y':[0,1]},
            title={'text': f"<b>Risk Score - {disease}</b>"},
            gauge={
                'axis': {'range':[0,max_score]},
                'bar': {'color': "#ef4444"},
                'steps': [
                    {'range':[0, max_score*0.33], 'color':'#bbf7d0'},
                    {'range':[max_score*0.33, max_score*0.66], 'color':'#fde68a'},
                    {'range':[max_score*0.66, max_score], 'color':'#fca5a5'}
                ]
            }
        ))
        fig_risk.update_layout(
            paper_bgcolor="white",
            height=350,
            margin=dict(l=40, r=40, t=60, b=40)
        )
        st.plotly_chart(fig_risk, use_container_width=True)
        
        # Content Tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📖 Overview",
            "⚠️ Precautions",
            "💊 Medications",
            "🥗 Diet Plan",
            "📊 Analytics"
        ])
        
        with tab1:
            st.markdown("<h3 style='color:#0ea5e9;'>📖 Complete Disease Description Table (Search & Scroll)</h3>", unsafe_allow_html=True)

            # Create a clean dataframe with all diseases and their descriptions
            disease_description_df = pd.DataFrame({
                "Disease": [
                    "Fungal infection", "Allergy", "GERD", "Chronic cholestasis", "Drug Reaction", "Peptic ulcer disease",
                    "AIDS", "Diabetes", "Gastroenteritis", "Bronchial Asthma", "Hypertension", "Migraine",
                    "Cervical spondylosis", "Paralysis (brain hemorrhage)", "Jaundice", "Malaria", "Chicken pox",
                    "Dengue", "Typhoid", "Hepatitis A", "Hepatitis B", "Hepatitis C", "Hepatitis D", "Hepatitis E",
                    "Alcoholic hepatitis", "Tuberculosis", "Common Cold", "Pneumonia", "Dimorphic hemmorhoids(piles)",
                    "Heart attack", "Varicose veins", "Hypothyroidism", "Hyperthyroidism", "Hypoglycemia",
                    "Osteoarthristis", "Arthritis", "(Vertigo) Paroxysmal Positional Vertigo", "Acne",
                    "Urinary tract infection", "Psoriasis", "Impetigo"
                ],
                "Description": [
                    "Fungal infection is a common skin condition caused by fungi that thrive in warm, moist areas.",
                    "Allergy is an immune system reaction to a substance that is typically harmless.",
                    "GERD (Gastroesophageal Reflux Disease) is a digestive disorder where stomach acid flows back into the esophagus.",
                    "Chronic cholestasis is a condition where bile flow from the liver slows or stops.",
                    "Drug Reaction occurs when the body reacts adversely to certain medications.",
                    "Peptic ulcer disease involves sores that develop on the lining of the stomach or duodenum.",
                    "AIDS (Acquired Immunodeficiency Syndrome) weakens the immune system caused by HIV.",
                    "Diabetes is a chronic condition that affects how the body turns food into energy.",
                    "Gastroenteritis is an inflammation of the stomach and intestines causing diarrhea and vomiting.",
                    "Bronchial Asthma is a respiratory condition characterized by airway inflammation and difficulty breathing.",
                    "Hypertension, or high blood pressure, increases the risk of heart disease and stroke.",
                    "Migraine is a type of headache that often involves severe pain, nausea, and light sensitivity.",
                    "Cervical spondylosis is a degenerative condition affecting the neck bones and discs.",
                    "Paralysis (brain hemorrhage) occurs due to bleeding in the brain affecting movement.",
                    "Jaundice causes yellowing of the skin due to increased bilirubin levels.",
                    "Malaria is a mosquito-borne infectious disease causing fever and chills.",
                    "Chicken pox is a highly contagious viral infection causing itchy blisters on the skin.",
                    "Dengue is a mosquito-borne viral infection causing fever, rash, and joint pain.",
                    "Typhoid is a bacterial infection that causes high fever, weakness, and abdominal pain.",
                    "Hepatitis A is a viral liver disease spread through contaminated food or water.",
                    "Hepatitis B is a viral infection that affects the liver and can become chronic.",
                    "Hepatitis C is a viral infection that leads to liver inflammation and damage.",
                    "Hepatitis D is a liver infection occurring only with hepatitis B infection.",
                    "Hepatitis E is a viral infection causing liver inflammation, often spread via water.",
                    "Alcoholic hepatitis is inflammation of the liver due to excessive alcohol use.",
                    "Tuberculosis is a bacterial infection that primarily affects the lungs.",
                    "Common Cold is a viral infection causing sneezing, sore throat, and congestion.",
                    "Pneumonia is an infection that inflames air sacs in one or both lungs.",
                    "Dimorphic hemorrhoids (piles) cause swollen veins in the rectum or anus.",
                    "Heart attack occurs when blood flow to the heart is blocked.",
                    "Varicose veins are enlarged, twisted veins that appear under the skin.",
                    "Hypothyroidism is a condition where the thyroid gland doesn't produce enough hormones.",
                    "Hyperthyroidism is a condition where the thyroid gland produces too many hormones.",
                    "Hypoglycemia is low blood sugar that can cause shaking, confusion, or fainting.",
                    "Osteoarthritis is a degenerative joint disease causing pain and stiffness.",
                    "Arthritis is inflammation of joints leading to pain and reduced motion.",
                    "(Vertigo) Paroxysmal Positional Vertigo causes dizziness when changing head position.",
                    "Acne is a skin condition caused by clogged hair follicles with oil and dead skin.",
                    "Urinary tract infection is an infection in any part of the urinary system.",
                    "Psoriasis is a chronic skin condition that speeds up skin cell growth.",
                    "Impetigo is a highly contagious bacterial skin infection."
                ]
            })

            # Add a search box
            search_query = st.text_input("🔍 Search Disease", "", placeholder="Type a disease name...")

            # Filter the dataframe based on search query
            if search_query:
                filtered_df = disease_description_df[
                    disease_description_df["Disease"].str.contains(search_query, case=False, na=False)
                ]
            else:
                filtered_df = disease_description_df

            # Display the scrollable table
            st.dataframe(
                filtered_df,
                use_container_width=True,
                height=500
            )
            
            # Quick Summary Cards
            st.markdown("### 📋 Quick Summary")
            summary_cols = st.columns(2)
            
            with summary_cols[0]:
                st.markdown(f"""
                    <div class='content-section' style='border-left-color: #10b981;'>
                        <h4 style='color: #10b981; margin-top: 0;'>🎯 Total Recommendations</h4>
                        <p style='font-size: 2.5em; font-weight: 800; color: #10b981; margin: 0.3em 0;'>
                            {len(prec) + len(meds) + len(diet) + len(work)}
                        </p>
                        <p style='color: #64748b; margin: 0;'>Combined guidance items</p>
                    </div>
                """, unsafe_allow_html=True)
            
            with summary_cols[1]:
                richness = "Comprehensive" if completeness >= 75 else "Moderate" if completeness >= 50 else "Basic"
                color = "#10b981" if completeness >= 75 else "#f59e0b" if completeness >= 50 else "#ef4444"
                st.markdown(f"""
                    <div class='content-section' style='border-left-color: {color};'>
                        <h4 style='color: {color}; margin-top: 0;'>📊 Information Level</h4>
                        <p style='font-size: 2em; font-weight: 800; color: {color}; margin: 0.3em 0;'>
                            {richness}
                        </p>
                        <p style='color: #64748b; margin: 0;'>{completeness:.1f}% complete profile</p>
                    </div>
                """, unsafe_allow_html=True)
            
            # Workout recommendations
            if work is not None and len(work) > 0:
                st.markdown(f"""
                    <div class='content-section' style='border-left-color: #ec4899;'>
                        <h4 style='color: #ec4899; margin-top: 0; display: flex; align-items: center; gap: 0.5em;'>
                            <span style='font-size: 1.3em;'>🏃</span> Recommended Physical Activities
                        </h4>
                        <div style='display: flex; flex-wrap: wrap; gap: 0.8em; margin-top: 1em;'>
                """, unsafe_allow_html=True)
                
                for item in work:
                    st.markdown(f"""
                        <div style='background: linear-gradient(135deg, #ec4899, #db2777); color: white; 
                        padding: 0.6em 1.2em; border-radius: 25px; font-weight: 600; 
                        box-shadow: 0 4px 15px rgba(236, 72, 153, 0.3);'>
                            {item}
                        </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("</div></div>", unsafe_allow_html=True)
        
        with tab2:
            st.markdown("<h3 class='section-header-premium' style='color: #0ea5e9;'><span>⚠️</span> Safety Precautions</h3>", unsafe_allow_html=True)
            
            if prec is not None and len(prec) > 0:
                for item in prec:
                    st.markdown(f"""
                        <div class="light-box" style="--box-color:#0ea5e9;">
                            {item}
                        </div>
                    """, unsafe_allow_html=True)
        
        with tab3:
            st.markdown("<h3 class='section-header-premium' style='color: #10b981;'><span>💊</span> Medications & Treatments</h3>", unsafe_allow_html=True)
            
            if meds is not None and len(meds) > 0:
                for idx, item in enumerate(meds, 1):
                    st.markdown(f"""
                        <div class='premium-list-item' style='border-left-color: #10b981; --color-start: #10b981; --color-end: #059669;'>
                            <div class='item-number'>{idx}</div>
                            <div style='flex: 1; font-size: 1.05em; line-height: 1.6;'>{item}</div>
                        </div>
                    """, unsafe_allow_html=True)
        
        with tab4:
            st.markdown("<h3 class='section-header-premium' style='color: #f59e0b;'><span>🥗</span> Dietary Recommendations</h3>", unsafe_allow_html=True)
            
            if diet is not None and len(diet) > 0:
                for idx, item in enumerate(diet, 1):
                    st.markdown(f"""
                        <div class='premium-list-item' style='border-left-color: #f59e0b; --color-start: #f59e0b; --color-end: #d97706;'>
                            <div class='item-number'>{idx}</div>
                            <div style='flex: 1; font-size: 1.05em; line-height: 1.6;'>{item}</div>
                        </div>
                    """, unsafe_allow_html=True)
        
        with tab5:
            st.markdown("<h3 class='section-header-premium' style='color: #667eea;'><span>📊</span> Visual Analytics</h3>", unsafe_allow_html=True)
            
            # Radar Chart
            data_counts = [len(prec), len(meds), len(diet), len(work)]
            # --- Place risk gauge above radar chart ---
            # (Already placed above tabs for prominence.)
            radar_fig = create_radar_chart(disease, data_counts)
            st.plotly_chart(radar_fig, use_container_width=True)
            
            # Distribution Chart
            if sum(data_counts) > 0:
                col1, col2 = st.columns(2)
                
                with col1:
                    # Pie Chart
                    fig_pie = go.Figure(data=[go.Pie(
                        labels=['Precautions', 'Medications', 'Diet', 'Workouts'],
                        values=data_counts,
                        hole=0.45,
                        marker=dict(
                            colors=['#0ea5e9', '#10b981', '#f59e0b', '#ec4899'],
                            line=dict(color='white', width=4)
                        ),
                        textinfo='label+percent',
                        textfont=dict(size=13, weight='bold'),
                        hovertemplate='<b>%{label}</b><br>Count: %{value}<br>%{percent}<extra></extra>'
                    )])
                    
                    fig_pie.update_layout(
                        title=dict(
                            text='<b>Distribution</b>',
                            font=dict(size=18, color='#1e293b'),
                            x=0.5,
                            xanchor='center'
                        ),
                        paper_bgcolor='white',
                        height=450,
                        annotations=[dict(
                            text=f'<b>{sum(data_counts)}</b><br>Total',
                            x=0.5, y=0.5,
                            font_size=20,
                            font_color='#667eea',
                            showarrow=False
                        )]
                    )
                    
                    st.plotly_chart(fig_pie, use_container_width=True)
                
                with col2:
                    # Bar Chart
                    fig_bar = go.Figure(data=[go.Bar(
                        y=['Precautions', 'Medications', 'Diet', 'Workouts'],
                        x=data_counts,
                        orientation='h',
                        marker=dict(
                            color=['#0ea5e9', '#10b981', '#f59e0b', '#ec4899'],
                            line=dict(color='white', width=2)
                        ),
                        text=data_counts,
                        textposition='outside',
                        textfont=dict(size=15, weight='bold'),
                        hovertemplate='<b>%{y}</b><br>Items: %{x}<extra></extra>'
                    )])
                    
                    fig_bar.update_layout(
                        title=dict(
                            text='<b>Item Count</b>',
                            font=dict(size=18, color='#1e293b'),
                            x=0.5,
                            xanchor='center'
                        ),
                        height=450,
                        paper_bgcolor='white',
                        plot_bgcolor='rgba(248, 250, 252, 0.5)',
                        xaxis=dict(showgrid=True, gridcolor='#e2e8f0'),
                        yaxis=dict(tickfont=dict(size=12, weight='bold')),
                        margin=dict(t=60, b=40)
                    )
                    
                    st.plotly_chart(fig_bar, use_container_width=True)
    
    # ========== COMPARISON VIEW ==========
    elif len(selected) == 2:
        d1, d2 = selected

        # Gather numeric data for both diseases (cleaned to remove empty/NULL entries)
        prec1 = [p for p in precautions.get(d1, []) if p]
        prec2 = [p for p in precautions.get(d2, []) if p]

        med1 = [m for m in medications.get(d1, []) if m]
        med2 = [m for m in medications.get(d2, []) if m]

        diet1 = [d for d in diets.get(d1, []) if d]
        diet2 = [d for d in diets.get(d2, []) if d]

        work1 = [w for w in workout.get(d1, []) if w]
        work2 = [w for w in workout.get(d2, []) if w]

        # Compute numeric risk scores
        max_score = 20
        risk_score1 = len(prec1) * 0.4 + len(med1) * 0.3 + len(diet1) * 0.2 + (5 - len(work1)) * 0.1
        risk_score2 = len(prec2) * 0.4 + len(med2) * 0.3 + len(diet2) * 0.2 + (5 - len(work2)) * 0.1

        # Determine which disease has higher risk score
        if risk_score1 > risk_score2:
            riskier = d1
            color = "#ef4444"
            order = [(d1, risk_score1, prec1, med1, diet1, work1), (d2, risk_score2, prec2, med2, diet2, work2)]
        elif risk_score2 > risk_score1:
            riskier = d2
            color = "#ef4444"
            order = [(d2, risk_score2, prec2, med2, diet2, work2), (d1, risk_score1, prec1, med1, diet1, work1)]
        else:
            riskier = None
            color = "#f59e0b"
            order = [(d1, risk_score1, prec1, med1, diet1, work1), (d2, risk_score2, prec2, med2, diet2, work2)]

        # Comparison Header: Only show diseases and Higher Risk indicator, no "Disease VS Description" or description comparison
        st.markdown(f"""
            <div class='comparison-header'>
                <h1 style='color: white; margin: 0; font-size: 2.5em; text-shadow: 2px 2px 8px rgba(0,0,0,0.2);'>
                    ⚖️ Disease Risk Comparison
                </h1>
                <div style='margin-top: 1.3em; font-size: 1.5em; font-weight: 600;'>
                    <span style='color: white;'>{order[0][0]}</span>
                    <span class='vs-divider'>VS</span>
                    <span style='color: white;'>{order[1][0]}</span>
                </div>
                <div style='margin-top:2em;'>
        """, unsafe_allow_html=True)
        if riskier:
            st.markdown(
                f"""<div style='display:inline-block; background:{color}; color:white; 
                font-size:1.2em; font-weight:700; padding:0.7em 1.8em; border-radius:30px; 
                box-shadow:0 2px 12px {color}66; margin-bottom:1em;'>
                <span style='font-size:1.3em;'>🛑</span> Higher Risk: {riskier}
                </div>""",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"""<div style='display:inline-block; background:{color}; color:white;
                font-size:1.2em; font-weight:700; padding:0.7em 1.8em; border-radius:30px;
                box-shadow:0 2px 12px {color}66; margin-bottom:1em;'>
                <span style='font-size:1.3em;'>⚠️</span> Both diseases have equal risk
                </div>""",
                unsafe_allow_html=True
            )
        st.markdown("</div></div>", unsafe_allow_html=True)

        # Show Gauge Indicators for both (prominently, higher risk on the left)
        col1, col2 = st.columns(2)
        for i, col in enumerate([col1, col2]):
            name, score, _, _, _, _ = order[i]
            with col:
                fg_color = "#667eea" if i == 0 else "#ec4899"
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=score,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': f"<b>Risk Score - {name}</b>", 'font': {'size': 21, 'color': '#1e293b'}},
                    number={'font': {'size': 48, 'color': fg_color}},
                    gauge={
                        'axis': {'range': [0, max_score], 'tickwidth': 2, 'tickcolor': "#cbd5e1"},
                        'bar': {'color': "#ef4444"},
                        'bgcolor': "white",
                        'borderwidth': 3,
                        'bordercolor': "#e2e8f0",
                        'steps': [
                            {'range': [0, max_score*0.33], 'color': '#bbf7d0'},
                            {'range': [max_score*0.33, max_score*0.66], 'color': '#fde68a'},
                            {'range': [max_score*0.66, max_score], 'color': '#fca5a5'}
                        ],
                        'threshold': {
                            'line': {'color': "#ef4444", 'width': 6},
                            'thickness': 0.8,
                            'value': score
                        }
                    }
                ))
                fig_gauge.update_layout(
                    paper_bgcolor="white",
                    height=280,
                    margin=dict(l=20, r=20, t=60, b=30)
                )
                st.plotly_chart(fig_gauge, use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Stacked bar chart of category-wise risk contributions (numeric only)
        categories = ['Precautions', 'Medications', 'Diet', 'Workouts']
        contribs = []
        for _, _, prec, med, diet, work in order:
            contribs.append([len(prec) * 0.4, len(med) * 0.3, len(diet) * 0.2, (5 - len(work)) * 0.1])
        colors_risk = ['#ef4444', '#fde68a', '#bbf7d0', '#fca5a5']
        fig_stacked = go.Figure()
        for i, (name, _, _, _, _, _) in enumerate(order):
            fig_stacked.add_trace(go.Bar(
                name=name,
                y=categories,
                x=contribs[i],
                orientation='h',
                marker=dict(color=colors_risk, line=dict(color='white', width=2), opacity=1.0 if i == 0 else 0.7),
                text=[f"{v:.2f}" for v in contribs[i]],
                textposition='outside',
                hovertemplate="%{y}: %{x:.2f}<extra></extra>",
            ))
        fig_stacked.update_layout(
            barmode='group',
            title="<b>Risk Contribution by Category</b>",
            height=350,
            paper_bgcolor='white',
            plot_bgcolor='rgba(248, 250, 252, 0.5)',
            xaxis=dict(title="Risk Contribution", gridcolor='#e2e8f0'),
            yaxis=dict(tickfont=dict(size=13)),
            legend=dict(orientation='h', y=1.08, x=0.5, xanchor='center'),
            margin=dict(t=60, b=30)
        )
        st.plotly_chart(fig_stacked, use_container_width=True)

        # Heatmap of risk levels (numeric only)
        risk_matrix = contribs
        fig_heat = go.Figure(data=go.Heatmap(
            z=risk_matrix,
            x=categories,
            y=[order[0][0], order[1][0]],
            colorscale=[ [0, '#bbf7d0'], [0.5, '#fde68a'], [1, '#ef4444'] ],
            colorbar=dict(title="Risk"),
            hovertemplate="Disease: %{y}<br>Category: %{x}<br>Risk: %{z:.2f}<extra></extra>"
        ))
        fig_heat.update_layout(
            title="<b>Category Risk Level Heatmap</b>",
            height=260,
            xaxis=dict(side='top'),
            paper_bgcolor='white',
            margin=dict(t=45, b=25)
        )
        st.plotly_chart(fig_heat, use_container_width=True)

        # Radar charts side by side (numeric counts only)
        st.markdown("### 📡 Recommendation Profile Radar")
        colr1, colr2 = st.columns(2)
        for i, col in enumerate([colr1, colr2]):
            name, _, prec, med, diet, work = order[i]
            data = [len(prec), len(med), len(diet), len(work)]
            with col:
                radar = create_radar_chart(name, data)
                st.plotly_chart(radar, use_container_width=True)
    
    # ========== FOOTER ==========
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    st.markdown("""
        <div class='footer-premium'>
            <h3 style='margin: 0; color: white; font-size: 2em;'>🔬 Disease Intelligence Hub</h3>
            <p style='margin: 1em 0; opacity: 0.9; font-size: 1.1em;'>
                Advanced Medical Information System | Powered by Smart Analytics
            </p>
            <div style='margin-top: 2em; padding-top: 2em; border-top: 1px solid rgba(255,255,255,0.2);'>
                <p style='margin: 0; opacity: 0.75; font-size: 0.95em;'>
                    📊 Visual Analytics • 🔍 Deep Insights • ⚖️ Intelligent Comparison • 💾 Data Export
                </p>
                <p style='margin: 1em 0 0 0; opacity: 0.6; font-size: 0.85em;'>
                    © 2024 Medical Knowledge System | Educational & Research Use Only
                </p>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # Medical Disclaimer
    st.markdown("""
        <div style='background: linear-gradient(135deg, #fef3c7, #fde68a); border-left: 6px solid #f59e0b; 
        border-radius: 20px; padding: 2em; margin-top: 2em; box-shadow: 0 8px 30px rgba(245, 158, 11, 0.15);'>
            <h4 style='color: #92400e; margin: 0 0 1em 0; font-size: 1.3em; display: flex; align-items: center; gap: 0.5em;'>
                <span style='font-size: 1.5em;'>⚠️</span> Medical Disclaimer
            </h4>
            <p style='color: #78350f; margin: 0; line-height: 1.8; font-size: 1.05em;'>
                <strong>Important Notice:</strong> This disease explorer provides educational information only 
                and should not replace professional medical advice, diagnosis, or treatment. Always consult 
                qualified healthcare providers for medical concerns. The information presented here is for 
                reference purposes and may not reflect the most current medical research. Never disregard 
                professional medical advice or delay seeking treatment based on information from this tool.
            </p>
        </div>
    """, unsafe_allow_html=True)




