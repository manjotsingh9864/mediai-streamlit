import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import io
from fpdf import FPDF
import os
from datetime import datetime
import numpy as np
from scipy.special import softmax
import base64

def render_symptom_input(all_symptoms, base_model, decode_label, training, helpers, TMP_DIR, user_id, sel_row):
    """
    Ultimate Symptom Input & Disease Prediction Module
    
    Parameters:
    - all_symptoms: list of all symptom strings
    - base_model: trained ML model for prediction
    - decode_label: function or dict to decode model output to disease names
    - training: training data or metadata with recommendations
    - helpers: optional helper models or data
    - TMP_DIR: directory path for temporary files
    - user_id: current user id (for saving predictions)
    - sel_row: selected patient row or None
    """
    
    # ========== CUSTOM CSS FOR STUNNING DESIGN ==========
    st.markdown("""
        <style>
        /* Main container styling */
        .main-container {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 25px;
            padding: 2.5em;
            margin: 1em 0;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
        }
        
        /* Symptom card styling */
        .symptom-card {
            background: white;
            border-radius: 15px;
            padding: 1.5em;
            margin: 1em 0;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }
        
        .symptom-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
        }
        
        /* Severity badge */
        .severity-badge {
            display: inline-block;
            padding: 0.5em 1em;
            border-radius: 20px;
            font-weight: bold;
            font-size: 0.9em;
            margin: 0.3em;
        }
        
        .severity-mild {
            background: linear-gradient(135deg, #4ade80, #22c55e);
            color: white;
        }
        
        .severity-moderate {
            background: linear-gradient(135deg, #fbbf24, #f59e0b);
            color: white;
        }
        
        .severity-severe {
            background: linear-gradient(135deg, #f87171, #dc2626);
            color: white;
        }
        
        /* Prediction card */
        .prediction-card {
            background: linear-gradient(135deg, #e0f2fe 0%, #f0f9ff 100%);
            border-left: 5px solid #0ea5e9;
            border-radius: 12px;
            padding: 1.5em;
            margin: 1em 0;
            animation: slideIn 0.5s ease;
        }
        
        @keyframes slideIn {
            from {
                opacity: 0;
                transform: translateX(-20px);
            }
            to {
                opacity: 1;
                transform: translateX(0);
            }
        }
        
        /* Recommendation card */
        .recommendation-card {
            background: white;
            border-radius: 15px;
            padding: 2em;
            margin: 1em 0;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
            border-top: 4px solid #8b5cf6;
        }
        
        /* Stats card */
        .stat-card {
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            border-radius: 15px;
            padding: 1.5em;
            text-align: center;
            box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
        }
        
        /* Button enhancement */
        .stButton>button {
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            border: none;
            border-radius: 25px;
            padding: 0.8em 2em;
            font-size: 1.1em;
            font-weight: bold;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        }
        
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
        }
        
        /* Download button */
        .stDownloadButton>button {
            background: linear-gradient(135deg, #10b981, #059669);
            color: white;
            border: none;
            border-radius: 20px;
            padding: 0.7em 1.5em;
            font-weight: 600;
            transition: all 0.3s ease;
        }
        
        .stDownloadButton>button:hover {
            transform: scale(1.05);
            box-shadow: 0 4px 15px rgba(16, 185, 129, 0.4);
        }
        
        /* Expander styling */
        .streamlit-expanderHeader {
            background: linear-gradient(135deg, #f3f4f6, #e5e7eb);
            border-radius: 10px;
            font-weight: 600;
            color: #1f2937;
        }
        
        /* Progress bar */
        .confidence-bar {
            height: 30px;
            border-radius: 15px;
            background: linear-gradient(90deg, #667eea, #764ba2);
            transition: width 0.5s ease;
        }
        
        /* Icon styling */
        .icon {
            font-size: 2em;
            margin-right: 0.3em;
            vertical-align: middle;
        }
        
        /* Alert styling */
        .custom-alert {
            background: linear-gradient(135deg, #f472b6, #8b5cf6);
            border-left: 5px solid #a21caf;
            border-radius: 10px;
            padding: 1.5em;
            margin: 1em 0;
        }
        
        /* Table styling */
        .dataframe {
            border-radius: 10px;
            overflow: hidden;
        }
        
        /* Multi-select enhancement */
        .stMultiSelect [data-baseweb="tag"] {
            background: linear-gradient(135deg, #667eea, #764ba2);
            border-radius: 15px;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # ========== HELPER FUNCTIONS ==========
    
    def build_weighted_vector(selected_symptoms_with_severity):
        """
        Build weighted feature vector with advanced severity mapping
        Severity mapping: Mild=0.33, Moderate=0.66, Severe=1.0
        """
        severity_map = {
            "Mild": 0.33,
            "Moderate": 0.66,
            "Severe": 1.0
        }
        vector = [0.0] * len(all_symptoms)
        symptom_index = {symptom: idx for idx, symptom in enumerate(all_symptoms)}
        
        for symptom, severity in selected_symptoms_with_severity.items():
            if symptom in symptom_index:
                vector[symptom_index[symptom]] = severity_map.get(severity, 0)
        
        return vector
    
    def topk_predictions(model, feature_vector, decode_label, training, helpers=None, k=5):
        """
        Get top K predictions with confidence scores using advanced probability estimation
        Returns: list of tuples (disease_label, confidence_score)
        """
        try:
            # Get probability predictions
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba([feature_vector])[0]
            else:
                # Use decision function with softmax
                scores = model.decision_function([feature_vector])[0]
                proba = softmax(scores)
            
            # Get top K indices
            topk_idx = proba.argsort()[-k:][::-1]
            results = []
            
            for idx in topk_idx:
                # Decode label
                if isinstance(decode_label, dict):
                    label = decode_label.get(idx, f"Unknown_{idx}")
                elif isinstance(decode_label, list):
                    label = decode_label[idx] if idx < len(decode_label) else f"Unknown_{idx}"
                else:
                    label = decode_label(idx)
                
                confidence = float(proba[idx])
                results.append((label, confidence))
            
            return results
        
        except Exception as e:
            st.error(f"Prediction error: {e}")
            return []
    
    def save_prediction_to_db(user_id, sel_row, predictions, selected_symptoms_with_severity):
        """
        Save prediction data with enhanced metadata
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"prediction_{user_id}_{timestamp}.json"
        filepath = os.path.join(TMP_DIR, filename)
        
        data = {
            "user_id": user_id,
            "patient": str(sel_row) if sel_row is not None else "Anonymous",
            "predictions": [{"disease": d, "confidence": float(c)} for d, c in predictions],
            "symptoms": selected_symptoms_with_severity,
            "timestamp": datetime.now().isoformat(),
            "top_prediction": predictions[0][0] if predictions else None,
            "confidence_score": float(predictions[0][1]) if predictions else 0.0
        }
        
        try:
            os.makedirs(TMP_DIR, exist_ok=True)
            with open(filepath, "w") as f:
                json.dump(data, f, indent=2)
            return True, filepath
        except Exception as e:
            st.error(f"Failed to save prediction: {e}")
            return False, None
    
    def generate_enhanced_json_report(predictions, selected_symptoms_with_severity):
        """
        Generate comprehensive JSON report with metadata
        """
        report = {
            "report_metadata": {
                "generated_at": datetime.now().isoformat(),
                "report_version": "3.0",
                "patient_id": user_id if user_id else "Anonymous"
            },
            "predictions": [
                {
                    "rank": idx + 1,
                    "disease": disease,
                    "confidence": float(confidence),
                    "confidence_percentage": f"{confidence * 100:.2f}%",
                    "risk_level": "High" if confidence > 0.7 else "Medium" if confidence > 0.4 else "Low"
                }
                for idx, (disease, confidence) in enumerate(predictions)
            ],
            "symptoms_analysis": {
                "total_symptoms": len(selected_symptoms_with_severity),
                "severity_breakdown": {
                    severity: sum(1 for s in selected_symptoms_with_severity.values() if s == severity)
                    for severity in ["Mild", "Moderate", "Severe"]
                },
                "symptoms_detail": [
                    {"symptom": symptom, "severity": severity}
                    for symptom, severity in selected_symptoms_with_severity.items()
                ]
            },
            "recommendations_summary": {
                "primary_diagnosis": predictions[0][0] if predictions else "Unknown",
                "confidence_level": float(predictions[0][1]) if predictions else 0.0,
                "requires_attention": predictions[0][1] > 0.6 if predictions else False
            }
        }
        return json.dumps(report, indent=2)
    
    def generate_professional_pdf_report(predictions, selected_symptoms_with_severity, training):
        """
        Generate professional medical-grade PDF report
        """
        pdf = FPDF()
        pdf.add_page()
        
        # Header
        pdf.set_fill_color(102, 126, 234)
        pdf.rect(0, 0, 210, 40, 'F')
        pdf.set_text_color(255, 255, 255)
        pdf.set_font("Arial", 'B', 24)
        pdf.cell(0, 15, "", ln=True)
        pdf.cell(0, 10, "AI Medical Diagnosis Report", ln=True, align="C")
        pdf.set_font("Arial", '', 10)
        pdf.cell(0, 8, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True, align="C")
        
        # Reset colors
        pdf.set_text_color(0, 0, 0)
        pdf.ln(10)
        
        # Patient Information
        pdf.set_font("Arial", 'B', 14)
        pdf.set_fill_color(240, 240, 240)
        pdf.cell(0, 10, "Patient Information", ln=True, fill=True)
        pdf.set_font("Arial", '', 11)
        pdf.cell(0, 8, f"Patient ID: {user_id if user_id else 'Anonymous'}", ln=True)
        if sel_row is not None:
            pdf.cell(0, 8, f"Record: {sel_row}", ln=True)
        pdf.ln(5)
        
        # Symptoms Section
        pdf.set_font("Arial", 'B', 14)
        pdf.set_fill_color(240, 240, 240)
        pdf.cell(0, 10, "Reported Symptoms", ln=True, fill=True)
        pdf.set_font("Arial", '', 11)
        
        for symptom, severity in selected_symptoms_with_severity.items():
            severity_color = {
                "Mild": (74, 222, 128),
                "Moderate": (251, 191, 36),
                "Severe": (248, 113, 113)
            }
            color = severity_color.get(severity, (150, 150, 150))
            pdf.set_fill_color(*color)
            pdf.cell(60, 8, f"  {symptom[:30]}", border=1)
            pdf.cell(40, 8, f" {severity}", border=1, fill=True, ln=True)
        
        pdf.ln(5)
        
        # Predictions Section
        pdf.set_font("Arial", 'B', 14)
        pdf.set_fill_color(240, 240, 240)
        pdf.cell(0, 10, "AI Predictions", ln=True, fill=True)
        pdf.set_font("Arial", '', 11)
        
        for idx, (disease, confidence) in enumerate(predictions, 1):
            pdf.set_font("Arial", 'B', 11)
            pdf.cell(0, 8, f"{idx}. {disease}", ln=True)
            pdf.set_font("Arial", '', 10)
            
            # Confidence bar
            bar_width = confidence * 170
            pdf.set_fill_color(102, 126, 234)
            pdf.rect(20, pdf.get_y(), bar_width, 6, 'F')
            pdf.cell(0, 8, f"   Confidence: {confidence:.2%}", ln=True)
            
            # Recommendations if available
            if disease in training:
                rec = training[disease]
                pdf.set_font("Arial", 'I', 9)
                
                desc = rec.get("Description", "")
                if desc:
                    pdf.multi_cell(0, 6, f"   Description: {desc[:200]}")
                
                prec = rec.get("Precautions", [])
                if prec:
                    pdf.cell(0, 6, "   Precautions:", ln=True)
                    for p in prec[:3]:
                        pdf.cell(0, 5, f"     - {p}", ln=True)
                
            pdf.ln(3)
        
        pdf.ln(5)
        
        # Disclaimer
        pdf.set_font("Arial", 'I', 9)
        pdf.set_text_color(128, 128, 128)
        pdf.multi_cell(0, 5, "DISCLAIMER: This report is generated by an AI system and should not replace professional medical advice. Please consult with a qualified healthcare provider for accurate diagnosis and treatment.")
        
        # Footer
        pdf.set_y(-20)
        pdf.set_font("Arial", 'I', 8)
        pdf.cell(0, 10, f"AI Medical System v3.0 | Page {pdf.page_no()}", align="C")
        
        pdf_bytes = pdf.output(dest='S').encode('latin-1')
        return pdf_bytes
    
    def create_confidence_gauge(confidence):
        """
        Create beautiful confidence gauge chart
        """
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=confidence * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Confidence Score", 'font': {'size': 24, 'color': '#1f2937'}},
            delta={'reference': 70, 'increasing': {'color': "#10b981"}, 'decreasing': {'color': '#ef4444'}},
            gauge={
                'axis': {'range': [None, 100], 'tickwidth': 2, 'tickcolor': "#6b7280"},
                'bar': {'color': "#667eea"},
                'bgcolor': "white",
                'borderwidth': 3,
                'bordercolor': "#e5e7eb",
                'steps': [
                    {'range': [0, 40], 'color': '#fecaca'},
                    {'range': [40, 70], 'color': '#fde68a'},
                    {'range': [70, 100], 'color': '#bbf7d0'}
                ],
                'threshold': {
                    'line': {'color': "#dc2626", 'width': 4},
                    'thickness': 0.75,
                    'value': 80
                }
            }
        ))
        
        fig.update_layout(
            paper_bgcolor="white",
            plot_bgcolor="white",
            font={'color': "#1f2937", 'family': "Inter, sans-serif"},
            height=350,
            margin=dict(l=20, r=20, t=60, b=20)
        )
        
        return fig
    
    def create_symptom_severity_chart(selected_symptoms_with_severity):
        """
        Create interactive symptom severity visualization
        """
        severity_order = {"Mild": 1, "Moderate": 2, "Severe": 3}
        severity_colors = {"Mild": "#4ade80", "Moderate": "#fbbf24", "Severe": "#f87171"}
        
        df = pd.DataFrame([
            {"Symptom": symptom, "Severity": severity, "Value": severity_order[severity]}
            for symptom, severity in selected_symptoms_with_severity.items()
        ])
        
        fig = go.Figure()
        
        for severity in ["Mild", "Moderate", "Severe"]:
            df_severity = df[df['Severity'] == severity]
            if not df_severity.empty:
                fig.add_trace(go.Bar(
                    y=df_severity['Symptom'],
                    x=df_severity['Value'],
                    name=severity,
                    orientation='h',
                    marker=dict(
                        color=severity_colors[severity],
                        line=dict(color='white', width=2)
                    ),
                    hovertemplate='<b>%{y}</b><br>Severity: ' + severity + '<extra></extra>'
                ))
        
        fig.update_layout(
            title="Symptom Severity Analysis",
            xaxis_title="Severity Level",
            yaxis_title="Symptoms",
            barmode='stack',
            height=max(400, len(selected_symptoms_with_severity) * 40),
            paper_bgcolor='white',
            plot_bgcolor='rgba(245, 245, 245, 0.5)',
            font=dict(family="Inter, sans-serif", size=12),
            showlegend=True,
            legend=dict(
                x=1.02,
                y=1,
                bgcolor='rgba(255,255,255,0.9)',
                bordercolor='#e5e7eb',
                borderwidth=2
            ),
            xaxis=dict(showgrid=True, gridcolor='lightgray'),
            yaxis=dict(showgrid=False)
        )
        
        return fig
    
    def create_prediction_waterfall(predictions):
        """
        Create waterfall chart showing prediction confidence distribution
        """
        diseases = [d for d, _ in predictions]
        confidences = [c * 100 for _, c in predictions]
        
        fig = go.Figure(go.Waterfall(
            name="Confidence",
            orientation="v",
            measure=["relative"] * len(predictions),
            x=diseases,
            textposition="outside",
            text=[f"{c:.1f}%" for c in confidences],
            y=confidences,
            connector={"line": {"color": "#667eea"}},
            increasing={"marker": {"color": "#10b981"}},
            decreasing={"marker": {"color": "#ef4444"}},
            totals={"marker": {"color": "#667eea"}}
        ))
        
        fig.update_layout(
            title="Confidence Distribution Across Predictions",
            xaxis_title="Diseases",
            yaxis_title="Confidence (%)",
            height=500,
            paper_bgcolor='white',
            plot_bgcolor='rgba(245, 245, 245, 0.5)',
            font=dict(family="Inter, sans-serif", size=13),
            showlegend=False,
            xaxis=dict(tickangle=45)
        )
        
        return fig
    
    # ========== MAIN UI RENDERING ==========
    
    # Header with gradient
    st.markdown("""
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
        padding: 2em; border-radius: 20px; margin-bottom: 2em; text-align: center; 
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);'>
        <h1 style='color: white; font-size: 2.5em; margin: 0; text-shadow: 2px 2px 4px rgba(0,0,0,0.2);'>
        🏥 AI-Powered Disease Prediction System</h1>
        <p style='color: rgba(255,255,255,0.9); font-size: 1.2em; margin-top: 0.5em;'>
        Advanced Machine Learning for Accurate Medical Diagnosis</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Create tabs for better organization
    tab1, tab2, tab3 = st.tabs(["🔍 Symptom Input", "📊 Analysis", "💾 Export & History"])
    
    with tab1:
        # Symptom Selection Section
        st.markdown("""
            <div style='background: white; border-radius: 15px; padding: 2em; 
            box-shadow: 0 4px 15px rgba(0,0,0,0.1); margin-bottom: 2em;'>
            <h3 style='color: #667eea; margin-top: 0;'>
            <span class='icon'>🩺</span>Select Your Symptoms</h3>
            </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns([2, 1])

        with col1:
            # Multi-select with search
            selected_symptoms = st.multiselect(
                "Search and select symptoms from the comprehensive database",
                options=sorted(all_symptoms),
                help="Start typing to search, select multiple symptoms",
                key="symptom_multiselect"
            )

        with col2:
            # Quick stats
            st.markdown(f"""
                <div class='stat-card'>
                <h2 style='margin: 0; font-size: 2.5em;'>{len(all_symptoms)}</h2>
                <p style='margin: 0.5em 0 0 0; opacity: 0.9;'>Available Symptoms</p>
                </div>
            """, unsafe_allow_html=True)

        # Manual input option
        st.markdown("### Or Enter Symptoms Manually")
        manual_symptoms_input = st.text_area(
            "Type symptoms separated by commas",
            placeholder="e.g., headache, fever, cough, fatigue",
            height=100,
            key="manual_symptoms_input"
        )

        # Process manual input
        manual_symptoms = []
        if manual_symptoms_input.strip():
            manual_symptoms = [s.strip() for s in manual_symptoms_input.split(",") if s.strip()]
            # Fuzzy matching (simple contains check)
            matched_manual = []
            for ms in manual_symptoms:
                ms_lower = ms.lower()
                matches = [sym for sym in all_symptoms if ms_lower in sym.lower()]
                if matches:
                    matched_manual.extend(matches[:1])  # Add best match
                else:
                    st.warning(f"⚠️ Symptom '{ms}' not found in database. Please select from the list.")
            manual_symptoms = matched_manual

        # Combine symptoms
        combined_symptoms = list(dict.fromkeys(selected_symptoms + manual_symptoms))

        if not combined_symptoms:
            st.markdown("""
                <div class='custom-alert'>
                <h4 style='margin-top: 0; color: #f59e0b;'>
                <span class='icon'>⚠️</span>No Symptoms Selected</h4>
                <p style='margin-bottom: 0;'>Please select at least one symptom to proceed with the analysis.</p>
                </div>
            """, unsafe_allow_html=True)
            return

        # Display selected symptoms count
        st.success(f"✅ **{len(combined_symptoms)} symptoms** selected for analysis")

        # Severity Selection Section
        st.markdown("""
            <div style='background: white; border-radius: 15px; padding: 2em; 
            box-shadow: 0 4px 15px rgba(0,0,0,0.1); margin-top: 2em;'>
            <h3 style='color: #667eea; margin-top: 0;'>
            <span class='icon'>📈</span>Set Severity Levels</h3>
            <p style='color: #6b7280;'>Adjust the intensity of each symptom for more accurate predictions</p>
            </div>
        """, unsafe_allow_html=True)

        severity_options = ["Mild", "Moderate", "Severe"]
        selected_symptoms_with_severity = {}

        # Create grid layout for severity sliders
        cols_per_row = 2
        for i in range(0, len(combined_symptoms), cols_per_row):
            cols = st.columns(cols_per_row)
            for j, col in enumerate(cols):
                idx = i + j
                if idx < len(combined_symptoms):
                    symptom = combined_symptoms[idx]
                    with col:
                        st.markdown(f"""
                            <div class='symptom-card'>
                            <h4 style='color: #1f2937; margin: 0 0 1em 0;'>
                            {symptom.replace('_', ' ').title()}</h4>
                            </div>
                        """, unsafe_allow_html=True)

                        severity = st.select_slider(
                            f"Severity",
                            options=severity_options,
                            value="Moderate",
                            key=f"severity_slider_{idx}",
                            label_visibility="collapsed"
                        )
                        selected_symptoms_with_severity[symptom] = severity

                        # Display severity badge
                        badge_class = f"severity-{severity.lower()}"
                        st.markdown(f"""
                            <div style='text-align: center; margin-top: -0.5em;'>
                            <span class='severity-badge {badge_class}'>{severity}</span>
                            </div>
                        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Predict Button (centered and prominent)
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            predict_button = st.button(
                "🔮 Generate AI Prediction",
                key="predict_button",
                use_container_width=True,
                type="primary"
            )
    
    # ========== PREDICTION PROCESSING ==========
    if predict_button:
        with st.spinner("🧠 AI is analyzing your symptoms..."):
            try:
                # Build feature vector
                feature_vector = build_weighted_vector(selected_symptoms_with_severity)
                
                # Get predictions
                predictions = topk_predictions(base_model, feature_vector, decode_label, training, helpers, k=5)
                
                if not predictions:
                    st.error("❌ No predictions could be generated. Please try again.")
                    return
                
                # Store in session state
                st.session_state['latest_predictions'] = predictions
                st.session_state['latest_symptoms'] = selected_symptoms_with_severity
                st.session_state['prediction_time'] = datetime.now()

                # --- Unified History Logging for Reports Section ---
                if "history" not in st.session_state:
                    st.session_state["history"] = []

                st.session_state["history"].append({
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "patient": str(sel_row) if sel_row is not None else "Anonymous",
                    "disease": predictions[0][0],
                    "confidence": float(predictions[0][1]),
                    "symptoms": selected_symptoms_with_severity
                })

                # Keep last 20 entries only
                st.session_state["history"] = st.session_state["history"][-20:]
                
                # Success message with animation
                st.markdown("""
                    <div style='background: linear-gradient(135deg, #10b981, #059669); 
                    color: white; padding: 1.5em; border-radius: 15px; text-align: center; 
                    animation: slideIn 0.5s ease; margin: 2em 0;'>
                    <h2 style='margin: 0;'>✨ Analysis Complete!</h2>
                    <p style='margin: 0.5em 0 0 0; opacity: 0.9;'>
                    AI has processed your symptoms and generated predictions</p>
                    </div>
                """, unsafe_allow_html=True)
                
                # Display Results in tab2
                with tab2:
                    st.markdown("## 🎯 Prediction Results")
                    
                    # Top Prediction Highlight
                    top_disease, top_confidence = predictions[0]
                    
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.markdown(f"""
                            <div style='background: linear-gradient(135deg, #667eea, #764ba2); 
                            color: white; padding: 2em; border-radius: 20px; 
                            box-shadow: 0 10px 30px rgba(102, 126, 234, 0.4);'>
                            <h2 style='margin: 0; font-size: 1.5em;'>🏆 Primary Diagnosis</h2>
                            <h1 style='margin: 0.5em 0; font-size: 2.5em; 
                            text-shadow: 2px 2px 4px rgba(0,0,0,0.2);'>{top_disease}</h1>
                            <div style='background: rgba(255,255,255,0.2); 
                            border-radius: 10px; padding: 1em; margin-top: 1em;'>
                            <h3 style='margin: 0; font-size: 1.2em;'>Confidence Level</h3>
                            <h2 style='margin: 0.3em 0 0 0; font-size: 2em;'>{top_confidence*100:.1f}%</h2>
                            </div>
                            </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        # Confidence gauge
                        gauge_fig = create_confidence_gauge(top_confidence)
                        st.plotly_chart(gauge_fig, use_container_width=True)
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    
                    # All Predictions Table
                    st.markdown("### 📋 Complete Prediction Analysis")
                    
                    df_preds = pd.DataFrame(predictions, columns=["Disease", "Confidence"])
                    df_preds["Rank"] = range(1, len(df_preds) + 1)
                    df_preds["Confidence %"] = df_preds["Confidence"].apply(lambda x: f"{x*100:.2f}%")
                    df_preds["Risk Level"] = df_preds["Confidence"].apply(
                        lambda x: "🔴 High" if x > 0.7 else "🟡 Medium" if x > 0.4 else "🟢 Low"
                    )
                    
                    # Reorder columns
                    df_display = df_preds[["Rank", "Disease", "Confidence %", "Risk Level"]]
                    
                    st.dataframe(
                        df_display,
                        use_container_width=True,
                        hide_index=True,
                        height=250
                    )
                    
                    # Horizontal bar chart
                    st.markdown("### 📊 Confidence Distribution")
                    prob_df = pd.DataFrame({
                        "Disease": [d for d, _ in predictions],
                        "Confidence": [c * 100 for _, c in predictions]
                    })
                    
                    fig_bar = px.bar(
                        prob_df,
                        x="Confidence",
                        y="Disease",
                        orientation="h",
                        text=prob_df["Confidence"].apply(lambda x: f"{x:.1f}%"),
                        color="Confidence",
                        color_continuous_scale="Viridis",
                        labels={"Confidence": "Confidence (%)", "Disease": ""}
                    )
                    
                    fig_bar.update_traces(
                        textposition='outside',
                        marker=dict(line=dict(color='white', width=2))
                    )
                    
                    fig_bar.update_layout(
                        yaxis={"categoryorder": "total ascending"},
                        height=400,
                        paper_bgcolor='white',
                        plot_bgcolor='rgba(245, 245, 245, 0.5)',
                        font=dict(family="Inter, sans-serif", size=13),
                        xaxis=dict(range=[0, 105], showgrid=True, gridcolor='lightgray'),
                        margin=dict(l=10, r=10, t=30, b=10),
                        showlegend=False
                    )
                    
                    st.plotly_chart(fig_bar, use_container_width=True)
                    
                    # Waterfall chart
                    st.markdown("### 💧 Comparative Analysis")
                    waterfall_fig = create_prediction_waterfall(predictions)
                    st.plotly_chart(waterfall_fig, use_container_width=True)
                    
                    # Symptom Analysis
                    st.markdown("### 🔬 Symptom Severity Breakdown")
                    severity_fig = create_symptom_severity_chart(selected_symptoms_with_severity)
                    st.plotly_chart(severity_fig, use_container_width=True)
                    
                    # Severity statistics
                    severity_counts = {
                        "Mild": sum(1 for s in selected_symptoms_with_severity.values() if s == "Mild"),
                        "Moderate": sum(1 for s in selected_symptoms_with_severity.values() if s == "Moderate"),
                        "Severe": sum(1 for s in selected_symptoms_with_severity.values() if s == "Severe")
                    }
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.markdown(f"""
                            <div style='background: linear-gradient(135deg, #4ade80, #22c55e); 
                            color: white; padding: 1.5em; border-radius: 15px; text-align: center;'>
                            <h3 style='margin: 0; font-size: 2em;'>{severity_counts['Mild']}</h3>
                            <p style='margin: 0.5em 0 0 0;'>Mild Symptoms</p>
                            </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown(f"""
                            <div style='background: linear-gradient(135deg, #fbbf24, #f59e0b); 
                            color: white; padding: 1.5em; border-radius: 15px; text-align: center;'>
                            <h3 style='margin: 0; font-size: 2em;'>{severity_counts['Moderate']}</h3>
                            <p style='margin: 0.5em 0 0 0;'>Moderate Symptoms</p>
                            </div>
                        """, unsafe_allow_html=True)
                    
                    with col3:
                        st.markdown(f"""
                            <div style='background: linear-gradient(135deg, #f87171, #dc2626); 
                            color: white; padding: 1.5em; border-radius: 15px; text-align: center;'>
                            <h3 style='margin: 0; font-size: 2em;'>{severity_counts['Severe']}</h3>
                            <p style='margin: 0.5em 0 0 0;'>Severe Symptoms</p>
                            </div>
                        """, unsafe_allow_html=True)
                    
                    with col4:
                        st.markdown(f"""
                            <div style='background: linear-gradient(135deg, #667eea, #764ba2); 
                            color: white; padding: 1.5em; border-radius: 15px; text-align: center;'>
                            <h3 style='margin: 0; font-size: 2em;'>{len(selected_symptoms_with_severity)}</h3>
                            <p style='margin: 0.5em 0 0 0;'>Total Symptoms</p>
                            </div>
                        """, unsafe_allow_html=True)
                    
                    st.markdown("<br><br>", unsafe_allow_html=True)
                    
                    # Recommendations Section
                    st.markdown("## 💊 Medical Recommendations")
                    
                    for idx, (disease, confidence) in enumerate(predictions, 1):
                        rec = training.get(disease, {})
                        
                        # Create expander with enhanced styling
                        with st.expander(
                            f"{'🥇' if idx == 1 else '🥈' if idx == 2 else '🥉' if idx == 3 else '📌'} "
                            f"{disease} (Confidence: {confidence*100:.1f}%)",
                            expanded=(idx == 1)
                        ):
                            st.markdown(f"""
                                <div class='recommendation-card'>
                            """, unsafe_allow_html=True)
                            
                            # Description
                            desc = rec.get("Description", "No description available.")
                            st.markdown(f"""
                                <div style='background: linear-gradient(135deg, #e0f2fe, #f0f9ff); 
                                padding: 1.5em; border-radius: 12px; margin-bottom: 1em; 
                                border-left: 4px solid #0ea5e9;'>
                                <h4 style='color: #0369a1; margin-top: 0;'>📖 Description</h4>
                                <p style='color: #1f2937; line-height: 1.6; margin-bottom: 0;'>{desc}</p>
                                </div>
                            """, unsafe_allow_html=True)
                            
                            # Create tabs for different recommendation types
                            rec_tabs = st.tabs(["⚠️ Precautions", "💊 Medications", "🥗 Diet", "🏃 Exercise"])
                            
                            with rec_tabs[0]:
                                prec = rec.get("Precautions", [])
                                if prec:
                                    for p in prec:
                                        st.markdown(f"""
                                            <div style='background: #fef3c7; padding: 1em; 
                                            border-radius: 10px; margin: 0.5em 0; 
                                            border-left: 4px solid #f59e0b;'>
                                            ⚠️ {p}
                                            </div>
                                        """, unsafe_allow_html=True)
                                else:
                                    st.info("No specific precautions listed.")
                            
                            with rec_tabs[1]:
                                meds = rec.get("Medications", [])
                                if meds:
                                    for m in meds:
                                        st.markdown(f"""
                                            <div style='background: #dbeafe; padding: 1em; 
                                            border-radius: 10px; margin: 0.5em 0; 
                                            border-left: 4px solid #3b82f6;'>
                                            💊 {m}
                                            </div>
                                        """, unsafe_allow_html=True)
                                else:
                                    st.info("No specific medications listed.")
                            
                            with rec_tabs[2]:
                                diet = rec.get("Diet", [])
                                if diet:
                                    for d in diet:
                                        st.markdown(f"""
                                            <div style='background: #d1fae5; padding: 1em; 
                                            border-radius: 10px; margin: 0.5em 0; 
                                            border-left: 4px solid #10b981;'>
                                            🥗 {d}
                                            </div>
                                        """, unsafe_allow_html=True)
                                else:
                                    st.info("No specific diet recommendations listed.")
                            
                            with rec_tabs[3]:
                                workouts = rec.get("Workouts", [])
                                if workouts:
                                    for w in workouts:
                                        st.markdown(f"""
                                            <div style='background: #fce7f3; padding: 1em; 
                                            border-radius: 10px; margin: 0.5em 0; 
                                            border-left: 4px solid #ec4899;'>
                                            🏃 {w}
                                            </div>
                                        """, unsafe_allow_html=True)
                                else:
                                    st.info("No specific exercise recommendations listed.")
                            
                            st.markdown("</div>", unsafe_allow_html=True)
                    
                    # Important Disclaimer
                    st.markdown("""
                        <div style='background: linear-gradient(135deg, #fee2e2, #fecaca); 
                        border-left: 5px solid #dc2626; border-radius: 12px; 
                        padding: 1.5em; margin: 2em 0;'>
                        <h4 style='color: #991b1b; margin-top: 0;'>⚠️ Medical Disclaimer</h4>
                        <p style='color: #7f1d1d; margin-bottom: 0; line-height: 1.6;'>
                        <strong>Important:</strong> This AI-powered system is designed to assist in preliminary 
                        disease identification and should NOT replace professional medical advice, diagnosis, 
                        or treatment. Always consult with a qualified healthcare provider for accurate 
                        diagnosis and personalized treatment plans. In case of emergency, contact your local 
                        emergency services immediately.
                        </p>
                        </div>
                    """, unsafe_allow_html=True)
                
                # Export & History Tab
                with tab3:
                    st.markdown("## 💾 Export & Save Results")
                    
                    # Save to database option
                    if sel_row is not None or user_id:
                        st.markdown("""
                            <div style='background: white; border-radius: 15px; padding: 2em; 
                            box-shadow: 0 4px 15px rgba(0,0,0,0.1); margin-bottom: 2em;'>
                            <h3 style='color: #667eea; margin-top: 0;'>
                            <span class='icon'>💾</span>Save to Database</h3>
                            </div>
                        """, unsafe_allow_html=True)
                        
                        if st.button("💾 Save Prediction to Patient Record", use_container_width=True):
                            success, filepath = save_prediction_to_db(
                                user_id, sel_row, predictions, selected_symptoms_with_severity
                            )
                            if success:
                                st.success(f"✅ Prediction saved successfully!\n\nFile: `{os.path.basename(filepath)}`")
                            else:
                                st.error("❌ Failed to save prediction. Please try again.")
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    
                    # Export options
                    st.markdown("""
                        <div style='background: white; border-radius: 15px; padding: 2em; 
                        box-shadow: 0 4px 15px rgba(0,0,0,0.1);'>
                        <h3 style='color: #667eea; margin-top: 0;'>
                        <span class='icon'>📄</span>Export Report</h3>
                        <p style='color: #6b7280;'>Download comprehensive medical reports in multiple formats</p>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        # JSON Export
                        json_report = generate_enhanced_json_report(predictions, selected_symptoms_with_severity)
                        st.download_button(
                            label="📥 Download JSON Report",
                            data=json_report,
                            file_name=f"medical_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json",
                            key="download_json",
                            use_container_width=True
                        )
                        st.markdown("""
                            <p style='text-align: center; color: #6b7280; font-size: 0.85em; margin-top: 0.5em;'>
                            Machine-readable format
                            </p>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        # PDF Export
                        pdf_report = generate_professional_pdf_report(
                            predictions, selected_symptoms_with_severity, training
                        )
                        st.download_button(
                            label="📥 Download PDF Report",
                            data=pdf_report,
                            file_name=f"medical_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                            mime="application/pdf",
                            key="download_pdf",
                            use_container_width=True
                        )
                        st.markdown("""
                            <p style='text-align: center; color: #6b7280; font-size: 0.85em; margin-top: 0.5em;'>
                            Professional format
                            </p>
                        """, unsafe_allow_html=True)
                    
                    with col3:
                        # CSV Export
                        csv_df = pd.DataFrame(predictions, columns=["Disease", "Confidence"])
                        csv_df["Confidence_Percentage"] = csv_df["Confidence"].apply(lambda x: f"{x*100:.2f}%")
                        csv_data = csv_df.to_csv(index=False)
                        
                        st.download_button(
                            label="📥 Download CSV Data",
                            data=csv_data,
                            file_name=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",
                            key="download_csv",
                            use_container_width=True
                        )
                        st.markdown("""
                            <p style='text-align: center; color: #6b7280; font-size: 0.85em; margin-top: 0.5em;'>
                            Spreadsheet format
                            </p>
                        """, unsafe_allow_html=True)
                    
                    st.markdown("<br><br>", unsafe_allow_html=True)
                    
                    # Report Preview
                    st.markdown("### 👁️ Report Preview")
                    
                    preview_tabs = st.tabs(["JSON Preview", "Summary"])
                    
                    with preview_tabs[0]:
                        st.json(json.loads(json_report))
                    
                    with preview_tabs[1]:
                        st.markdown(f"""
                            <div style='background: white; padding: 2em; border-radius: 15px; 
                            box-shadow: 0 4px 15px rgba(0,0,0,0.1);'>
                            <h4 style='color: #667eea;'>Report Summary</h4>
                            <ul style='line-height: 2; color: #374151;'>
                                <li><strong>Patient ID:</strong> {user_id if user_id else 'Anonymous'}</li>
                                <li><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</li>
                                <li><strong>Total Symptoms:</strong> {len(selected_symptoms_with_severity)}</li>
                                <li><strong>Predictions:</strong> {len(predictions)}</li>
                                <li><strong>Top Diagnosis:</strong> {predictions[0][0]}</li>
                                <li><strong>Confidence:</strong> {predictions[0][1]*100:.2f}%</li>
                            </ul>
                            </div>
                        """, unsafe_allow_html=True)
                    
                    # Prediction History (if available in session state)
                    st.markdown("<br><br>", unsafe_allow_html=True)
                    st.markdown("### 📜 Recent Predictions")
                    
                    if 'prediction_history' not in st.session_state:
                        st.session_state['prediction_history'] = []
                    
                    # Add current prediction to history
                    st.session_state['prediction_history'].append({
                        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'disease': predictions[0][0],
                        'confidence': predictions[0][1],
                        'symptoms_count': len(selected_symptoms_with_severity)
                    })
                    
                    # Keep only last 10 predictions
                    if len(st.session_state['prediction_history']) > 10:
                        st.session_state['prediction_history'] = st.session_state['prediction_history'][-10:]
                    
                    # Display history
                    if st.session_state['prediction_history']:
                        history_df = pd.DataFrame(st.session_state['prediction_history'])
                        history_df['Confidence'] = history_df['confidence'].apply(lambda x: f"{x*100:.2f}%")
                        history_df = history_df[['timestamp', 'disease', 'Confidence', 'symptoms_count']]
                        history_df.columns = ['Timestamp', 'Disease', 'Confidence', 'Symptoms']
                        
                        st.dataframe(
                            history_df,
                            use_container_width=True,
                            hide_index=True,
                            height=300
                        )
                        
                        # Clear history button
                        if st.button("🗑️ Clear History", key="clear_history"):
                            st.session_state['prediction_history'] = []
                            st.rerun()
                    else:
                        st.info("No prediction history available yet.")
            
            except Exception as e:
                st.error(f"❌ An error occurred during prediction: {str(e)}")
                st.exception(e)
    
    # If no prediction yet, show helpful info in other tabs
    else:
        with tab2:
            st.markdown("""
                <div style='background: linear-gradient(135deg, #e0f2fe, #f0f9ff); 
                padding: 3em; border-radius: 20px; text-align: center;'>
                <h2 style='color: #0369a1;'>📊 Analysis Dashboard</h2>
                <p style='color: #075985; font-size: 1.2em;'>
                Select symptoms and click "Generate AI Prediction" to see detailed analysis here
                </p>
                <div style='margin-top: 2em;'>
                <span style='font-size: 4em;'>🔍</span>
                </div>
                </div>
            """, unsafe_allow_html=True)
        
        with tab3:
            st.markdown("""
                <div style='background: linear-gradient(135deg, #f0fdf4, #dcfce7); 
                padding: 3em; border-radius: 20px; text-align: center;'>
                <h2 style='color: #15803d;'>💾 Export & Save</h2>
                <p style='color: #166534; font-size: 1.2em;'>
                Generate a prediction to access export and save options
                </p>
                <div style='margin-top: 2em;'>
                <span style='font-size: 4em;'>📄</span>
                </div>
                </div>
            """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("""
        <div style='background: linear-gradient(135deg, #1f2937, #111827); 
        color: white; padding: 2em; border-radius: 20px; margin-top: 3em; text-align: center;'>
        <h3 style='margin: 0; color: white;'>🏥 AI Medical Diagnosis System</h3>
        <p style='margin: 0.5em 0; opacity: 0.8;'>
        Powered by Advanced Machine Learning | Version 3.0
        </p>
        <p style='margin: 1em 0 0 0; opacity: 0.6; font-size: 0.9em;'>
        © 2024 All Rights Reserved | For Educational and Research Purposes
        </p>
        </div>
    """, unsafe_allow_html=True)