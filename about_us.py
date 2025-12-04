import streamlit as st
import pandas as pd

def render_about_us():
    st.markdown("""
    <style>
    /* ====== Gradient Heading ====== */
    .about-gradient-heading {
        font-size: 2.8rem;
        font-weight: 800;
        text-align: center;
        margin-bottom: 0.2em;
        background: linear-gradient(270deg,#2563eb,#06b6d4,#10b981,#f472b6,#a21caf);
        background-size: 1200% 1200%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: gradientGlowHeading2 8s ease infinite;
        letter-spacing: -1px;
    }
    @keyframes gradientGlowHeading2 {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    /* ====== Banner Section ====== */
    .about-banner-bg {
        background: linear-gradient(120deg,#1e293b 60%,#2563eb 100%);
        border-radius: 22px;
        box-shadow: 0 4px 24px rgba(16,185,129,0.15);
        padding: 38px 18px 28px 18px;
        margin-bottom: 35px;
        color: #f1f5f9;
        text-align: center;
    }

    /* ====== Feature Cards ====== */
    .about-key-features {
        display: flex;
        flex-wrap: wrap;
        justify-content: center;
        gap: 26px;
        margin-bottom: 40px;
    }
    .about-feature-card {
        background: linear-gradient(110deg,#f8fafc 60%,#e0e7ef 100%);
        border-radius: 12px;
        box-shadow: 0 2px 10px rgba(59,130,246,0.1);
        padding: 24px 18px;
        min-width: 220px;
        max-width: 300px;
        border: 1.2px solid #38bdf8;
        color: #1e293b;
        text-align: center;
        transition: all 0.2s ease-in-out;
    }
    .about-feature-card:hover {
        transform: translateY(-6px) scale(1.04);
        box-shadow: 0 6px 26px rgba(16,185,129,0.15);
    }
    .about-feature-icon {
        font-size: 2.1rem;
        margin-bottom: 8px;
        color: #06b6d4;
        text-shadow: 0 2px 6px rgba(16,185,129,0.1);
    }
    .about-feature-title {
        font-size: 1.1rem;
        font-weight: bold;
        margin-bottom: 6px;
        color: #0e7490;
    }
    .about-feature-desc {
        font-size: 0.96rem;
        color: #334155;
        opacity: 0.9;
    }

    /* ====== Section Titles ====== */
    .about-section-title {
        font-size: 1.5rem;
        font-weight: 700;
        color: #3b82f6;
        margin-top: 1.2em;
        margin-bottom: 0.5em;
    }

    /* ====== Lists & Unique Items ====== */
    .about-benefit-list {
        margin-left: 0;
        padding-left: 0;
        list-style: none;
    }
    .about-benefit-list li {
        margin-bottom: 7px;
        font-size: 1.05rem;
        color: #334155;
        padding-left: 1.2em;
        position: relative;
    }
    .about-benefit-list li:before {
        content: '✔️';
        position: absolute;
        left: 0;
        color: #06b6d4;
    }
    .about-unique-grid {
        display: flex;
        flex-wrap: wrap;
        gap: 20px;
        justify-content: center;
        margin-bottom: 18px;
    }
    .about-unique-item {
        background: #f1f5f9;
        border-radius: 10px;
        padding: 14px 20px;
        min-width: 180px;
        max-width: 260px;
        box-shadow: 0 1px 6px rgba(59,130,246,0.1);
        text-align: center;
        color: #0e7490;
        font-size: 1.02rem;
        display: flex;
        align-items: center;
        gap: 10px;
        justify-content: center;
        transition: transform 0.2s ease;
    }
    .about-unique-item:hover {
        transform: translateY(-4px);
    }

    /* ====== Team Section ====== */
    .about-team-card {
        background: linear-gradient(90deg,#f0f9ff 60%,#e0e7ef 100%);
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(59,130,246,0.1);
        padding: 18px;
        margin-bottom: 20px;
        color: #1e293b;
        max-width: 420px;
        margin-left: auto;
        margin-right: auto;
    }
    .about-team-name {
        font-size: 1.25rem;
        font-weight: 700;
        color: #0e7490;
        margin-bottom: 6px;
    }
    .about-team-contact {
        font-size: 1rem;
        color: #334155;
        margin-bottom: 3px;
    }
    .about-team-link a {
        color: #2563eb;
        text-decoration: none;
    }
    .about-team-link a:hover {
        text-decoration: underline;
    }

    /* ====== Goal & Disclaimer ====== */
    .about-goal, .about-disclaimer {
        background: #f1f5f9;
        border-radius: 8px;
        padding: 14px 18px;
        margin: 18px 0 12px 0;
        color: #1e293b;
        font-size: 1.05rem;
        box-shadow: 0 1px 6px rgba(59,130,246,0.07);
    }
    .about-goal strong, .about-disclaimer strong {
        color: #eab308;
    }
    </style>
    """, unsafe_allow_html=True)

    # ---------- Intro Banner ----------
    st.markdown("""
    <div class="about-banner-bg">
      <div class="about-gradient-heading">MediAI – Your Intelligent Healthcare Companion</div>
      <div style="font-size:1.2rem; font-weight:500; margin-bottom:0.4em;">
        Empowering you with AI-driven, actionable medical insights.
      </div>
      <div style="font-size:1.05rem; max-width:680px; margin:auto; opacity:0.95;">
        <b>MediAI</b> uses cutting-edge artificial intelligence and data analytics to provide personalized, educational,
        and secure medical recommendations designed to promote awareness and proactive health management.
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ---------- Key Features ----------
    st.markdown("""
    <div class="about-key-features">
      <div class="about-feature-card">
        <div class="about-feature-icon">🤖</div>
        <div class="about-feature-title">AI-Powered Predictions</div>
        <div class="about-feature-desc">Receive accurate, data-driven predictions using advanced ensemble AI models.</div>
      </div>
      <div class="about-feature-card">
        <div class="about-feature-icon">📊</div>
        <div class="about-feature-title">Interactive Visualizations</div>
        <div class="about-feature-desc">Explore predictions with visual charts, heatmaps, and insights for better understanding.</div>
      </div>
      <div class="about-feature-card">
        <div class="about-feature-icon">💬</div>
        <div class="about-feature-title">AI Medical Assistant</div>
        <div class="about-feature-desc">Chat with our AI assistant to get educational responses to medical queries instantly.</div>
      </div>
      <div class="about-feature-card">
        <div class="about-feature-icon">🔒</div>
        <div class="about-feature-title">Secure & Private</div>
        <div class="about-feature-desc">Your information is safe — we follow strict privacy and role-based access control.</div>
      </div>
      <div class="about-feature-card">
        <div class="about-feature-icon">📄</div>
        <div class="about-feature-title">Downloadable Reports</div>
        <div class="about-feature-desc">Generate personalized reports in PDF or JSON format for sharing or storage.</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ---------- Platform Capabilities ----------
    st.markdown("""
    <div class="about-section-title">🚀 Platform Capabilities</div>
    <ul class="about-benefit-list">
      <li><b>Smart Symptom Input:</b> Enter symptoms with severity for personalized predictions.</li>
      <li><b>Explainable AI:</b> Understand model behavior through SHAP-based insights.</li>
      <li><b>Comprehensive Recommendations:</b> Get details about precautions, diet, medication, and workouts.</li>
      <li><b>All-User History:</b> Track predictions and insights across time and users.</li>
      <li><b>Retrain Models:</b> Admins can update and retrain the AI model with new datasets.</li>
    </ul>
    """, unsafe_allow_html=True)

    # ---------- Unique Features ----------
    st.markdown("""
    <div class="about-section-title">✨ Unique Features</div>
    <div class="about-unique-grid">
      <div class="about-unique-item">🧬 Personalized Recommendations</div>
      <div class="about-unique-item">📈 Health Trend Analytics</div>
      <div class="about-unique-item">🧠 Smart Learning AI</div>
      <div class="about-unique-item">🗂️ Patient Management</div>
      <div class="about-unique-item">🔗 Seamless Workflow Integration</div>
    </div>
    """, unsafe_allow_html=True)

    # ---------- Who Can Benefit ----------
    st.markdown("""
    <div class="about-section-title">👥 Who Can Benefit?</div>
    <ul class="about-benefit-list">
      <li><b>Students:</b> Learn how AI assists in medical predictions and decision support.</li>
      <li><b>Doctors:</b> Use as a reference for quick insights and educational demos.</li>
      <li><b>Patients:</b> Understand symptoms and next steps for better health awareness.</li>
      <li><b>Researchers:</b> Analyze results to enhance healthcare data science.</li>
    </ul>
    """, unsafe_allow_html=True)

    # ---------- Team Section ----------
    st.markdown("""
    <div class="about-section-title">👨‍💻 Developer</div>
    <div class="about-team-card">
      <div class="about-team-name">Manjot Singh</div>
      <div class="about-team-contact">📞 7087736640</div>
      <div class="about-team-contact">📧 singhtmanjot@gmail.com</div>
      <div class="about-team-link">
        🔗 <a href="https://www.linkedin.com/in/manjot-singh-ds/" target="_blank">LinkedIn</a> &nbsp; | &nbsp;
        💻 <a href="https://github.com/manjotsingh9864" target="_blank">GitHub</a>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ---------- Visualization Example ----------
    st.markdown("""
    <div class="about-section-title">📊 Example Visualization</div>
    <div style="margin-bottom:10px; color:#64748b;">Sample chart based on user history data:</div>
    """, unsafe_allow_html=True)

    if "history" in st.session_state and st.session_state.history:
        df = pd.DataFrame(st.session_state.history)
        if "prediction" in df.columns and not df.empty:
            chart_data = df["prediction"].value_counts().head(5)
            st.bar_chart(chart_data)
    else:
        st.info("No history data yet. Make a prediction to visualize here!")

    # ---------- Goal & Disclaimer ----------
    st.markdown("""
    <div class="about-section-title">🎯 Project Goal</div>
    <div class="about-goal"><strong>MediAI</strong> aims to combine artificial intelligence and healthcare knowledge to 
    make medical awareness, education, and early understanding accessible to everyone.</div>

    <div class="about-section-title">⚠️ Disclaimer</div>
    <div class="about-disclaimer"><strong>MediAI</strong> is an educational tool — not a substitute for professional medical advice. 
    Always consult a certified healthcare provider for medical diagnosis or treatment.</div>

    <div style="text-align:center; font-size:1.15rem; margin-top:16px; color:#06b6d4;">
      🏥 <i>MediAI – Developed with ❤️ by Manjot Singh</i>
    </div>
    """, unsafe_allow_html=True)