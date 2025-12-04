import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import networkx as nx
from fpdf import FPDF
import sqlite3, json, os, hashlib, tempfile, textwrap
from datetime import datetime
from pathlib import Path

# Optional deps (handled with graceful fallbacks)
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.model_selection import train_test_split
    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False

# Optional Google OAuth (requires: pip install streamlit-oauth)
OAUTH_OK = False
try:
    if 'oauth' in st.secrets:
        from streamlit_oauth import OAuth2Component  # type: ignore
        OAUTH_OK = True
except Exception:
    OAUTH_OK = False

from utils.common import (
    load_data, load_model, build_symptom_dict, get_label_decoder,
    fetch_recommendations, topk_predictions
)

# ---------------------- Config & Styles ----------------------
st.set_page_config(page_title="AI Medical Recommendation System", page_icon="🩺", layout="wide")

st.markdown(
    """
    <style>
    :root { --card-bg: #ffffff; --card-grad: linear-gradient(135deg,#e0f7fa 0%, #ffffff 100%); }
    .hero {padding: 18px 20px; border-radius: 16px; background: var(--card-grad); color:#1e293b; border:1px solid #cbd5e1;}
    .metric {background: rgba(255,255,255,0.7); border: 1px solid #e2e8f0; border-radius: 14px; padding: 14px; box-shadow: 0 2px 6px rgba(0,0,0,0.08);}
    .metric h4 {margin:0; color:#2563eb; font-weight:600}
    .metric h2 {margin:4px 0 0 0; color:#0f172a}
    .login-card {border:1px solid #e2e8f0; padding:18px; border-radius:14px; background:rgba(255,255,255,0.8); backdrop-filter: blur(6px);}
    .btn-google {display:inline-flex; align-items:center; gap:10px; padding:10px 14px; border:1px solid #cbd5e1; border-radius:10px; color:#2563eb; text-decoration:none; background:white; transition: all 0.3s ease-in-out;}
    .btn-google:hover {background: linear-gradient(90deg, #2563eb, #10b981); color:white; }
    .subtle {color:#475569; font-size:12px}
    </style>
    """,
    unsafe_allow_html=True,
)

APP_DIR = Path.cwd()
DATA_DIR = APP_DIR / "data"
TMP_DIR = Path(tempfile.gettempdir())

# ---------------------- Database (SQLite) ----------------------
DB_PATH = APP_DIR / "app.db"

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            pw_hash TEXT,
            created_at TEXT
        );
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS patients (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            name TEXT,
            age INTEGER,
            gender TEXT,
            notes TEXT,
            created_at TEXT,
            FOREIGN KEY(user_id) REFERENCES users(id)
        );
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            patient_id INTEGER,
            timestamp TEXT,
            symptoms_json TEXT,
            prediction TEXT,
            confidence REAL,
            details_json TEXT,
            FOREIGN KEY(user_id) REFERENCES users(id),
            FOREIGN KEY(patient_id) REFERENCES patients(id)
        );
    """)
    conn.commit(); conn.close()

init_db()

# ---------------------- Auth Helpers ----------------------

def hash_password(pw: str) -> str:
    return hashlib.sha256(pw.encode()).hexdigest()

@st.cache_data(show_spinner=False)
def fetch_user(username: str):
    conn = get_db(); cur = conn.cursor()
    cur.execute("SELECT * FROM users WHERE username=?", (username,))
    row = cur.fetchone(); conn.close(); return row

@st.cache_data(show_spinner=False)
def list_patients(user_id: int) -> pd.DataFrame:
    conn = get_db(); df = pd.read_sql_query("SELECT * FROM patients WHERE user_id=? ORDER BY created_at DESC", conn, params=(user_id,))
    conn.close(); return df


def create_user(username: str, pw: str) -> bool:
    try:
        conn = get_db(); cur = conn.cursor()
        cur.execute("INSERT INTO users (username, pw_hash, created_at) VALUES (?,?,?)",
                    (username, hash_password(pw), datetime.now().isoformat()))
        conn.commit(); conn.close(); return True
    except Exception:
        return False


def get_or_create_patient(user_id: int, name: str, age: int, gender: str, notes: str="") -> int:
    conn = get_db(); cur = conn.cursor()
    cur.execute("INSERT INTO patients (user_id, name, age, gender, notes, created_at) VALUES (?,?,?,?,?,?)",
                (user_id, name, age, gender, notes, datetime.now().isoformat()))
    conn.commit(); pid = cur.lastrowid; conn.close(); return pid


def insert_history(user_id: int, patient_id: int, symptoms: list, prediction: str, confidence: float, details: dict):
    conn = get_db(); cur = conn.cursor()
    cur.execute("INSERT INTO history (user_id, patient_id, timestamp, symptoms_json, prediction, confidence, details_json) VALUES (?,?,?,?,?,?,?)",
                (user_id, patient_id, datetime.now().isoformat(), json.dumps(symptoms), prediction, float(confidence), json.dumps(details)))
    conn.commit(); conn.close()


# ---------------------- Load Data/Model ----------------------
# (Relies on your utils.common)

description, precautions, medications, diets, workout, training, symptoms_df, severity = load_data()
base_model = load_model()
decode_label = get_label_decoder(base_model)
symptoms_dict = build_symptom_dict(training)
all_symptoms = list(symptoms_dict.keys()) if symptoms_dict else []

# ---------------------- Login / Register UI REMOVED ----------------------
# Bypass login: set default user in session state
if "auth" not in st.session_state or st.session_state.auth.get("user_id") is None:
    st.session_state.auth = {"user_id": 1, "username": "Guest"}

# ---------------------- Sidebar (Patient profile) ----------------------
user_id = st.session_state.auth["user_id"]
username = st.session_state.auth["username"]

st.sidebar.markdown(f"### 👤 {username}")
if st.sidebar.button("Logout"):
    st.session_state.auth = {"user_id": None, "username": None}
    st.experimental_rerun()

st.sidebar.markdown("---")
pat_df = list_patients(user_id)

patient_choice = st.sidebar.selectbox(
    "Select patient", options=["➕ Add new patient"] + pat_df.get("name", pd.Series([])).tolist()
)

if patient_choice == "➕ Add new patient":
    with st.sidebar.form("new_patient"):
        p_name = st.text_input("Full name")
        p_age = st.number_input("Age", 0, 120, 25)
        p_gender = st.selectbox("Gender", ["Prefer not to say","Male","Female","Other"])
        p_notes = st.text_area("Notes (optional)")
        if st.form_submit_button("Create"):
            if p_name.strip():
                pid = get_or_create_patient(user_id, p_name.strip(), int(p_age), p_gender, p_notes)
                st.success(f"Patient created: {p_name}")
                st.experimental_rerun()
            else:
                st.warning("Enter a name")
else:
    # set selected patient row
    sel_row = pat_df[pat_df.name == patient_choice].iloc[0] if not pat_df.empty else None

st.sidebar.markdown("---")
st.sidebar.info("**Disclaimer:** Educational use only — not medical advice.")

# ---------------------- Top Metrics ----------------------
col1, col2, col3, col4 = st.columns(4)

def metric_card(title, value, subtitle=""):
    with st.container():
        st.markdown(f"<div class='metric'><h4>{title}</h4><h2>{value}</h2><div class='subtle'>{subtitle}</div></div>", unsafe_allow_html=True)

# history metrics (from DB)
conn = get_db()
hist_count = conn.execute("SELECT COUNT(*) c FROM history WHERE user_id=?", (user_id,)).fetchone()["c"]
avg_conf = conn.execute("SELECT COALESCE(AVG(confidence),0) a FROM history WHERE user_id=?", (user_id,)).fetchone()["a"]
unique_symptoms = len(all_symptoms)
metric_card("Predictions", hist_count, "All time")
metric_card("Avg confidence", f"{avg_conf:.1%}")
metric_card("Unique symptoms", unique_symptoms)
metric_card("Patients", len(pat_df))
conn.close()

st.markdown("<div class='hero'><b>Tip:</b> Weight symptom severity for better predictions. Upload notes/images to include in the report. Use the Assistant tab to ask domain questions from your own data.</div>", unsafe_allow_html=True)

# ---------------------- Tabs ----------------------
tabs = st.tabs(["🔎 Predict","📚 Explorer","📊 Visualizations","📁 Reports","🤖 Assistant"])

# ------------ Helper: Weighted vector -------------

def build_weighted_vector(selected: list, severity_map: dict) -> np.ndarray:
    vec = np.zeros(len(symptoms_dict))
    for s in selected:
        idx = symptoms_dict.get(s)
        if idx is None: continue
        vec[idx] = float(severity_map.get(s, 1.0))
    return vec.reshape(1,-1)

# ------------ Helper: Train small helper models (optional) -------------
cache_decorator = getattr(st, "cache_resource", getattr(st, "cache", lambda f: f))

@cache_decorator
def train_helper_models(training_df: pd.DataFrame):
    helpers = {}
    if not SKLEARN_OK or training_df is None or training_df.empty:
        return helpers
    X = training_df.iloc[:, :-1].values
    y = training_df.iloc[:, -1].values
    try:
        X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    except Exception:
        X_tr, y_tr = X, y
        X_va, y_va = X, y
    try:
        rf = RandomForestClassifier(n_estimators=120, random_state=42)
        rf.fit(X_tr, y_tr)
        try:
            rf = CalibratedClassifierCV(rf, cv='prefit').fit(X_va, y_va)
        except Exception:
            pass
        helpers['rf'] = rf
    except Exception:
        pass
    try:
        lr = LogisticRegression(max_iter=600)
        lr.fit(X_tr, y_tr)
        try:
            lr = CalibratedClassifierCV(lr, cv='prefit').fit(X_va, y_va)
        except Exception:
            pass
        helpers['lr'] = lr
    except Exception:
        pass
    return helpers

helpers = train_helper_models(training)

# ---------------------- PREDICT TAB ----------------------
with tabs[0]:
    st.subheader("Symptom Input & Prediction")
    if all_symptoms:
        c1, c2 = st.columns([2,1])
        with c1:
            selected = st.multiselect("Select symptoms", options=all_symptoms)
            typed = st.text_input("...or type comma-separated", placeholder="fever, cough, headache")
            if typed.strip():
                extra = [s.strip() for s in typed.split(',') if s.strip()]
                invalid = [s for s in extra if s not in symptoms_dict]
                if invalid:
                    st.warning(f"Ignored invalid: {', '.join(invalid)}")
                selected = sorted(set(selected).union([s for s in extra if s in symptoms_dict]))
            st.markdown("#### Severity per symptom")
            severity_map = {}
            for s in selected:
                sev = st.select_slider(f"{s}", options=["Mild","Moderate","Severe"], value="Moderate", key=f"sev_{s}")
                severity_map[s] = {"Mild":0.6,"Moderate":1.0,"Severe":1.6}[sev]
        with c2:
            st.markdown("#### Attachments (optional)")
            up_text = st.file_uploader("Notes (.txt/.md)", type=["txt","md"], key="note_up")
            up_image = st.file_uploader("Image (.png/.jpg)", type=["png","jpg","jpeg"], key="img_up")
            up_image_path = None
            if up_image is not None:
                up_image_path = str(TMP_DIR / f"upload_{datetime.now().timestamp()}_{up_image.name}")
                with open(up_image_path, "wb") as f:
                    f.write(up_image.read())
                st.image(up_image_path, caption="Uploaded image", use_container_width=True)

        if st.button("Predict", type="primary"):
            if not selected:
                st.warning("Select or type at least one valid symptom.")
            else:
                X = build_weighted_vector(selected, severity_map)
                # Base model (utils)
                try:
                    idxs, probs = topk_predictions(base_model, X.flatten(), k=5)
                except Exception:
                    # fallback if topk helper fails
                    idxs, probs = np.array([0]), np.array([0.5])
                decoded = [get_label_decoder(base_model)(i) for i in idxs]

                # Helper models votes
                names, model_preds, model_probs = [], [], []
                if 'rf' in helpers:
                    try:
                        p = helpers['rf'].predict_proba(X)[0]; ti = int(np.argmax(p))
                        names.append('RF'); model_preds.append(decode_label(ti)); model_probs.append(float(p[ti]))
                    except Exception: pass
                if 'lr' in helpers:
                    try:
                        p = helpers['lr'].predict_proba(X)[0]; ti = int(np.argmax(p))
                        names.append('LR'); model_preds.append(decode_label(ti)); model_probs.append(float(p[ti]))
                    except Exception: pass
                # include base
                names.insert(0, 'Base'); model_preds.insert(0, decoded[0]); model_probs.insert(0, float(probs[0]))

                # Majority vote
                if model_preds:
                    consensus = max(set(model_preds), key=model_preds.count)
                else:
                    consensus = decoded[0]
                conf_vals = [pr for pr, prd in zip(model_probs, model_preds) if prd == consensus] or [model_probs[0]]
                conf = float(np.mean(conf_vals))

                top_disease = consensus
                st.success(f"**Prediction:** {top_disease} — **Confidence:** {conf:.2%}")

                # Display table
                pred_df = pd.DataFrame({"Model": names, "Prediction": model_preds, "Confidence": model_probs})
                st.dataframe(pred_df, use_container_width=True)

                # Fetch recommendations
                desc, pre, meds, diet, wrk = fetch_recommendations(top_disease, description, precautions, medications, diets, workout)
                with st.expander("Recommendations"):
                    st.write(desc or "No description available.")
                    cA, cB, cC = st.columns(3)
                    with cA:
                        st.markdown("**🛡️ Precautions**"); [st.write('-', p) for p in pre or []]
                    with cB:
                        st.markdown("**💊 Medications**"); [st.write('•', m) for m in meds or []]
                    with cC:
                        st.markdown("**🥗 Diet**"); [st.write('•', d) for d in diet or []]
                    st.markdown("**🏃 Workouts**"); [st.write('•', w) for w in wrk or []]

                # Visual bar
                fig = go.Figure(go.Bar(x=model_probs, y=[f"{n}: {p}" for n,p in zip(names,model_preds)], orientation='h'))
                fig.update_layout(height=320, margin=dict(l=10,r=10,t=10,b=10))
                st.plotly_chart(fig, use_container_width=True)

                # Save to DB if a patient is selected
                if patient_choice != "➕ Add new patient" and not pat_df.empty:
                    pid = int(sel_row["id"])  # selected patient id
                    details = {
                        "selected": selected,
                        "severity": severity_map,
                        "models": list(zip(names, model_preds, model_probs)),
                        "desc": desc, "pre": pre, "meds": meds, "diet": diet, "wrk": wrk,
                        "notes_file": up_text.name if up_text else None,
                        "image_path": up_image_path,
                    }
                    insert_history(user_id, pid, selected, top_disease, conf, details)
                    st.caption("Saved to patient history.")

                # Export buttons
                if st.button("Download JSON report"):
                    entry = {
                        "timestamp": datetime.now().isoformat(),
                        "user": username,
                        "patient": None if patient_choice=="➕ Add new patient" else dict(sel_row),
                        "prediction": top_disease,
                        "confidence": conf,
                        "models": list(zip(names, model_preds, model_probs)),
                        "symptoms": selected,
                        "severity": severity_map,
                        "desc": desc, "pre": pre, "meds": meds, "diet": diet, "wrk": wrk,
                    }
                    st.download_button("⬇️ report.json", data=json.dumps(entry, indent=2), file_name="report.json")

                if st.button("Download PDF report"):
                    pdf = FPDF(); pdf.set_auto_page_break(auto=True, margin=12); pdf.add_page()
                    try:
                        pdf.add_font("DejaVu","", "DejaVuSans.ttf", uni=True)
                        pdf.add_font("DejaVu","B","DejaVuSans-Bold.ttf", uni=True)
                        base_font="DejaVu"; bold="B"
                    except Exception:
                        base_font="Helvetica"; bold=""
                    pdf.set_font(base_font, bold, 16); pdf.cell(0,10, "Medical Recommendation Report", ln=True)
                    pdf.set_font(base_font, "", 12)
                    pdf.cell(0,8, f"User: {username}", ln=True)
                    pdf.cell(0,8, f"Prediction: {top_disease}   Confidence: {conf:.2%}", ln=True)
                    pdf.ln(2)
                    def add_section(title, items):
                        pdf.set_font(base_font, "B" if bold else "", 12); pdf.cell(0,8, title, ln=True)
                        pdf.set_font(base_font, "", 12)
                        if isinstance(items, (list, tuple)):
                            for it in items:
                                if it: pdf.multi_cell(0,7, f"• {it}")
                        elif isinstance(items, str) and items:
                            for line in textwrap.wrap(items, 110):
                                pdf.multi_cell(0,7, line)
                        pdf.ln(1)
                    add_section("Description", desc or "")
                    add_section("Precautions", pre or [])
                    add_section("Medications", meds or [])
                    add_section("Diet", diet or [])
                    add_section("Workouts", wrk or [])
                    if up_image_path and os.path.exists(up_image_path):
                        pdf.set_font(base_font, "B" if bold else "", 12); pdf.cell(0,8, "Attachment", ln=True)
                        try:
                            pdf.image(up_image_path, w=170)
                        except Exception:
                            pdf.multi_cell(0,7, f"[Image attached: {up_image_path}]")
                    st.download_button(
                        "⬇️ medical_report.pdf",
                        data=bytes(pdf.output(dest="S")),
                        file_name="medical_report.pdf",
                        mime="application/pdf",
                    )
    else:
        st.info("Training data not loaded. Please check your data pipeline.")

# ---------------------- EXPLORER TAB ----------------------
with tabs[1]:
    st.subheader("Disease Explorer")
    diseases = sorted(description["Disease"].dropna().unique()) if isinstance(description, pd.DataFrame) and not description.empty else []
    if diseases:
        disease_pick = st.selectbox("Choose disease", options=diseases)
        if disease_pick:
            desc, pre, meds, diet, wrk = fetch_recommendations(disease_pick, description, precautions, medications, diets, workout)
            st.write(desc or "No description")
            c1, c2, c3 = st.columns(3)
            with c1: st.markdown("### 🛡️ Precautions"); [st.write('-', p) for p in pre or []]
            with c2: st.markdown("### 💊 Medications"); [st.write('•', m) for m in meds or []]
            with c3: st.markdown("### 🥗 Diet"); [st.write('•', d) for d in diet or []]
            st.markdown("### 🏃 Workouts"); [st.write('•', w) for w in wrk or []]
        st.markdown("---")
        st.subheader("Compare diseases")
        cA, cB = st.columns(2)
        a = cA.selectbox("Disease A", diseases, key="cmpA")
        b = cB.selectbox("Disease B", diseases, key="cmpB")
        if st.button("Compare"):
            A = fetch_recommendations(a, description, precautions, medications, diets, workout)
            B = fetch_recommendations(b, description, precautions, medications, diets, workout)
            c1, c2 = st.columns(2)
            with c1:
                st.write(f"### {a}")
                st.write(A[0] or "No description")
                st.markdown("**Precautions**"); [st.write('-', p) for p in A[1] or []]
            with c2:
                st.write(f"### {b}")
                st.write(B[0] or "No description")
                st.markdown("**Precautions**"); [st.write('-', p) for p in B[1] or []]
    else:
        st.info("No disease metadata available.")

# ---------------------- VISUALIZATIONS TAB ----------------------
with tabs[2]:
    st.subheader("Visualizations & Insights")
    if isinstance(training, pd.DataFrame) and not training.empty:
        sym_cols = training.columns[:-1]
        freq = training[sym_cols].sum().sort_values(ascending=False)[:25]
        st.plotly_chart(go.Figure([go.Bar(x=freq.index, y=freq.values)]).update_layout(height=350), use_container_width=True)

        # Heatmap of co-occurrence
        top_syms = list(freq.index[:20])
        sub = training[top_syms]
        corr = sub.corr()
        fig_hm = px.imshow(corr, color_continuous_scale="RdBu", zmin=-1, zmax=1, title="Symptom Co-occurrence (Top 20)")
        st.plotly_chart(fig_hm, use_container_width=True)

        # Network graph (thresholded)
        G = nx.Graph()
        for s in top_syms: G.add_node(s)
        for i, a in enumerate(top_syms):
            for j, b in enumerate(top_syms):
                if j <= i: continue
                w = corr.loc[a, b]
                if w > 0.35: G.add_edge(a, b, weight=float(w))
        if G.number_of_edges() > 0:
            pos = nx.spring_layout(G, seed=42)
            edge_x, edge_y = [], []
            for e in G.edges():
                x0, y0 = pos[e[0]]; x1, y1 = pos[e[1]]
                edge_x += [x0, x1, None]; edge_y += [y0, y1, None]
            edge_trace = go.Scatter(x=edge_x, y=edge_y, mode='lines', opacity=0.5)
            node_x, node_y, text = [], [], []
            for n in G.nodes():
                x, y = pos[n]; node_x.append(x); node_y.append(y); text.append(n)
            node_trace = go.Scatter(x=node_x, y=node_y, text=text, mode='markers+text', textposition='bottom center')
            fig_net = go.Figure(data=[edge_trace, node_trace])
            fig_net.update_layout(title="Symptom Relationship Network", showlegend=False, height=500)
            st.plotly_chart(fig_net, use_container_width=True)
    else:
        st.info("Training data not available.")

# ---------------------- REPORTS TAB ----------------------
with tabs[3]:
    st.subheader("Reports & History")
    conn = get_db(); df_hist = pd.read_sql_query(
        "SELECT h.id, h.timestamp, p.name as patient, h.prediction, h.confidence FROM history h LEFT JOIN patients p ON h.patient_id=p.id WHERE h.user_id=? ORDER BY h.timestamp DESC",
        conn, params=(user_id,)
    )
    conn.close()
    if not df_hist.empty:
        st.dataframe(df_hist, use_container_width=True)
        st.download_button("⬇️ Export CSV", data=df_hist.to_csv(index=False).encode('utf-8'), file_name="history.csv")
    else:
        st.caption("No history yet.")

# ---------------------- ASSISTANT (RAG-lite) ----------------------
with tabs[4]:
    st.subheader("Assistant — Ask about your data (RAG-lite)")
    st.caption("This assistant searches your uploaded notes and the disease metadata to answer.")

    # Build corpus from description table + latest notes
    corpus_texts, corpus_labels = [], []
    if isinstance(description, pd.DataFrame) and not description.empty:
        for _, r in description.iterrows():
            txt = " ".join([str(r.get(c, "")) for c in description.columns])
            corpus_texts.append(txt)
            corpus_labels.append(f"DiseaseMeta:{r.get('Disease','')}" )

    # Also include user's saved history details
    conn = get_db()
    rows = conn.execute("SELECT details_json FROM history WHERE user_id=? ORDER BY timestamp DESC LIMIT 50", (user_id,)).fetchall()
    conn.close()
    for rr in rows:
        try:
            details = json.loads(rr["details_json"]) if isinstance(rr, sqlite3.Row) else json.loads(rr[0])
            text_blob = " ".join([
                " ".join(details.get("selected", [])),
                json.dumps(details.get("severity", {})),
                " ".join(details.get("pre", []) or []),
                " ".join(details.get("meds", []) or []),
                " ".join(details.get("diet", []) or []),
                " ".join(details.get("wrk", []) or []),
            ])
            corpus_texts.append(text_blob)
            corpus_labels.append("UserHistory")
        except Exception:
            pass

    query = st.text_input("Ask a question (e.g., recommended diet for migraine?)")
    if st.button("Search & Answer") and query.strip():
        if SKLEARN_OK and corpus_texts:
            vec = TfidfVectorizer(max_features=5000, stop_words='english')
            X = vec.fit_transform(corpus_texts)
            qv = vec.transform([query])
            sims = cosine_similarity(qv, X)[0]
            top_idx = np.argsort(sims)[::-1][:5]
            snippets = [corpus_texts[i][:600] for i in top_idx]
            answer = (
                "Here are the most relevant pieces of information I found:\n\n" +
                "\n\n".join([f"• {textwrap.shorten(s, width=250, placeholder='…')}" for s in snippets]) +
                "\n\n(Always consult a professional.)"
            )
        else:
            answer = "Not enough data or sklearn unavailable."
        st.text_area("Answer", answer, height=220)

st.markdown("---")
st.caption("Built with ❤️ — OAuth optional, DB-backed patients & history, RAG-lite assistant, professional dashboard UI.")
