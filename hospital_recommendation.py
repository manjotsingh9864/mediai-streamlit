import streamlit as st
import pandas as pd
import numpy as np
from geopy.distance import geodesic
from geopy.geocoders import Nominatim
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

import re

# Union Territories require special handling in Govt datasets
UNION_TERRITORIES = [
    "Chandigarh",
    "Delhi",
    "Puducherry",
    "Dadra and Nagar Haveli",
    "Daman and Diu",
    "Jammu and Kashmir",
    "Ladakh",
    "Lakshadweep",
    "Andaman and Nicobar Islands"
]

# Disease to specialty mapping (expanded)
DISEASE_SPECIALTY_MAP = {
    # Cardiovascular
    "Hypertension": "Cardiology",
    "Heart Attack": "Cardiology",
    "Angina": "Cardiology",
    "Arrhythmia": "Cardiology",
    "Heart Disease": "Cardiology",
    # Endocrine
    "Diabetes": "Endocrinology",
    "Thyroid Disorder": "Endocrinology",
    "Hypothyroidism": "Endocrinology",
    "Hyperthyroidism": "Endocrinology",
    # Respiratory
    "Asthma": "Pulmonology",
    "Bronchial Asthma": "Pulmonology",
    "COPD": "Pulmonology",
    "Pneumonia": "Pulmonology",
    "Tuberculosis": "Pulmonology",
    # Gastrointestinal
    "GERD": "Gastroenterology",
    "Peptic Ulcer": "Gastroenterology",
    "IBS": "Gastroenterology",
    "Crohn's Disease": "Gastroenterology",
    "Hepatitis": "Gastroenterology",
    "Jaundice": "Hepatology",
    "Liver Disease": "Hepatology",
    # Neurological
    "Migraine": "Neurology",
    "Stroke": "Neurology",
    "Epilepsy": "Neurology",
    "Parkinson's": "Neurology",
    "Alzheimer's": "Neurology",
    # Dermatological
    "Acne": "Dermatology",
    "Eczema": "Dermatology",
    "Psoriasis": "Dermatology",
    "Skin Allergy": "Dermatology",
    # Orthopedic
    "Arthritis": "Orthopedics",
    "Back Pain": "Orthopedics",
    "Fracture": "Orthopedics",
    "Joint Pain": "Orthopedics",
    # General
    "Common Cold": "General Medicine",
    "Fever": "General Medicine",
    "Malaria": "Internal Medicine",
    "Dengue": "Internal Medicine",
    # Others
    "Cancer": "Oncology",
    "Kidney Disease": "Nephrology",
    "Pregnancy": "Obstetrics and Gynaecology",
    "Eye Problems": "Ophthalmology",
    "Ear Problems": "Otorhinolaryngology",
    "Depression": "Psychiatry",
    "Anxiety": "Psychiatry"
}
# Symptom to specialty mapping
SYMPTOM_SPECIALTY_MAP = {
    "chest pain": "Cardiology",
    "shortness of breath": "Pulmonology",
    "headache": "Neurology",
    "stomach pain": "Gastroenterology",
    "joint pain": "Orthopedics",
    "skin rash": "Dermatology",
    "frequent urination": "Endocrinology",
    "fatigue": "Internal Medicine",
    "cough": "Pulmonology",
    "fever": "General Medicine",
    "nausea": "Gastroenterology",
    "dizziness": "Neurology",
    "back pain": "Orthopedics",
    "anxiety": "Psychiatry",
    "depression": "Psychiatry",
    "blurred vision": "Ophthalmology",
    "hearing loss": "Otorhinolaryngology"
}

@st.cache_data(ttl=3600)
def load_hospital_data():
    """Load and preprocess hospital data"""
    try:
        df = pd.read_csv("hospital_directory.csv")
        # Clean and preprocess
        df = df.replace('0', np.nan)
        df['Number_Doctor'] = pd.to_numeric(df['Number_Doctor'], errors='coerce').fillna(0).astype(int)
        df['Total_Num_Beds'] = pd.to_numeric(df['Total_Num_Beds'], errors='coerce').fillna(0).astype(int)
        # Parse coordinates
        coords = df['Location_Coordinates'].str.split(',', expand=True)
        df['Latitude'] = pd.to_numeric(coords[0], errors='coerce')
        df['Longitude'] = pd.to_numeric(coords[1], errors='coerce')
        # Clean specialties
        df['Discipline_Systems_of_Medicine'] = df['Discipline_Systems_of_Medicine'].fillna('General Medicine')
        df['Specialties'] = df['Specialties'].fillna('')
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def geocode_location(location_str):
    """Get coordinates for a location"""
    try:
        geolocator = Nominatim(user_agent="smart_hospital_finder", timeout=10)
        location = geolocator.geocode(location_str)
        if location:
            return (location.latitude, location.longitude)
    except Exception as e:
        st.warning(f"Geocoding error: {e}")
    return None

def calculate_distance(user_coords, hospital_coords):
    """Calculate distance between user and hospital"""
    try:
        return geodesic(user_coords, hospital_coords).km
    except:
        return float('inf')

def match_specialty_from_symptoms(symptoms):
    """Match symptoms to medical specialties"""
    symptoms_lower = symptoms.lower()
    matched_specialties = []
    for symptom, specialty in SYMPTOM_SPECIALTY_MAP.items():
        if symptom in symptoms_lower:
            matched_specialties.append(specialty)
    return list(set(matched_specialties))

def filter_hospitals(df, specialty=None, location=None, max_distance=None, min_doctors=0, state=None):
    """Advanced hospital filtering"""
    filtered = df.copy()
    # Filter by specialty
    if specialty:
        mask = (
            filtered['Discipline_Systems_of_Medicine'].str.contains(specialty, case=False, na=False) |
            filtered['Specialties'].str.contains(specialty, case=False, na=False)
        )
        filtered = filtered[mask]
    # Filter by state
    if state and state != "All States":
        filtered = filtered[filtered['State'] == state]
    # Filter by minimum doctors
    if min_doctors > 0:
        filtered = filtered[filtered['Number_Doctor'] >= min_doctors]
    # Calculate distance if location provided
    if location:
        user_coords = geocode_location(location)
        if user_coords:
            filtered = filtered.dropna(subset=['Latitude', 'Longitude'])
            filtered['Distance_km'] = filtered.apply(
                lambda row: calculate_distance(user_coords, (row['Latitude'], row['Longitude'])),
                axis=1
            )
            filtered = filtered.sort_values('Distance_km')
            # Filter by max distance
            if max_distance:
                filtered = filtered[filtered['Distance_km'] <= max_distance]
    return filtered

def create_map(df, user_coords=None):
    """Create interactive map with hospital locations"""
    if df.empty:
        return None
    map_df = df.dropna(subset=['Latitude', 'Longitude']).copy()
    if map_df.empty:
        return None
    # Create hover text
    map_df['hover_text'] = map_df.apply(
        lambda row: f"<b>{row['Hospital_Name']}</b><br>" +
                f"Location: {row['Location']}, {row['State']}<br>" +
                f"Doctors: {row['Number_Doctor']}<br>" +
                (f"Distance: {row['Distance_km']:.2f} km<br>" if 'Distance_km' in row and pd.notna(row['Distance_km']) else ""),
        axis=1
    )
    fig = go.Figure()
    # Add hospital markers
    fig.add_trace(go.Scattermapbox(
        lat=map_df['Latitude'],
        lon=map_df['Longitude'],
        mode='markers',
        marker=dict(size=12, color='red', opacity=0.7),
        text=map_df['hover_text'],
        hoverinfo='text',
        name='Hospitals'
    ))
    # Add user location if available
    if user_coords:
        fig.add_trace(go.Scattermapbox(
            lat=[user_coords[0]],
            lon=[user_coords[1]],
            mode='markers',
            marker=dict(size=15, color='blue', symbol='circle'),
            text=['Your Location'],
            hoverinfo='text',
            name='You'
        ))
    # Set map layout
    center_lat = user_coords[0] if user_coords else map_df['Latitude'].mean()
    center_lon = user_coords[1] if user_coords else map_df['Longitude'].mean()
    fig.update_layout(
        mapbox=dict(
            style="open-street-map",
            center=dict(lat=center_lat, lon=center_lon),
            zoom=10 if user_coords else 5
        ),
        showlegend=True,
        height=500,
        margin=dict(l=0, r=0, t=0, b=0)
    )
    return fig

def is_valid_value(val):
    if val is None:
        return False
    v = str(val).strip().lower()
    return v not in ["", "0", "nan", "none", "null"]

def display_hospital_card(row, index):
    """Display a beautiful hospital card"""
    with st.container():
        # Compose specialty/discipline info
        discipline = row.get('Discipline_Systems_of_Medicine', '')
        specialties = [s.strip() for s in str(row.get('Specialties', '')).split(',') if s.strip()]
        # Compose contact details (updated)
        tel = None  # ❌ stop using unreliable main phone
        mobile = row.get('Mobile_Number', '')
        emergency = row.get('Emergency_Num', '')
        ambulance = row.get('Ambulance_Phone_No', '')
        bloodbank = row.get('Bloodbank_Phone_No', '')
        website = row.get('Website', '')
        email_primary = row.get('Hospital_Primary_Email_Id', '')
        email_secondary = row.get('Hospital_Secondary_Email_Id', '')
        accreditation = row.get('Accreditation', '')
        est_year = row.get('Establised_Year', '')

        # Extra richer identifiers
        registration_no = row.get('Hospital_Regis_Number', '')
        ownership = row.get('Hospital_Care_Type', '')

        category = row.get('Hospital_Category', '')
        care_type = row.get('Hospital_Care_Type', '')
        facilities = row.get('Facilities', '')
        state = row.get('State', '')
        district = row.get('District', '') if 'District' in row else ''
        # Compose numbers
        num_doctor = int(row['Number_Doctor']) if row['Number_Doctor'] > 0 else None
        num_beds = int(row['Total_Num_Beds']) if row['Total_Num_Beds'] > 0 else None
        # Start card
        st.markdown(f"""
        <div class="hospital-card">
            <h3 style="color: #667eea; margin-bottom: 0.5rem;">🏥 {row['Hospital_Name']}</h3>
            <p style="color: #6b7280; margin-bottom: 1rem;">
                📍 {row['Location']}{', ' + state if state else ''}{', ' + district if district else ''}
            </p>
        """, unsafe_allow_html=True)
        # Info columns: Doctors, Beds, Distance
        col1, col2, col3 = st.columns(3)
        with col1:
            if num_doctor is not None:
                st.markdown(f"**👨‍⚕️ Doctors:** {num_doctor}")
        with col2:
            if 'Distance_km' in row and pd.notna(row['Distance_km']):
                st.markdown(f"**📏 Distance:** {row['Distance_km']:.2f} km")
        with col3:
            if num_beds is not None:
                st.markdown(f"**🛏️ Beds:** {num_beds}")
        # Discipline and specialties
        if discipline or specialties:
            disc_spec_html = ""
            if discipline:
                disc_spec_html += f'<span class="tag tag-specialty">{discipline}</span>'
            if specialties:
                # Show up to 5 specialties (distinct from discipline)
                for s in specialties[:5]:
                    if s and s != discipline:
                        disc_spec_html += f' <span class="tag tag-specialty">{s}</span>'
            st.markdown("**🩺 Specialties / Discipline:**", unsafe_allow_html=True)
            st.markdown(disc_spec_html, unsafe_allow_html=True)

        # RICH CONTACT & INFO SECTION
        info_html = ""

        if is_valid_value(mobile):
            info_html += f"<div>📱 <b>Mobile:</b> {mobile}</div>"

        if is_valid_value(emergency):
            info_html += f"<div>🚨 <b>Emergency:</b> {emergency}</div>"

        if is_valid_value(ambulance):
            info_html += f"<div>🚑 <b>Ambulance:</b> {ambulance}</div>"

        if is_valid_value(bloodbank):
            info_html += f"<div>🩸 <b>Blood Bank:</b> {bloodbank}</div>"

        if is_valid_value(email_primary):
            info_html += f"<div>✉️ <b>Email:</b> {email_primary}</div>"

        if is_valid_value(email_secondary):
            info_html += f"<div>✉️ <b>Alt Email:</b> {email_secondary}</div>"

        if is_valid_value(website):
            info_html += f'<div>🌐 <b>Website:</b> <a href="{website}" target="_blank">{website}</a></div>'

        # ✅ Show accreditation ONLY if meaningful
        if is_valid_value(accreditation) and str(accreditation) not in ["0", "0.0"]:
            info_html += f"<div>🏅 <b>Accreditation:</b> {accreditation}</div>"

        if is_valid_value(category):
            info_html += f"<div>🏷️ <b>Category:</b> {category}</div>"

        if is_valid_value(care_type):
            info_html += f"<div>🏥 <b>Care Type:</b> {care_type}</div>"

        # ✅ Show establishment year ONLY if realistic
        try:
            year_int = int(float(est_year))
            if year_int > 1800:
                info_html += f"<div>📅 <b>Established:</b> {year_int}</div>"
        except:
            pass

        # ✅ Alternative trusted identifiers (instead of bad values)
        if is_valid_value(registration_no):
            info_html += f"<div>🆔 <b>Hospital Registration No:</b> {registration_no}</div>"

        if is_valid_value(ownership):
            info_html += f"<div>🏥 <b>Ownership / Care Type:</b> {ownership}</div>"

        if info_html:
            st.markdown("<br/>**Hospital Information:**", unsafe_allow_html=True)
            st.markdown(info_html, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)


# --- STEP 1: State Healthcare Overview Helper ---
def display_state_overview(df, selected_state):
    """Display rich, data-driven State / UT overview"""
    if df.empty or selected_state == "All States":
        return

    st.markdown(f"## 🩺 {selected_state} – Healthcare Overview")

    # --- KPIs ---
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("🏥 Hospitals", len(df))
    with c2:
        doctors = int(df['Number_Doctor'].sum())
        st.metric("👨‍⚕️ Doctors", doctors if doctors > 0 else "Data NA")
    with c3:
        beds = int(df['Total_Num_Beds'].sum())
        st.metric("🛏️ Beds", beds if beds > 0 else "Data NA")
    with c4:
        systems = (
            df['Discipline_Systems_of_Medicine']
            .dropna()
            .str.split(',')
            .explode()
            .str.strip()
            .nunique()
        )
        st.metric("🩺 Systems", systems)

    # --- District breakdown ---
    if 'District' in df.columns:
        district_counts = (
            df['District']
            .dropna()
            .value_counts()
            .head(10)
        )
        if not district_counts.empty:
            st.subheader("📍 Hospital Distribution")
            for d, c in district_counts.items():
                st.write(f"• **{d}** – {c} hospitals")

    # --- Major systems of medicine ---
    systems_breakdown = (
        df['Discipline_Systems_of_Medicine']
        .dropna()
        .str.split(',')
        .explode()
        .str.strip()
        .value_counts()
        .head(6)
    )
    if not systems_breakdown.empty:
        st.subheader("🩺 Major Systems of Medicine")
        for s, c in systems_breakdown.items():
            st.write(f"• **{s}** – {c} hospitals")

    # --- Facilities overview ---
    if 'Facilities' in df.columns:
        facilities = (
            df['Facilities']
            .dropna()
            .str.split(',')
            .explode()
            .str.strip()
            .value_counts()
            .head(8)
        )
        if not facilities.empty:
            st.subheader("🏥 Common Facilities Available")
            for f, _ in facilities.items():
                st.write(f"• {f}")

    st.markdown("---")

def render_hospital_recommendation():
    # Custom CSS for better UI
    st.markdown("""
    <style>
        .main-header {
            font-size: 3rem;
            font-weight: 700;
            text-align: center;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
        }
        .sub-header {
            text-align: center;
            color: #6b7280;
            font-size: 1.1rem;
            margin-bottom: 2rem;
        }
        .metric-card {
            background: #e0f7fa !important;
            padding: 1.5rem;
            border-radius: 10px;
            color: #006064 !important;
            text-align: center;
        }
        /* Style for st.metric blocks */
        div[data-testid="stMetric"] {
            background: #e0f7fa !important;
            color: #006064 !important;
            border-radius: 10px;
            padding: 1.2rem 0.5rem 1.2rem 0.5rem;
            margin-bottom: 0.5rem;
            box-shadow: 0 2px 6px rgba(0,0,0,0.05);
            border: 1px solid #b2ebf2;
        }
        div[data-testid="stMetric"] label, div[data-testid="stMetric"] p, div[data-testid="stMetric"] span, div[data-testid="stMetric"] div {
            color: #006064 !important;
        }
        .hospital-card {
            border: 2px solid #e5e7eb;
            border-radius: 10px;
            padding: 1.5rem;
            margin: 1rem 0;
            background: white;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            transition: transform 0.2s;
        }
        .hospital-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 8px 12px rgba(0,0,0,0.15);
        }
        .tag {
            display: inline-block;
            padding: 0.3rem 0.8rem;
            margin: 0.2rem;
            border-radius: 20px;
            font-size: 0.85rem;
            font-weight: 600;
        }
        .tag-specialty {
            background: #dbeafe;
            color: #1e40af;
        }
        .tag-distance {
            background: #dcfce7;
            color: #15803d;
        }
        .stAlert {
            border-radius: 10px;
        }
        /* Light feature boxes */
        .light-feature-box {
            background: #f8fafc;
            border-radius: 14px;
            padding: 1.4rem;
            text-align: center;
            color: #334155;
            box-shadow: 0 6px 18px rgba(0,0,0,0.06);
            border-left: 5px solid #93c5fd;
            transition: all 0.25s ease;
        }
        .light-feature-box:hover {
            transform: translateY(-4px);
            box-shadow: 0 10px 26px rgba(0,0,0,0.1);
        }
        .light-feature-title {
            font-size: 1.2rem;
            font-weight: 700;
            color: #1e3a8a;
            margin-bottom: 0.6rem;
        }
        .light-feature-text {
            font-size: 0.95rem;
            color: #475569;
        }
    </style>
    """, unsafe_allow_html=True)

    # Header centered at the top
    st.markdown('<h1 class="main-header">🏥 Smart Hospital Finder</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-Powered Healthcare Navigation | Find the Right Hospital & Doctor Instantly</p>', unsafe_allow_html=True)

    # Load data
    df = load_hospital_data()
    if df.empty:
        st.error("Unable to load hospital data. Please ensure 'hospital_directory.csv' is in the same directory.")
        return

    # Layout: left (main), right (filters)
    main_col, filter_col = st.columns([2, 1], gap="large")

    # Filters on the right
    with filter_col:
        st.markdown("### 🔍 Search Filters")
        selected_disease = st.selectbox(
            "Select Medical Condition",
            ["None"] + sorted(list(DISEASE_SPECIALTY_MAP.keys())),
            help="Choose your medical condition for specialized recommendations"
        )
        symptoms = st.text_area(
            "Describe Your Symptoms (Optional)",
            placeholder="e.g., chest pain, difficulty breathing, fever...",
            help="Enter symptoms to get better recommendations"
        )
        location = st.text_input(
            "📍 Your Location",
            placeholder="e.g., Mumbai, Maharashtra",
            help="Enter your city or state to find nearby hospitals"
        )
        with st.expander("⚙️ Advanced Filters"):
            states = ["All States"] + sorted(df['State'].dropna().unique().tolist())
            selected_state = st.selectbox("Filter by State", states, key="state_filter")
            max_distance = st.slider(
                "Maximum Distance (km)",
                min_value=5,
                max_value=100,
                value=50,
                step=5,
                disabled=not location
            )
            min_doctors = st.slider(
                "Minimum Number of Doctors",
                min_value=0,
                max_value=50,
                value=0,
                step=1
            )
        search_button = st.button("🔎 Find Hospitals", type="primary", use_container_width=True)

    # Main content on the left
    with main_col:
        if search_button:
            with st.spinner("🔍 Finding the best hospitals for you..."):
                # Determine specialty
                specialty = None
                if selected_disease != "None":
                    specialty = DISEASE_SPECIALTY_MAP[selected_disease]
                    st.info(f"🔍 Searching for hospitals specializing in **{specialty}** for **{selected_disease}**")
                # Match symptoms
                if symptoms:
                    symptom_specialties = match_specialty_from_symptoms(symptoms)
                    if symptom_specialties:
                        st.info(f"💡 Based on symptoms, recommended specialties: **{', '.join(symptom_specialties)}**")
                        if not specialty:
                            specialty = symptom_specialties[0]
                # ✅ Smart state / UT filtering
                if selected_state in UNION_TERRITORIES:
                    filtered_df = df[
                        (df["State"] == selected_state) |
                        (df.get("District", "") == selected_state)
                    ]
                elif selected_state != "All States":
                    filtered_df = df[df["State"] == selected_state]
                else:
                    filtered_df = df.copy()

                # Apply medical filters AFTER state handling
                filtered_df = filter_hospitals(
                    filtered_df,
                    specialty=specialty,
                    location=location,
                    max_distance=max_distance if location else None,
                    min_doctors=min_doctors,
                    state=None
                )
                if filtered_df.empty:
                    st.warning("⚠️ No hospitals found matching your criteria. Try adjusting filters.")
                    return
                # ✅ Show State Healthcare Overview when browsing by state
                if selected_state != "All States" and not location:
                    display_state_overview(filtered_df, selected_state)
                # Display metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("🏥 Hospitals Found", len(filtered_df))
                with col2:
                    st.metric("👨‍⚕️ Total Doctors", int(filtered_df['Number_Doctor'].sum()))
                with col3:
                    if 'Distance_km' in filtered_df.columns:
                        st.metric("📏 Nearest Hospital", f"{filtered_df['Distance_km'].min():.2f} km")
                    else:
                        st.metric("🗺️ States Covered", filtered_df['State'].nunique())
                with col4:
                    st.metric("🛏️ Total Beds", int(filtered_df['Total_Num_Beds'].sum()))
                # Tabs for different views
                tab1, tab2, tab3 = st.tabs(["🗺️ Map View", "📋 List View", "📊 Analytics"])
                with tab1:
                    st.subheader("🗺️ Hospital Location Overview")
                    user_coords = geocode_location(location) if location else None
                    map_fig = create_map(filtered_df.head(100), user_coords)
                    if map_fig:
                        st.plotly_chart(map_fig, use_container_width=True)
                with tab2:
                    st.subheader("Hospital Details")
                    display_count = min(10, len(filtered_df))
                    st.write(f"Showing top {display_count} hospitals:")
                    for idx, (_, row) in enumerate(filtered_df.head(display_count).iterrows(), 1):
                        display_hospital_card(row, idx)
                    # Download option
                    csv = filtered_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Full Results (CSV)",
                        data=csv,
                        file_name=f"hospitals_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                with tab3:
                    st.subheader("Healthcare Analytics")
                    col1, col2 = st.columns(2)
                    with col1:
                        # Doctors distribution
                        if filtered_df['Number_Doctor'].sum() > 0:
                            top_hospitals = filtered_df.nlargest(10, 'Number_Doctor')[['Hospital_Name', 'Number_Doctor']]
                            fig = px.bar(
                                top_hospitals,
                                x='Number_Doctor',
                                y='Hospital_Name',
                                orientation='h',
                                title="Top 10 Hospitals by Doctor Count",
                                color='Number_Doctor',
                                color_continuous_scale='Viridis'
                            )
                            fig.update_layout(showlegend=False)
                            st.plotly_chart(fig, use_container_width=True)
                    with col2:
                        # State-wise distribution
                        state_dist = filtered_df['State'].value_counts().head(10)
                        fig = px.pie(
                            values=state_dist.values,
                            names=state_dist.index,
                            title="Hospitals by State (Top 10)"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    # Distance distribution
                    if 'Distance_km' in filtered_df.columns:
                        fig = px.histogram(
                            filtered_df,
                            x='Distance_km',
                            nbins=20,
                            title="Distance Distribution",
                            labels={'Distance_km': 'Distance (km)'}
                        )
                        st.plotly_chart(fig, use_container_width=True)
        else:
            # Welcome screen
            st.info("👉 Use the filters on the right to search for hospitals based on your medical needs and location")
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("""
                <div class="light-feature-box">
                    <div class="light-feature-title">🎯 Smart Search</div>
                    <div class="light-feature-text">
                        Our AI matches your symptoms and conditions with the right medical specialists
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                st.markdown("""
                <div class="light-feature-box">
                    <div class="light-feature-title">📍 Location-Based</div>
                    <div class="light-feature-text">
                        Find hospitals near you with distance calculations and interactive maps
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with col3:
                st.markdown("""
                <div class="light-feature-box">
                    <div class="light-feature-title">📊 Detailed Info</div>
                    <div class="light-feature-text">
                        View doctor count, specialties, contact details, and more
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # Statistics
            st.subheader("📈 Database Statistics")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Hospitals", len(df))
            with col2:
                st.metric("Total Doctors", int(df['Number_Doctor'].sum()))
            with col3:
                st.metric("States Covered", df['State'].nunique())
            with col4:
                st.metric("Total Beds", int(df['Total_Num_Beds'].sum()))

if __name__ == "__main__":
    render_hospital_recommendation()