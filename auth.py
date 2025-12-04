def init_user_store():
    if "users" not in st.session_state:
        st.session_state.users = {}
import streamlit as st
import streamlit.components.v1 as components
import base64
from pathlib import Path

def get_base64_image(image_path):
    """Convert image to base64 for background using absolute path based on this file"""
    try:
        img_path = Path(__file__).parent / image_path
        if not img_path.is_file():
            return None
        with open(img_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except Exception:
        return None

def render_login():
    """Render premium animated glassmorphism login page"""
    
    init_user_store()
    if "auth_page" not in st.session_state:
        st.session_state.auth_page = "login"

    bg_image = get_base64_image("a.png")
    
    if bg_image:
        bg_style = f'background-image: url("data:image/png;base64,{bg_image}");'
    else:
        # Fallback gradient background if image is missing
        bg_style = "background: radial-gradient(circle at top, #0f172a, #020617);"
    
    # Premium CSS with animations and exact reference design
    st.markdown(f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
        
        /* Remove all Streamlit branding */
        #MainMenu {{visibility: hidden;}}
        footer {{visibility: hidden;}}
        header {{visibility: hidden;}}
        .stDeployButton {{visibility: hidden;}}
        
        * {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }}
        
        /* Premium full-screen background */
        .stApp {{
            {bg_style}
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        
        /* Dark overlay for contrast */
        .stApp::before {{
            content: '';
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: linear-gradient(135deg, rgba(0, 0, 0, 0.4) 0%, rgba(0, 20, 40, 0.5) 100%);
            z-index: 0;
            pointer-events: none;
        }}
        
        /* Remove default padding */
        .block-container {{
            padding: 0 !important;
            max-width: 100% !important;
            position: relative;
            z-index: 1;
        }}
        
        /* Animated floating particles background */
        @keyframes float {{
            0%, 100% {{ transform: translateY(0px) rotate(0deg); }}
            50% {{ transform: translateY(-20px) rotate(180deg); }}
        }}
        
        @keyframes glow {{
            0%, 100% {{ box-shadow: 0 0 20px rgba(99, 102, 241, 0.3), 0 0 40px rgba(139, 92, 246, 0.2); }}
            50% {{ box-shadow: 0 0 30px rgba(99, 102, 241, 0.5), 0 0 60px rgba(139, 92, 246, 0.3); }}
        }}
        
        @keyframes slideIn {{
            from {{
                opacity: 0;
                transform: translateY(-30px);
            }}
            to {{
                opacity: 1;
                transform: translateY(0);
            }}
        }}
        
        @keyframes pulse {{
            0%, 100% {{ transform: scale(1); }}
            50% {{ transform: scale(1.05); }}
        }}
        
        /* Input labels - exact match */
        .stTextInput label, .stNumberInput label, .stSelectbox label {{
            color: #e2e8f0 !important;
            font-size: 13px !important;
            font-weight: 500 !important;
            margin-bottom: 8px !important;
            letter-spacing: 0.3px !important;
        }}
        
        /* Premium input fields with icons */
        .stTextInput > div > div > input,
        .stNumberInput > div > div > input,
        .stSelectbox > div > div > select {{
            background: rgba(30, 41, 59, 0.7) !important;
            border: 1px solid rgba(71, 85, 105, 0.5) !important;
            border-radius: 8px !important;
            color: #ffffff !important;
            padding: 12px 16px 12px 44px !important;
            font-size: 14px !important;
            height: 48px !important;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
            backdrop-filter: blur(10px) !important;
        }}
        
        .stTextInput > div > div > input:focus,
        .stSelectbox > div > div > select:focus {{
            border-color: rgba(99, 102, 241, 0.8) !important;
            background: rgba(30, 41, 59, 0.85) !important;
            box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.15), 0 4px 12px rgba(0, 0, 0, 0.2) !important;
            outline: none !important;
        }}
        
        .stTextInput > div > div > input::placeholder {{
            color: rgba(148, 163, 184, 0.5) !important;
            font-size: 13px !important;
        }}
        
        /* Icon indicators for inputs */
        .stTextInput:nth-of-type(1) > div > div::before {{
            content: '✉';
            position: absolute;
            left: 16px;
            top: 50%;
            transform: translateY(-50%);
            color: rgba(148, 163, 184, 0.6);
            font-size: 16px;
            z-index: 1;
        }}
        
        .stTextInput:nth-of-type(2) > div > div::before {{
            content: '🔒';
            position: absolute;
            left: 16px;
            top: 50%;
            transform: translateY(-50%);
            color: rgba(148, 163, 184, 0.6);
            font-size: 16px;
            z-index: 1;
        }}
        
        /* Checkbox styling */
        .stCheckbox {{
            margin: 12px 0 !important;
        }}
        
        .stCheckbox label {{
            color: #e2e8f0 !important;
            font-size: 13px !important;
            font-weight: 400 !important;
        }}
        
        .stCheckbox input[type="checkbox"] {{
            width: 18px !important;
            height: 18px !important;
            border-radius: 4px !important;
        }}
        
        /* Premium animated button */
        .stButton > button {{
            background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #a855f7 100%) !important;
            color: #ffffff !important;
            border: none !important;
            border-radius: 8px !important;
            padding: 14px 28px !important;
            font-size: 14px !important;
            font-weight: 600 !important;
            width: 100% !important;
            text-transform: uppercase !important;
            letter-spacing: 1px !important;
            height: 52px !important;
            box-shadow: 0 8px 20px rgba(99, 102, 241, 0.35), 0 2px 8px rgba(0, 0, 0, 0.2) !important;
            transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1) !important;
            position: relative !important;
            overflow: hidden !important;
            margin-top: 12px !important;
        }}
        
        .stButton > button::before {{
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
            transition: left 0.5s;
        }}
        
        .stButton > button:hover {{
            transform: translateY(-2px) !important;
            box-shadow: 0 12px 28px rgba(99, 102, 241, 0.45), 0 4px 12px rgba(0, 0, 0, 0.3) !important;
            background: linear-gradient(135deg, #5558e3 0%, #7c4ee8 50%, #9333ea 100%) !important;
        }}
        
        .stButton > button:hover::before {{
            left: 100%;
        }}
        
        .stButton > button:active {{
            transform: translateY(0px) !important;
        }}
        
        /* Tab styling - exact reference */
        .stTabs [data-baseweb="tab-list"] {{
            gap: 0 !important;
            background: transparent !important;
            border-bottom: 1px solid rgba(71, 85, 105, 0.3) !important;
            margin-bottom: 28px !important;
        }}
        
        .stTabs [data-baseweb="tab"] {{
            background: transparent !important;
            color: rgba(203, 213, 225, 0.5) !important;
            border: none !important;
            padding: 14px 36px !important;
            font-size: 14px !important;
            font-weight: 600 !important;
            border-bottom: 2px solid transparent !important;
            transition: all 0.3s ease !important;
        }}
        
        .stTabs [data-baseweb="tab"]:hover {{
            color: rgba(203, 213, 225, 0.8) !important;
        }}
        
        .stTabs [aria-selected="true"] {{
            background: transparent !important;
            color: #ffffff !important;
            border-bottom: 2px solid #6366f1 !important;
        }}
        
        /* Links */
        a {{
            color: #818cf8 !important;
            text-decoration: none !important;
            font-size: 13px !important;
            font-weight: 500 !important;
            transition: all 0.2s ease !important;
        }}
        
        a:hover {{
            color: #a5b4fc !important;
        }}
        
        /* Alert styling */
        .stSuccess, .stError, .stWarning {{
            background: rgba(30, 41, 59, 0.9) !important;
            border-radius: 8px !important;
            padding: 14px 18px !important;
            margin: 12px 0 !important;
            border: 1px solid rgba(71, 85, 105, 0.3) !important;
            backdrop-filter: blur(10px) !important;
            animation: slideIn 0.3s ease-out !important;
        }}
        
        /* Number input no icon */
        .stNumberInput > div > div > input {{
            padding-left: 16px !important;
        }}
        
        /* Selectbox styling */
        .stSelectbox > div > div {{
            background: rgba(30, 41, 59, 0.7) !important;
        }}
        
        /* Hide number input arrows for Chrome */
        input[type=number]::-webkit-inner-spin-button,
        input[type=number]::-webkit-outer-spin-button {{
            -webkit-appearance: none;
            margin: 0;
        }}
        
        /* Hide number input arrows for Firefox */
        input[type=number] {{
            -moz-appearance: textfield;
        }}
        
        @keyframes spin {{
            from {{ transform: rotate(0deg); }}
            to {{ transform: rotate(360deg); }}
        }}
        </style>
    """, unsafe_allow_html=True)
    
    # Vertical spacing
    st.markdown("<div style='height: 60px;'></div>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1.4, 1, 1.4])
    with col2:
        # Glass card open
        st.markdown("""
<div class="auth-card">
    <div style="text-align:center;margin-bottom:22px;">
        <div style="width:76px;height:76px;border-radius:50%;background:linear-gradient(135deg,#6366f1,#8b5cf6,#a855f7);margin:0 auto 10px;display:flex;align-items:center;justify-content:center;box-shadow:0 0 40px rgba(139,92,246,.45);">👤</div>
        <h2 style="margin:6px 0">MediAI</h2>
        <p style="opacity:.7;font-size:13px">AI-Powered Healthcare Insights</p>
    </div>
""", unsafe_allow_html=True)

        # Tabs
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            if st.button("Login", use_container_width=True, key="tab_login"):
                st.session_state.auth_page = "login"
                st.rerun()
        with col_t2:
            if st.button("Register", use_container_width=True, key="tab_register"):
                st.session_state.auth_page = "register"
                st.rerun()

        # Forms inside card
        if st.session_state.auth_page == "login":
            email = st.text_input("Email Address", placeholder="your.email@example.com")
            password = st.text_input("Password", type="password", placeholder="Enter your password")
            st.checkbox("Remember me")

            if st.button("SIGN IN", use_container_width=True):
                if email in st.session_state.users and st.session_state.users[email]["password"] == password:
                    st.session_state.logged_in = True
                    st.session_state.user_email = email
                    st.session_state.user_name = st.session_state.users[email]["name"]
                    st.session_state.user_age = st.session_state.users[email]["age"]
                    st.session_state.user_gender = st.session_state.users[email]["gender"]
                    st.rerun()
                else:
                    st.error("Invalid email or password")

        else:
            name = st.text_input("Full Name")
            email = st.text_input("Email Address")
            age = st.number_input("Age", min_value=1, max_value=120)
            gender = st.selectbox("Gender", ["Male", "Female", "Other"])
            password = st.text_input("Password", type="password")
            confirm = st.text_input("Confirm Password", type="password")

            if st.button("REGISTER ✅", use_container_width=True):
                if not (name and email and password and confirm):
                    st.error("Please fill all fields")
                elif password != confirm:
                    st.error("Passwords do not match")
                elif email in st.session_state.users:
                    st.error("User already exists")
                else:
                    st.session_state.users[email] = {
                        "name": name,
                        "age": age,
                        "gender": gender,
                        "password": password
                    }
                    st.success("✅ Registration successful")
                    st.session_state.auth_page = "login"
                    st.rerun()

        # Glass card close
        st.markdown("</div>", unsafe_allow_html=True)

# Main execution
if __name__ == "__main__":
    st.set_page_config(
        page_title="MediAI - Healthcare Login Portal",
        page_icon="⚕️",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    
    if not st.session_state.logged_in:
        render_login()
    else:
        # Premium dashboard after login
        st.markdown(f"""
            <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
            
            #MainMenu {{visibility: hidden;}}
            footer {{visibility: hidden;}}
            header {{visibility: hidden;}}
            .stDeployButton {{visibility: hidden;}}
            
            * {{
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            }}
            
            .stApp {{
                background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #334155 100%);
            }}
            
            .block-container {{
                padding: 0 !important;
                max-width: 100% !important;
            }}
            
            @keyframes glow {{
                0%, 100% {{ box-shadow: 0 0 20px rgba(99, 102, 241, 0.3), 0 0 40px rgba(139, 92, 246, 0.2); }}
                50% {{ box-shadow: 0 0 30px rgba(99, 102, 241, 0.5), 0 0 60px rgba(139, 92, 246, 0.3); }}
            }}
            
            @keyframes slideIn {{
                from {{
                    opacity: 0;
                    transform: translateY(-30px);
                }}
                to {{
                    opacity: 1;
                    transform: translateY(0);
                }}
            }}
            
            .stButton > button {{
                background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #a855f7 100%) !important;
                color: #ffffff !important;
                border: none !important;
                border-radius: 8px !important;
                padding: 14px 28px !important;
                font-size: 14px !important;
                font-weight: 600 !important;
                width: 100% !important;
                text-transform: uppercase !important;
                letter-spacing: 1px !important;
                height: 52px !important;
                box-shadow: 0 8px 20px rgba(99, 102, 241, 0.35) !important;
                transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1) !important;
            }}
            
            .stButton > button:hover {{
                transform: translateY(-2px) !important;
                box-shadow: 0 12px 28px rgba(99, 102, 241, 0.45) !important;
            }}
            </style>
        """, unsafe_allow_html=True)
        
        st.markdown("<div style='height: 60px;'></div>", unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 1.2, 1])
        with col2:
            st.markdown(f"""
                <div style="
                    background: linear-gradient(135deg, rgba(15, 23, 42, 0.92) 0%, rgba(30, 41, 59, 0.88) 100%);
                    backdrop-filter: blur(24px) saturate(180%);
                    border-radius: 20px;
                    padding: 56px 48px;
                    box-shadow: 0 25px 80px rgba(0, 0, 0, 0.6),
                                0 0 0 1px rgba(255, 255, 255, 0.08) inset,
                                0 8px 32px rgba(99, 102, 241, 0.15);
                    border: 1px solid rgba(71, 85, 105, 0.25);
                    text-align: center;
                    animation: slideIn 0.6s cubic-bezier(0.4, 0, 0.2, 1);
                ">
                    <div style="
                        width: 110px;
                        height: 110px;
                        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #a855f7 100%);
                        border-radius: 50%;
                        margin: 0 auto 28px auto;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        box-shadow: 0 12px 32px rgba(99, 102, 241, 0.4), 0 0 80px rgba(139, 92, 246, 0.3);
                        animation: glow 3s ease-in-out infinite;
                    ">
                        <svg width="56" height="56" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                            <path d="M12 12C14.21 12 16 10.21 16 8C16 5.79 14.21 4 12 4C9.79 4 8 5.79 8 8C8 10.21 9.79 12 12 12Z" fill="white"/>
                            <path d="M12 14C8.67 14 2 15.67 2 19V21H22V19C22 15.67 15.33 14 12 14Z" fill="white"/>
                        </svg>
                    </div>
                    
                    <h1 style="
                        background: linear-gradient(135deg, #38bdf8 0%, #818cf8 50%, #a78bfa 100%);
                        -webkit-background-clip: text;
                        -webkit-text-fill-color: transparent;
                        background-clip: text;
                        font-size: 38px;
                        font-weight: 800;
                        margin: 0 0 16px 0;
                        letter-spacing: 1px;
                    ">Welcome Back!</h1>
                    
                    <p style="
                        color: rgba(203, 213, 225, 0.8);
                        font-size: 18px;
                        margin: 0 0 32px 0;
                        font-weight: 500;
                    ">{st.session_state.user_name}</p>
                    
                    <div style="
                        background: rgba(51, 65, 85, 0.4);
                        border-radius: 12px;
                        padding: 28px;
                        margin: 0 0 32px 0;
                        border: 1px solid rgba(71, 85, 105, 0.3);
                        backdrop-filter: blur(10px);
                    ">
                        <div style="display: grid; gap: 16px;">
                            <div style="text-align: left;">
                                <p style="color: rgba(148, 163, 184, 0.8); font-size: 12px; margin: 0 0 4px 0; text-transform: uppercase; letter-spacing: 1px;">Email</p>
                                <p style="color: #e5e7eb; font-size: 15px; margin: 0; font-weight: 500;">{st.session_state.user_email}</p>
                            </div>
                            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 16px;">
                                <div style="text-align: left;">
                                    <p style="color: rgba(148, 163, 184, 0.8); font-size: 12px; margin: 0 0 4px 0; text-transform: uppercase; letter-spacing: 1px;">Age</p>
                                    <p style="color: #e5e7eb; font-size: 15px; margin: 0; font-weight: 500;">{st.session_state.user_age} years</p>
                                </div>
                                <div style="text-align: left;">
                                    <p style="color: rgba(148, 163, 184, 0.8); font-size: 12px; margin: 0 0 4px 0; text-transform: uppercase; letter-spacing: 1px;">Gender</p>
                                    <p style="color: #e5e7eb; font-size: 15px; margin: 0; font-weight: 500;">{st.session_state.user_gender}</p>
                                </div>
                            </div>
                        </div>
                    </div>
                    
                    <p style="
                        color: rgba(203, 213, 225, 0.6);
                        font-size: 14px;
                        margin: 0 0 28px 0;
                        line-height: 1.6;
                    ">
                        ✨ You've successfully logged into MediAI Healthcare Platform<br>
                        🏥 Your personalized dashboard is ready
                    </p>
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
            
            if st.button("🚪 LOGOUT", key="logout_btn", use_container_width=True):
                st.session_state.logged_in = False
                st.session_state.clear()
                st.success("✓ Successfully logged out!")
                st.rerun()