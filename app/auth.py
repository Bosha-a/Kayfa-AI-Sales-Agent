import os
import hashlib
import bcrypt
import certifi
import streamlit as st
from dotenv import load_dotenv
from pymongo import MongoClient, ASCENDING
import base64

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

st.secrets.get("MONGO_URI")

mongo_uri = st.secrets.get("MONGO_URI") or os.getenv("MONGODB_URI")
_client = MongoClient(mongo_uri, tls=True, tlsCAFile=certifi.where())
_coll = _client.Sales_Agent.users
_coll.create_index([("username", ASCENDING)], unique=True)
_kayfa_users = _client.kayfa.users

ROLE_PERMISSIONS = {
    "admin": {"chat": True, "crm": True},
    "sales": {"chat": False, "crm": True},
    "user":  {"chat": True, "crm": False},
}

ICON_B64 = None

with open("../images/kayfa.png", "rb") as f:
    LOGO_B64 = base64.b64encode(f.read()).decode()

with open("../images/kayfa_icon.png", "rb") as f:
    ICON_B64 = base64.b64encode(f.read()).decode()


def verify_user(username: str, password: str):
    for coll in (_coll, _kayfa_users):
        doc = coll.find_one({"username": username.strip()})
        if not doc:
            continue
        pw_field = doc.get("password_hash") or doc.get("password")
        if not pw_field:
            continue
        try:
            match = bcrypt.checkpw(password.encode(), pw_field.encode())
        except Exception:
            match = pw_field == hashlib.sha256(password.encode()).hexdigest()
        if match:
            role = doc.get("role", "user")
            name = doc.get("name") or doc.get("username", username)
            return {"username": doc["username"], "role": role, "name": name}
    return None

def login_form():
    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] {{ font-family: 'Inter', sans-serif !important; }}
    #MainMenu, .stAppDeployButton, footer {{ visibility: hidden; }}
    [data-testid="stHeader"] {{ height: 0 !important; min-height: 0 !important; background: transparent !important; }}
    [data-testid="stAppViewContainer"] {{
        background: linear-gradient(180deg, #1a1f3d 0%, #12152b 100%) !important;
    }}
    .login-title {{
        text-align: center; font-size: 20px; font-weight: 700;
        color: #ffffff; margin: 40px 0 4px;
    }}
    .login-sub {{
        text-align: center; font-size: 13px;
        color: rgba(200,204,224,0.5); margin-bottom: 24px;
    }}
    .login-box {{
        max-width: 340px; margin: 0 auto;
        padding: 32px 28px;
        background: rgba(255,255,255,0.04);
        border-radius: 20px;
        border: 1px solid rgba(255,255,255,0.06);
        box-shadow: 0 4px 24px rgba(0,0,0,0.2);
    }}
    .stTextInput input {{
        background: rgba(255,255,255,0.06) !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        border-radius: 10px !important;
        padding: 10px 14px !important;
        font-size: 14px !important;
        color: #e8ecf2 !important;
        caret-color: #e8ecf2 !important;
    }}
    .stTextInput input:focus {{
        border-color: #2D3BE0 !important;
        box-shadow: 0 0 0 2px rgba(45,59,224,0.2) !important;
    }}
    .stTextInput input::placeholder {{ color: rgba(200,204,224,0.35) !important; }}
    .stSelectbox div[data-baseweb="select"] > div {{
        background: rgba(255,255,255,0.06) !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        border-radius: 10px !important;
        color: #e8ecf2 !important;
    }}
    .stSelectbox div[data-baseweb="select"] > div:hover {{
        border-color: #2D3BE0 !important;
    }}
    .stSelectbox svg {{ fill: #e8ecf2 !important; }}
    .stButton button {{
        background: #2D3BE0 !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 10px !important;
        font-weight: 600 !important;
        font-size: 14px !important;
        width: 100% !important;
        transition: all 0.2s ease !important;
    }}
    .stButton button:hover {{
        background: #2533c9 !important;
        box-shadow: 0 4px 14px rgba(45,59,224,0.3) !important;
    }}
    .login-error {{ text-align: center; color: #ef4444; font-size: 13px; margin-top: 12px; }}
    .login-footer {{
        text-align: center; margin-top: 28px;
        font-size: 12px; color: rgba(200,204,224,0.25);
    }}
    .kayfa-corner {{
        position: fixed; top: 14px; right: 20px; z-index: 999;
        display: flex; align-items: center; padding: 7px 10px;
    }}
    .kayfa-corner img {{ height: 45px; display: block; filter: brightness(0) invert(1); }}
    </style>

    <div class="kayfa-corner">
        <img src="data:image/png;base64,{ICON_B64}" alt="Kayfa">
    </div>
    """, unsafe_allow_html=True)

    logo_html = f'<img src="data:image/png;base64,{LOGO_B64}" style="max-width:140px;filter:brightness(0) invert(1)">' if LOGO_B64 else ''
    st.markdown(f'<div style="text-align:center;margin-top:20px">{logo_html}</div>', unsafe_allow_html=True)
    st.markdown('<div class="login-title">Login</div>', unsafe_allow_html=True)
    st.markdown('<div class="login-sub">Choose user type and login</div>', unsafe_allow_html=True)

    with st.container():
        _, col, _ = st.columns([1, 1.5, 1])
        with col:
            username = st.text_input("", placeholder="Username", key="login_user")
            password = st.text_input("", type="password", placeholder="Password", key="login_pass")
            role_hint = st.selectbox("", ["admin", "sales", "user"], key="login_role", label_visibility="collapsed")

            role_map = {"admin": "admin", "sales": "sales", "user": "user"}
            role_val = role_map[role_hint]

            if st.button("Login", use_container_width=True):
                user = verify_user(username, password)
                if user and user["role"] == role_val:
                    st.session_state.authenticated = True
                    st.session_state.user = user
                    st.rerun()
                else:
                    st.markdown('<p class="login-error">❌ Username or password is incorrect</p>', unsafe_allow_html=True)

    st.markdown('<div class="login-footer" style="color:white;font-weight:bold;">'
    'Kayfa Agent &middot; All rights reserved'
    '</div>', unsafe_allow_html=True)

def require_auth(page_type: str):
    if "authenticated" not in st.session_state or not st.session_state.authenticated:
        login_form()
        st.stop()
    role = st.session_state.user["role"]
    perms = ROLE_PERMISSIONS.get(role, {})
    if not perms.get(page_type, False):
        st.error(f"Do not have permission for {role}")
        if st.button("Logout"):
            st.session_state.clear()
            st.rerun()
        st.stop()
