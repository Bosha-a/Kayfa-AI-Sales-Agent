import os
import hashlib
import bcrypt
import certifi
import streamlit as st
from dotenv import load_dotenv
from pymongo import MongoClient, ASCENDING
import base64

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

mongo_uri = st.secrets.get("MONGODB_URI") 
_client = MongoClient(mongo_uri, tls=True, tlsCAFile=certifi.where())
_coll = _client.kayfa.users
_coll.create_index([("username", ASCENDING)], unique=True)
_kayfa_users = _client.kayfa.users

ROLE_PERMISSIONS = {
    "admin": {"chat": False, "crm": True, "dashboard": True},
    "user":  {"chat": True, "crm": False, "dashboard": False},
}

ICON_B64 = None

COUNTRIES = {
    "Egypt (+20)": "+20",
    "Saudi Arabia (+966)": "+966",
    "UAE (+971)": "+971",
    "Kuwait (+965)": "+965",
    "Qatar (+974)": "+974",
    "Oman (+968)": "+968",
    "Bahrain (+973)": "+973",
    "Jordan (+962)": "+962",
    "Lebanon (+961)": "+961",
    "Palestine (+970)": "+970",
    "Syria (+963)": "+963",
    "Iraq (+964)": "+964",
    "Yemen (+967)": "+967",
    "Libya (+218)": "+218",
    "Tunisia (+216)": "+216",
    "Algeria (+213)": "+213",
    "Morocco (+212)": "+212",
    "Sudan (+249)": "+249",
    "Other": "",
}

with open("./images/kayfa.png", "rb") as f:
    LOGO_B64 = base64.b64encode(f.read()).decode()

with open("./images/kayfa_icon.png", "rb") as f:
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
            name = doc.get("name") or f"{doc.get('first_name', '')} {doc.get('last_name', '')}".strip() or doc.get("username", username)
            phone = doc.get("phone", "")
            country = doc.get("country", "")
            return {"username": doc["username"], "role": role, "name": name, "phone": phone, "country": country}
    return None


def signup_user(first_name, last_name, phone, username, password, country=""):
    if _coll.find_one({"username": username.strip()}):
        return False, "Username already exists"
    pw_hash = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
    doc = {
        "first_name": first_name.strip(),
        "last_name": last_name.strip(),
        "phone": phone.strip(),
        "username": username.strip(),
        "password_hash": pw_hash,
        "country": country.strip(),
        "role": "user",
        "name": f"{first_name.strip()} {last_name.strip()}",
    }
    _coll.insert_one(doc)
    return True, None


def render_login():
    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] {{ font-family: 'Inter', sans-serif !important; }}
    #MainMenu, .stAppDeployButton, footer {{ visibility: hidden; }}
    [data-testid="stHeader"] {{ height: 0 !important; min-height: 0 !important; background: transparent !important; }}
    [data-testid="stAppViewContainer"] {{
        background: linear-gradient(180deg, #1a1f3d 0%, #12152b 100%) !important;
    }}
    .auth-title {{
        text-align: center; font-size: 20px; font-weight: 700;
        color: #ffffff; margin: 40px 0 4px;
    }}
    .auth-sub {{
        text-align: center; font-size: 13px;
        color: rgba(200,204,224,0.5); margin-bottom: 24px;
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
    .auth-error {{ text-align: center; color: #ef4444; font-size: 13px; margin-top: 12px; }}
    .auth-success {{ text-align: center; color: #22c55e; font-size: 13px; margin-top: 12px; }}
    .auth-footer {{
        text-align: center; margin-top: 28px;
        font-size: 12px; color: rgba(200,204,224,0.25);
    }}
    .kayfa-corner {{
        position: fixed; top: 14px; right: 20px; z-index: 999;
        display: flex; align-items: center; padding: 7px 10px;
    }}
    .kayfa-corner img {{ height: 45px; display: block; filter: brightness(0) invert(1); }}
    .stSelectbox div[data-baseweb="select"] > div {{
        background: rgba(255,255,255,0.06) !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        border-radius: 10px !important;
        color: #e8ecf2 !important;
    }}
    .stSelectbox div[data-baseweb="select"] > div:hover {{ border-color: #2D3BE0 !important; }}
    .stSelectbox svg {{ fill: #e8ecf2 !important; }}
    </style>
    <div class="kayfa-corner">
        <img src="data:image/png;base64,{ICON_B64}" alt="Kayfa">
    </div>
    """, unsafe_allow_html=True)

    logo_html = f'<img src="data:image/png;base64,{LOGO_B64}" style="max-width:140px;filter:brightness(0) invert(1)">' if LOGO_B64 else ''
    st.markdown(f'<div style="text-align:center;margin-top:20px">{logo_html}</div>', unsafe_allow_html=True)

    if "auth_mode" not in st.session_state:
        st.session_state.auth_mode = "login"

    if st.session_state.auth_mode == "signup":
        st.markdown('<div class="auth-title">Sign Up</div>', unsafe_allow_html=True)
        st.markdown('<div class="auth-sub">Create a new account</div>', unsafe_allow_html=True)
        _, col, _ = st.columns([1, 1.5, 1])
        with col:
            first = st.text_input("First name", placeholder="First name", key="signup_first", label_visibility="collapsed")
            last = st.text_input("Last name", placeholder="Last name", key="signup_last", label_visibility="collapsed")
            country_label = st.selectbox("Country", list(COUNTRIES.keys()), key="signup_country", label_visibility="collapsed")
            country_code = COUNTRIES[country_label]
            code_hint = f"{country_code} " if country_code else ""
            phone_local = st.text_input("Phone", placeholder=f"{code_hint}Phone number", key="signup_phone", label_visibility="collapsed")
            username = st.text_input("Username", placeholder="Username", key="signup_user", label_visibility="collapsed")
            password = st.text_input("Password", type="password", placeholder="Password", key="signup_pass", label_visibility="collapsed")
            if st.button("Sign Up", use_container_width=True):
                full_phone = (country_code + phone_local.strip()) if phone_local.strip() else phone_local.strip()
                if not all([first, last, phone_local, username, password]):
                    st.markdown('<p class="auth-error">❌ All fields are required</p>', unsafe_allow_html=True)
                elif len(password) < 4:
                    st.markdown('<p class="auth-error">❌ Password must be at least 4 characters</p>', unsafe_allow_html=True)
                elif not full_phone.replace("+", "").replace(" ", "").isdigit():
                    st.markdown('<p class="auth-error">❌ Invalid phone number</p>', unsafe_allow_html=True)
                else:
                    ok, err = signup_user(first, last, full_phone, username, password, country=country_label)
                    if ok:
                        st.markdown('<p class="auth-success">✅ Account created! You can now log in.</p>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<p class="auth-error">❌ {err}</p>', unsafe_allow_html=True)
        if st.button("← Back to Login", key="goto_login"):
            st.session_state.auth_mode = "login"
            st.rerun()
        st.markdown('<div class="auth-footer">Kayfa Agent &middot; All rights reserved</div>', unsafe_allow_html=True)
        return

    st.markdown('<div class="auth-title">Login</div>', unsafe_allow_html=True)
    st.markdown('<div class="auth-sub">Enter your credentials</div>', unsafe_allow_html=True)
    _, col, _ = st.columns([1, 1.5, 1])
    with col:
        username = st.text_input("Username", placeholder="Username", key="login_user", label_visibility="collapsed")
        password = st.text_input("Password", type="password", placeholder="Password", key="login_pass", label_visibility="collapsed")
        if st.button("Login", use_container_width=True):
            user = verify_user(username, password)
            if user:
                st.session_state.authenticated = True
                st.session_state.user = user
                st.rerun()
            else:
                st.markdown('<p class="auth-error">❌ Username or password is incorrect</p>', unsafe_allow_html=True)

    if st.button("Create an account →", key="goto_signup"):
        st.session_state.auth_mode = "signup"
        st.rerun()

    st.markdown('<div class="auth-footer">Kayfa Agent &middot; All rights reserved</div>', unsafe_allow_html=True)
