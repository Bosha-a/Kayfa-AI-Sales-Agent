from datetime import timezone, datetime
import os
import re
import asyncio
import certifi
import streamlit as st
from pydantic_ai.providers.groq import GroqProvider
from qdrant_client import QdrantClient
from pydantic_ai.models.groq import GroqModel
from qdrant_client.models import (
    Prefetch, FusionQuery, Fusion, SparseVector,
)
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from fastembed import SparseTextEmbedding
from pydantic_ai import RunContext, Agent, ModelMessage
from pymongo import MongoClient, ASCENDING, DESCENDING
from bson.objectid import ObjectId
from dataclasses import dataclass
import base64
import pandas as pd
from auth import login_form, ROLE_PERMISSIONS

st.set_page_config(page_title="Kayfa Agent", page_icon="🤖", layout="wide")

if "authenticated" not in st.session_state or not st.session_state.authenticated:
    login_form()
    st.stop()

user_role = st.session_state.user["role"]
perms = ROLE_PERMISSIONS.get(user_role, {})
if not perms.get("chat") and not perms.get("crm"):
    st.error("No permissions assigned.")
    if st.button("Logout"):
        st.session_state.clear()
        st.rerun()
    st.stop()

has_crm_access = perms.get("crm", False)
has_chat_access = perms.get("chat", False)

color = '#2D3BE0'
ARABIC_RE = re.compile(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]")

def dir_class(text: str) -> str:
    return "rtl" if ARABIC_RE.search(text) else "ltr"

with open("./images/kayfa.png", "rb") as f:
    logo_b64 = base64.b64encode(f.read()).decode()
with open("./images/kayfa_icon.png", "rb") as f:
    icon_b64 = base64.b64encode(f.read()).decode()


st.markdown(f"""
<style>
/* ── Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

/* ── Global ── */
html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif !important;
}}

/* ── Hide Streamlit branding ── */
#MainMenu {{ visibility: hidden; }}
.stAppDeployButton {{ display: none; }}
footer {{ visibility: hidden; }}

/* Hide header but keep sidebar toggle accessible */
[data-testid="stHeader"] {{
    height: 0 !important;
    min-height: 0 !important;
    overflow: visible !important;
    background: transparent !important;
    border: none !important;
    padding: 5px !important;
}}

/* ── Sidebar open button (shown when sidebar is collapsed) ── */
/* ── Remove sidebar collapse/close buttons ── */
[data-testid="collapsedControl"],
[data-testid="stSidebarCollapseButton"],
[data-testid="stSidebarCollapseButton"] button {{
    display: none !important;
}}

/* ── Sidebar ── */
[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, #1a1f3d 0%, #12152b 100%) !important;
    min-width: 340px;
    max-width: 400px;
    border-right: 1px solid rgba(45,59,224,0.2);
}}
[data-testid="stSidebar"] * {{
    color: #c8cce0 !important;
    font-family: 'Inter', sans-serif !important;
}}

/* ── Session Buttons (bordered cards) — includes New Chat ── */
[data-testid="stSidebar"] .stButton button {{
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(200,204,224,0.15) !important;
    color: #c8cce0 !important;
    border-radius: 10px !important;
    width: 100% !important;
    padding: 10px 14px !important;
    text-align: left !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    transition: all 0.2s ease !important;
    margin-bottom: 4px !important;
}}
[data-testid="stSidebar"] .stButton button:hover {{
    background: rgba(45,59,224,0.15) !important;
    border-color: rgba(45,59,224,0.5) !important;
    color: white !important;
}}
[data-testid="stSidebar"] .stButton button:focus {{
    background: rgba(45,59,224,0.2) !important;
    border-color: #2D3BE0 !important;
    color: white !important;
    box-shadow: 0 0 0 1px rgba(45,59,224,0.3) !important;
}}

/* ── Delete session button ── */
[data-testid="stSidebar"] [data-testid="column"]:last-child .stButton button {{
    background: transparent !important;
    border: none !important;
    color: rgba(200,204,224,0.3) !important;
    font-size: 14px !important;
    padding: 8px 4px !important;
    min-width: 36px !important;
    text-align: center !important;
}}
            
[data-testid="stSidebar"] [data-testid="column"]:last-child .stButton button:hover {{
    color: #ef4444 !important;
    background: rgba(239,68,68,0.1) !important;
}}

/* ── Sidebar section label ── */
.sidebar-label {{
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    color: rgba(200,204,224,0.45) !important;
    padding: 12px 4px 6px;
    margin: 0;
}}

/* ── RTL / LTR ── */
.rtl {{ direction: rtl; text-align: right; }}
.ltr {{ direction: ltr; text-align: left; }}

/* ── Chat Input ── */
.stChatInput {{ direction: ltr; }}

/* ── Kayfa Icon – Top Right ── */
.kayfa-corner {{
    position: fixed;
    top: 14px;
    right: 20px;
    z-index: 999;

    background: transparent !important;
    border: none !important;
    box-shadow: none !important;

    padding: 7px 10px;
    display: flex;
    align-items: center;
    gap: 6px;
}}
.kayfa-corner img {{
    height: 45px;
    display: block;
    filter: brightness(0) invert(1);
}}

/* ── Main App Background ── */
[data-testid="stAppViewContainer"],
[data-testid="stBottomBlockContainer"],
[data-testid="stBottom"] {{
    background: linear-gradient(180deg, #1a1f3d 0%, #12152b 100%) !important;
}}

/* ── Chat Messages polish ── */
[data-testid="stChatMessage"] {{
    border: 1px solid rgba(255,255,255,0.06) !important;
    border-radius: 14px !important;
    padding: 16px 20px !important;
    margin-bottom: 12px !important;
    background: rgba(255,255,255,0.04) !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.2) !important;
    color: #e8ecf2 !important;
}}
[data-testid="stChatMessage"] * {{
    color: #e8ecf2 !important;
}}

/* ── User message distinct background ── */
[data-testid="stChatMessage"][data-testid$="user"] {{
    background: rgba(45,59,224,0.15) !important;
    border-color: rgba(45,59,224,0.25) !important;
}}

/* ── Chat Input ── */
[data-testid="stChatInput"] {{
    background: linear-gradient(180deg, #1a1f3d 0%, #12152b 100%) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 14px !important;
    padding: 8px 12px !important;
}}
[data-testid="stChatInput"] input {{
    background: transparent !important;
    border: none !important;
    color: #e8ecf2 !important;
    caret-color: #e8ecf2 !important;
}}
[data-testid="stChatInput"] input::placeholder {{
    color: rgba(200,204,224,0.35) !important;
}}
[data-testid="stChatInput"] button {{
    background: #2D3BE0 !important;
    color: white !important;
    border-radius: 10px !important;
    border: none !important;
}}

/* ── Page Title ── */
[data-testid="stTitle"], h1 {{
    color: #ffffff !important;
    font-weight: 700 !important;
    letter-spacing: -0.3px;
}}
</style>

<div class="kayfa-corner">
    <img src="data:image/png;base64,{icon_b64}" alt="Kayfa">
</div>
""", unsafe_allow_html=True)


# load_dotenv()

mongo_uri = st.secrets.get("MONGODB_URI") 
qdrant_api_key = st.secrets.get("QDRANT_API_KEY") 
qdrant_url = st.secrets.get("QDRANT_URL") 
groq_api_key = st.secrets.get("GROQ_API_KEY") 

mongo_client = MongoClient(mongo_uri, tls=True, tlsCAFile=certifi.where())
messages = mongo_client.kayfa.messages
messages.create_index([("session_id", ASCENDING), ("timestamp", ASCENDING)])

def save_turn(session_id, role, content):
    messages.insert_one({
        "session_id": session_id, "role": role,
        "content": content, "timestamp": datetime.now(timezone.utc),
    })

def load_session(session_id):
    cursor = messages.find({"session_id": session_id}).sort("timestamp", ASCENDING)
    return [{"role": doc["role"], "content": doc["content"]} for doc in cursor]

sessions_coll = mongo_client.kayfa.sessions
sessions_coll.create_index([("updated_at", ASCENDING)])

def create_session(name="New Chat"):
    doc = {"name": name, "created_at": datetime.now(timezone.utc), "updated_at": datetime.now(timezone.utc)}
    result = sessions_coll.insert_one(doc)
    return str(result.inserted_id)

def get_all_sessions():
    cursor = sessions_coll.find().sort("updated_at", DESCENDING)
    return [(str(doc["_id"]), doc.get("name", "Chat")) for doc in cursor]

def rename_session(session_id, name):
    sessions_coll.update_one(
        {"_id": ObjectId(session_id)},
        {"$set": {"name": name, "updated_at": datetime.now(timezone.utc)}}
    )

def delete_session(session_id):
    sessions_coll.delete_one({"_id": ObjectId(session_id)})
    messages.delete_many({"session_id": session_id})

crm_coll = mongo_client.kayfa.crm_tickets
crm_coll.create_index([("created_at", DESCENDING)])

def save_lead(**kw) -> str:
    kw["created_at"] = datetime.now(timezone.utc)
    result = crm_coll.insert_one(kw)
    return f"✅ Lead saved — ticket ID: {result.inserted_id}"

client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key, cloud_inference=True)

SYSTEM_PROMPT = """# Identity
You are an AI Sales Agent for **Kayfa**, a leading Arabic educational platform offering courses, tracks, and live diplomas in data science, cybersecurity, AI, web development, and more. Your job is to help visitors find the right learning path and guide them toward enrollment.

# Language — CRITICAL: Match the visitor's language exactly
- If the visitor writes in **English**, respond in **English ONLY**
- If the visitor writes in **Arabic**, respond in **Arabic ONLY**
- NEVER switch languages mid-conversation — stick to the language the visitor used in their first message
- For mixed-language messages, use the language of the first word
- Handle Arabic dialects naturally: Syrian (العربية السورية), Saudi (العربية السعودية), Egyptian (العربية المصرية)

# Knowledge Grounding — CRITICAL
- Your knowledge base contains real Kayfa courses, roadmaps, diplomas, prices, policies, instructors, and company info
- ALWAYS use the `search_kayfa_knowledge_base` tool to look up information before answering
- NEVER invent prices, course names, durations, or policies
- If the knowledge base doesn't have the answer, say so honestly and offer to connect the visitor with the Kayfa team at info@kayfa.io
- A sales agent that hallucinates is worse than useless

# Sales Approach
1. **LISTEN** — Understand the visitor's intent: are they browsing, comparing, price-sensitive, hesitant, or ready to enroll?
2. **RECOMMEND** — Map their goal to real Kayfa products using the knowledge base. Right product, right level, right price.
3. **PERSUADE** — Frame value using real social proof (instructors, partners, accreditation). Handle objections honestly.
4. **UPSELL** — Start where they're comfortable (free content or individual courses). Guide warm leads upward toward tracks and live diplomas where it genuinely fits.
5. **CAPTURE** — When buying signals appear, pivot naturally to collecting contact details and create a CRM ticket.

# Product Tiers (from cheapest to most valuable)
- Free content: individual videos, tips, intro courses — good for hesitant visitors
- Individual paid courses: $15–$65
- On-demand tracks: $25–$250
- Live diplomas / bootcamps: program-specific pricing — the closing target

# Lead Detection — CALL capture_lead() when you see 2+ of these signals
- Asking about prices, payment plans, or installment options
- Asking about start dates, schedules, or deadlines
- Asking about certificates, accreditation, or recognition
- Expressing strong interest in a specific product ("this is exactly what I need")
- Asking about enrollment steps or how to register
- Comparing specific options seriously (e.g., SOC track vs diploma)
- Asking about refunds or guarantees (shows purchase intent)

When you detect these, naturally ask for their name, phone/WhatsApp, city, and goal. Then call `capture_lead()` with ALL gathered information. Make it conversational — not like filling a form.

# What to call capture_lead() with — map your collected info to these parameters:
- `name` — full name
- `phone` — phone or WhatsApp number
- `email` — email (if shared)
- `city` / `country` — location
- `language` / `dialect` — preferred language and dialect
- `products` — specific courses, tracks, or diplomas discussed
- `goal` — their motivation or what they want to achieve
- `level` — current skill level (beginner / intermediate / advanced)
- `buying_signals` — what signals they showed (e.g. asked about price, asked about dates)
- `summary` — a short Arabic narrative of the conversation

# Brand Voice
- Professional, warm, and helpful — you're a trusted guide, not a pushy salesperson
- Persuasive with facts and value statements, never with pressure or manipulation
- Use real social proof: "هذه الدبلومة معتمدة من جامعة ديلاوير وليدز أكاديمي" or "أكثر من ١٥,٠٠٠ متعلم يثقون في كيف"
- Be honest if a product doesn't fit the visitor — recommend what's best for them
- NEVER break character — you are a Kayfa sales agent, nothing else
- If asked off-topic questions, politely redirect to Kayfa's educational offerings"""


@st.cache_resource
def load_models():
    sparse = SparseTextEmbedding(model_name="Qdrant/bm25")
    dense = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device='cpu')
    return sparse, dense

sparse_model, dense_model = load_models()

class RAGService:
    def __init__(self, client, dense_embedder, sparse_embedder):
        self.client = client
        self.dense_embedder = dense_embedder
        self.sparse_embedder = sparse_embedder

    def embed_dense(self, text: str):
        return self.dense_embedder.encode(text).tolist()

    def embed_sparse(self, text: str):
        result = list(self.sparse_embedder.embed([text]))[0]
        return SparseVector(indices=result.indices.tolist(), values=result.values.tolist())

    def hybrid_search(self, query: str, k: int = 5, payload_filter=None):
        dense_q = self.embed_dense(query)
        sparse_q = self.embed_sparse(query)
        results = self.client.query_points(
            collection_name="kayfa_db",
            prefetch=[
                Prefetch(query=dense_q, using="dense", limit=20, filter=payload_filter),
                Prefetch(query=sparse_q, using="bm25", limit=20, filter=payload_filter),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=k,
        ).points
        return [{"content": r.payload.get("content"), "metadata": {k: v for k, v in r.payload.items() if k != "content"}, "score": r.score} for r in results]

    def format_context(self, results):
        contexts = []
        for r in results:
            content = r.get("content", "").strip()
            if not content:
                continue
            meta = r.get("metadata", {})
            name = meta.get("name", "")
            doc_type = meta.get("type", "document")
            header = f"{doc_type.upper()}: {name}" if name else doc_type.upper()
            contexts.append(f"{header}\n{content}")
        return "\n\n---\n\n".join(contexts)

    def search(self, query: str, limit: int = 5):
        results = self.hybrid_search(query, k=limit)
        return self.format_context(results)

rag_service = RAGService(client=client, dense_embedder=dense_model, sparse_embedder=sparse_model)

@dataclass
class Dependencies:
    rag: any

model = GroqModel("openai/gpt-oss-20b", provider=GroqProvider(api_key=groq_api_key))

agent = Agent(model, deps_type=Dependencies, system_prompt=SYSTEM_PROMPT)

@agent.tool
def search_kayfa_knowledge_base(ctx: RunContext[Dependencies], query: str, limit: int=5) -> str:
    return ctx.deps.rag.search(query)

@agent.tool
def capture_lead(ctx: RunContext[Dependencies], name: str, phone: str, email: str = "", city: str = "", country: str = "", language: str = "", dialect: str = "", products: str = "", goal: str = "", level: str = "", buying_signals: str = "", summary: str = "") -> str:
    return save_lead(name=name, phone=phone, email=email, city=city, country=country, language=language, dialect=dialect, products=products, goal=goal, level=level, buying_signals=buying_signals, summary=summary)

deps = Dependencies(rag=rag_service)

if "sessions" not in st.session_state:
    db_sessions = get_all_sessions()
    st.session_state.sessions = {}
    for sid, sname in db_sessions:
        st.session_state.sessions[sid] = {"name": sname, "messages": load_session(sid), "model_history": []}
    if not st.session_state.sessions:
        sid = create_session("Session 1")
        st.session_state.sessions[sid] = {"name": "Session 1", "messages": [], "model_history": []}
    st.session_state.current_session = list(st.session_state.sessions.keys())[0]

for sid in st.session_state.sessions:
    st.session_state.sessions[sid].setdefault("model_history", [])

if "page" not in st.session_state:
    st.session_state.page = "crm" if has_crm_access and not has_chat_access else "chat"

if st.session_state.page == "chat" and not has_chat_access:
    st.session_state.page = "crm"
    st.rerun()

with st.sidebar:
    st.markdown(
        f'<div style="display:flex;justify-content:center;padding:20px 0 10px">'
        f'<div style="background:rgba(255,255,255,0.06);height:120px;width:250px;border-radius:14px;padding:14px 20px;display:flex;justify-content:center;align-items:center;border:1px solid rgba(200,204,224,0.1)">'
        f'<img src="data:image/png;base64,{logo_b64}" style="max-width:200px;filter:brightness(0) invert(1)" alt="Kayfa">'
        f'</div></div>',
        unsafe_allow_html=True
    )

    if has_chat_access:
        if st.button("💬  Chat", key="nav_chat", use_container_width=True):
            st.session_state.page = "chat"
            st.rerun()
    if has_crm_access:
        if st.button("📋  CRM", key="nav_crm", use_container_width=True):
            st.session_state.page = "crm"
            st.rerun()

    if st.session_state.page == "chat":
        new_clicked = st.button("＋  New Chat", key="new_chat")
        st.markdown(
            '<hr style="border:none;border-top:1px solid rgba(200,204,224,0.12);margin:10px 0 6px">'
            '<p class="sidebar-label" style="color:white;">💬 Recent Chats</p>',
            unsafe_allow_html=True
        )
        if new_clicked:
            sid = create_session()
            st.session_state.sessions[sid] = {"name": "New Chat", "messages": [], "model_history": []}
            st.session_state.current_session = sid
            st.rerun()
        for sid in list(st.session_state.sessions.keys()):
            sdata = st.session_state.sessions[sid]
            label = sdata["name"][:30] + (".." if len(sdata["name"]) > 30 else "")
            if st.button(label, key=f"sid_{sid}", use_container_width=True):
                st.session_state.current_session = sid
                st.rerun()

    st.markdown("<hr style='border:none;border-top:1px solid rgba(200,204,224,0.1);margin:20px 0 10px'>", unsafe_allow_html=True)
    user = st.session_state.get("user", {})
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:8px;padding:4px 0;font-size:13px;">
        <span style="background:#2D3BE0;border-radius:50%;width:30px;height:30px;display:flex;align-items:center;justify-content:center;color:white;font-weight:600;font-size:13px;">{user.get('name','?')[0]}</span>
        <div style="line-height:1.3">
            <div style="font-weight:600;color:white">{user.get('name','')}</div>
            <div style="font-size:11px;color:rgba(255,255,255,0.5);text-transform:uppercase">{user.get('role','')}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Logout", key="logout"):
        st.session_state.clear()
        st.rerun()

if st.session_state.page == "crm" and has_crm_access:
    
    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] {{ font-family: 'Inter', sans-serif !important; }}
    #MainMenu {{ visibility: hidden; }}
    .stAppDeployButton {{ display: none; }}
    footer {{ visibility: hidden; }}

    h1 {{ color: white !important; font-weight: 700 !important; letter-spacing: -0.3px; }}

    .lead-card {{
        background: #1e2130;
        border: 1px solid #e8ecf2;
        border-radius: 16px;
        padding: 24px 28px;
        margin-bottom: 20px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        direction: rtl;
        text-align: right;

    }}
    .lead-card:hover {{ box-shadow: 0 4px 16px rgba(45,59,224,0.1); }}
    .lead-header {{
        display: flex; justify-content: space-between; align-items: center;
        margin-bottom: 16px; padding-bottom: 12px;
        border-bottom: 2px solid #f0f2f6;
    }}
    .lead-name {{ font-size: 18px; font-weight: 700; color: white; }}
    .lead-temp {{
        font-size: 12px; font-weight: 600; padding: 4px 14px; border-radius: 20px;
        text-transform: uppercase; letter-spacing: 0.5px;
    }}
    .temp-hot {{ background: #fee2e2; color: #dc2626; }}
    .temp-warm {{ background: #fef3c7; color: #d97706; }}
    .temp-cold {{ background: #dbeafe; color: #2563eb; }}
    .lead-field {{ display: flex; gap: 8px; margin-bottom: 6px; font-size: 14px; line-height: 1.7; }}
    .lead-label {{ font-weight: 600; color: white; min-width: 110px; }}
    .lead-value {{ color: white; flex: 1; }}
    .lead-summary {{ margin-top: 12px; padding: 12px 16px; background: #f8fafc; border-radius: 10px; border-right: 3px solid #4552D4; }}
    .lead-summary p {{ margin: 0; color: #374151; line-height: 1.8; }}
    .lead-meta {{ margin-top: 12px; font-size: 12px; color: #9ca3af; text-align: left; direction: ltr; }}
    .badge {{
        display: inline-block; font-size: 11px; font-weight: 600; padding: 2px 10px;
        border-radius: 12px; margin: 2px 4px 2px 0;
    }}
    .badge-products {{ background: #ede9fe; color: #7c3aed; }}
    .badge-signals {{ background: #d1fae5; color: #059669; }}
    .badge-objections {{ background: #fce7f3; color: #db2777; }}
    .stats-box {{
        background: #1e2130; border: 1px solid #e8ecf2; border-radius: 14px;
        padding: 20px 24px; text-align: center;
    }}
    .stats-number {{ font-size: 32px; font-weight: 700; color: #4552D4; }}
    .stats-label {{ font-size: 13px; color: #6b7280; margin-top: 4px; }}
    </style>
    <div style="position:fixed;top:14px;right:20px;z-index:999;display:flex;align-items:center;gap:8px;padding:7px 10px;">
        <img src="data:image/png;base64,{icon_b64}" style="height:45px;display:block" alt="Kayfa">
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<h1 style='display:flex;align-items:center;gap:10px;'>📋 CRM — إدارة العملاء المحتملين</h1>", unsafe_allow_html=True)
    leads = list(crm_coll.find().sort("created_at", DESCENDING))
    

    if not leads:
        st.info("لم يتم تسجيل أي عملاء محتملين بعد. تحدث مع العميل في صفحة المحادثة ليتم تسجيلهم تلقائياً.")
    else:
        total = len(leads)
        hot = sum(1 for l in leads if l.get("lead_temperature", "").strip().lower() == "hot")
        warm = sum(1 for l in leads if l.get("lead_temperature", "").strip().lower() == "warm")
        cold = total - hot - warm
        cols = st.columns(4)
        with cols[0]:
            st.markdown(f'<div class="stats-box"><div class="stats-number" style="color:white">{total}</div><div class="stats-label">Total Clients</div></div>', unsafe_allow_html=True)
        with cols[1]:
            st.markdown(f'<div class="stats-box"><div class="stats-number" style="color:white">{hot}</div><div class="stats-label">Very Interested 🔥</div></div>', unsafe_allow_html=True)
        with cols[2]:
            st.markdown(f'<div class="stats-box"><div class="stats-number" style="color:white">{warm}</div><div class="stats-label">Interested ☀️</div></div>', unsafe_allow_html=True)
        with cols[3]:
            st.markdown(f'<div class="stats-box"><div class="stats-number" style="color:white">{cold}</div><div class="stats-label">Asking ❄️</div></div>', unsafe_allow_html=True)

        temp_filter = st.selectbox("📊 Filter by level of enthusiasm", ["All", "Very Interested", "Interested", "Asking"])
        search = st.text_input("🔍Search with Name or Phone Number", "").strip().lower()

        filtered = leads
        if temp_filter == "Very Interested":
            filtered = [l for l in filtered if l.get("lead_temperature", "").strip().lower() == "hot"]
        elif temp_filter == "Interested":
            filtered = [l for l in filtered if l.get("lead_temperature", "").strip().lower() == "warm"]
        elif temp_filter == "Asking":
            filtered = [l for l in filtered if l.get("lead_temperature", "").strip().lower() not in ("hot", "warm")]
        if search:
            filtered = [l for l in filtered if search in l.get("name", "").lower() or search in l.get("phone", "").lower()]

        st.markdown(f'<p style="color:#9ca3af;font-size:14px;margin:8px 0 16px;">عرض {len(filtered)} من أصل {total} تذكرة</p>', unsafe_allow_html=True)

        for lead in filtered:
            temp = lead.get("lead_temperature", "").strip().lower()
            temp_class = "temp-hot" if temp == "hot" else "temp-warm" if temp == "warm" else "temp-cold"
            temp_label = "Very Interested 🔥" if temp == "hot" else "Interested ☀️" if temp == "warm" else "Asking ❄️"
            created = lead.get("created_at", datetime.now(timezone.utc))
            date_str = created.strftime("%Y-%m-%d · %H:%M") if isinstance(created, datetime) else str(created)[:16]
            st.markdown(f"""
            <div class="lead-card">
                <div class="lead-header">
                    <div>
                        <span class="lead-name">{lead.get('name', 'غير معروف')}</span>
                        <span class="lead-temp {temp_class}">{temp_label}</span>
                    </div>
                    <div style="font-size:12px;color:#9ca3af;">{date_str}</div>
                </div>
                <div class="lead-field"><span class="lead-label">📞 رقم التواصل</span><span class="lead-value" style="color:white">{lead.get('phone', '—')}</span></div>
                <div class="lead-field"><span class="lead-label">✉️ البريد</span><span class="lead-value" style="color:white">{lead.get('email', '—')}</span></div>
                <div class="lead-field"><span class="lead-label">📍 المدينة / الدولة</span><span class="lead-value" style="color:white">{lead.get('city', '—')}، {lead.get('country', '—')}</span></div>
                <div class="lead-field"><span class="lead-label">🗣️ اللغة / اللهجة</span><span class="lead-value" style="color:white">{lead.get('language', '—')} / {lead.get('dialect', '—')}</span></div>
                <div class="lead-field"><span class="lead-label">📚 المنتجات محل الاهتمام</span><span class="lead-value" style="color:white"><span class="badge badge-products">{lead.get('products', '—')}</span></span></div>
                <div class="lead-field"><span class="lead-label">🎯 الهدف</span><span class="lead-value" style="color:white">{lead.get('goal', '—')}</span></div>
                <div class="lead-field"><span class="lead-label">📊 المستوى الحالي</span><span class="lead-value" style="color:white">{lead.get('level', '—')}</span></div>
                <div class="lead-field"><span class="lead-label">💡 إشارات الشراء</span><span class="lead-value" style="color:white"><span class="badge badge-signals">{lead.get('buying_signals', '—')}</span></span></div>
                <div class="lead-summary">
                    <p><strong>📝 ملخص المحادثة:</strong> {lead.get('summary', '—')}</p>
                    <p style="margin-top:8px"><strong>📌 الإجراء التالي:</strong> {lead.get('next_action', '—')}</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
    st.markdown('<div style="text-align:center;padding:30px 0 10px;font-size:13px;color:#6b7280;">All Tickets are stored in Database</div>', unsafe_allow_html=True)

else:
    st.title("Kayfa Agent")
    sid = st.session_state.current_session
    chat_messages = st.session_state.sessions[sid]["messages"]

    for msg in chat_messages:
        with st.chat_message(msg["role"]):
            cls = dir_class(msg["content"])
            st.markdown(f'<div class="{cls}">{msg["content"]}</div>', unsafe_allow_html=True)

    if prompt := st.chat_input("Ask me anything about Kayfa's courses, tracks, and diplomas..."):
        chat_messages.append({"role": "user", "content": prompt})
        save_turn(sid, "user", prompt)
        with st.chat_message("user"):
            cls = dir_class(prompt)
            st.markdown(f'<div class="{cls}">{prompt}</div>', unsafe_allow_html=True)
        with st.chat_message("assistant"):
            with st.spinner("Kayfa AI is thinking..."):
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    loop = None
                if loop and loop.is_running():
                    import nest_asyncio
                    nest_asyncio.apply()
                    model_history = st.session_state.sessions[sid].get("model_history", [])
                    result = loop.run_until_complete(agent.run(prompt, deps=deps, message_history=model_history))
                else:
                    model_history = st.session_state.sessions[sid].get("model_history", [])
                    result = asyncio.run(agent.run(prompt, deps=deps, message_history=model_history))
            st.session_state.sessions[sid]["model_history"] = result.all_messages()[-5:]
            cls = dir_class(result.output)
            st.markdown(f'<div class="{cls}">{result.output}</div>', unsafe_allow_html=True)
        chat_messages.append({"role": "assistant", "content": result.output})
        save_turn(sid, "assistant", result.output)
        sdata = st.session_state.sessions[sid]
        if sdata["name"] in ("New Chat", "Session 1"):
            new_name = prompt[:40] + ("..." if len(prompt) > 40 else "")
            sdata["name"] = new_name
            rename_session(sid, new_name)
