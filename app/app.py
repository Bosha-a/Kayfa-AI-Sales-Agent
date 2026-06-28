from datetime import timezone, datetime, timedelta

UTC_PLUS_3 = timezone(timedelta(hours=3))
import os
import re
import asyncio
import certifi
import html
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
from pydantic_ai import RunContext, Agent
from pymongo import MongoClient, ASCENDING, DESCENDING
from bson.objectid import ObjectId
from dataclasses import dataclass, asdict
import base64
import json
import time
from auth import render_login, ROLE_PERMISSIONS
from dashboard import render_monitoring_dashboard
from chat import render_chat_page
from crm import render_crm_page
from optimizations import SemanticCache, route_query
from PIL import Image as _PILImage

try:
    from genai_prices import Usage as GenAIUsage, calc_price as genai_calc_price
except Exception:
    GenAIUsage = None
    genai_calc_price = None


load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

_kayfa_page_icon = _PILImage.open(os.path.join(os.path.dirname(__file__), "..", "images", "kayfa_icon.png"))
_kayfa_chat_icon = _PILImage.open(os.path.join(os.path.dirname(__file__), "..", "images", "kayfa_icon_white.png"))
st.set_page_config(page_title="Kayfa Agent", page_icon=_kayfa_page_icon, layout="wide")

if (
    "authenticated" not in st.session_state
    or not st.session_state.authenticated
    or not st.session_state.get("user")
):
    render_login()
    st.stop()

user_role = st.session_state.get("user", {}).get("role", "user")
perms = ROLE_PERMISSIONS.get(user_role, {})
if not perms.get("chat") and not perms.get("crm") and not perms.get("dashboard"):
    st.error("No permissions assigned.")
    if st.button("Logout"):
        st.session_state.clear()
        st.rerun()
    st.stop()

has_crm_access = perms.get("crm", False)
has_chat_access = perms.get("chat", False)
has_dashboard_access = perms.get("dashboard", False)

color = '#2D3BE0'
ARABIC_RE = re.compile(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]")

def dir_class(text: str) -> str:
    return "rtl" if ARABIC_RE.search(text) else "ltr"

def esc(value) -> str:
    return html.escape(str(value if value is not None else ""), quote=True)

def approximate_tokens(text: str) -> int:
    return max(1, len(str(text or "")) // 4)

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
with open(os.path.join(_ROOT, "images", "kayfa.png"), "rb") as f:
    logo_b64 = base64.b64encode(f.read()).decode()
with open(os.path.join(_ROOT, "images", "kayfa_icon.png"), "rb") as f:
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

mongo_uri = st.secrets.get("MONGODB_URI") 
qdrant_api_key = st.secrets.get("QDRANT_API_KEY") 
qdrant_url = st.secrets.get("QDRANT_URL") 
groq_api_key = st.secrets.get("GROQ_API_KEY") 
mongo_uri = mongo_uri or os.getenv("MONGODB_URI")
qdrant_api_key = qdrant_api_key or os.getenv("QDRANT_API_KEY")
qdrant_url = qdrant_url or os.getenv("QDRANT_URL")
groq_api_key = groq_api_key or os.getenv("GROQ_API_KEY")

mongo_client = MongoClient(mongo_uri, tls=True, tlsCAFile=certifi.where())
messages = mongo_client.kayfa.messages
messages.create_index([("session_id", ASCENDING), ("timestamp", ASCENDING)])
messages.create_index([("username", ASCENDING)])

def _current_user():
    return st.session_state.get("user", {}).get("username", "anonymous")

def _current_user_id():
    user = st.session_state.get("user", {})
    return user.get("id") or user.get("username", "anonymous")

def save_turn(session_id, role, content):
    messages.insert_one({
        "session_id": session_id, "role": role,
        "content": content, "timestamp": datetime.now(UTC_PLUS_3),
        "username": _current_user(),
        "user_id": _current_user_id(),
    })

def load_session(session_id):
    cursor = messages.find({"session_id": session_id}).sort("timestamp", ASCENDING)
    return [{"role": doc["role"], "content": doc["content"]} for doc in cursor]

sessions_coll = mongo_client.kayfa.sessions
sessions_coll.create_index([("updated_at", ASCENDING)])
sessions_coll.create_index([("username", ASCENDING)])

def create_session(name="New Chat"):
    doc = {
        "name": name, "username": _current_user(), "user_id": _current_user_id(),
        "created_at": datetime.now(UTC_PLUS_3),
        "updated_at": datetime.now(UTC_PLUS_3),
    }
    result = sessions_coll.insert_one(doc)
    return str(result.inserted_id)

def get_all_sessions():
    cursor = sessions_coll.find({"username": _current_user()}).sort("updated_at", DESCENDING)
    return [(str(doc["_id"]), doc.get("name", "Chat")) for doc in cursor]

def rename_session(session_id, name):
    sessions_coll.update_one(
        {"_id": ObjectId(session_id)},
        {"$set": {"name": name, "updated_at": datetime.now(UTC_PLUS_3)}}
    )

def delete_session(session_id):
    sessions_coll.delete_one({"_id": ObjectId(session_id)})
    messages.delete_many({"session_id": session_id})

crm_coll = mongo_client.kayfa.crm_tickets
crm_coll.create_index([("created_at", DESCENDING)])

@st.cache_resource(show_spinner=False)
def load_models():
    sparse = SparseTextEmbedding(model_name="Qdrant/bm25")
    dense = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device='cpu')
    return sparse, dense

sparse_model, dense_model = load_models()

semantic_cache = SemanticCache(mongo_db=mongo_client.kayfa, dense_model=dense_model)

# Usage logs collection for monitoring
usage_logs = mongo_client.kayfa.usage_logs
usage_logs.create_index([("conversation_id", ASCENDING), ("timestamp", ASCENDING)])
usage_logs.create_index([("message_id", ASCENDING), ("timestamp", ASCENDING)])
usage_logs.create_index([("user_id", ASCENDING), ("timestamp", ASCENDING)])
usage_logs.create_index([("timestamp", ASCENDING)])

GENAI_PROVIDER_IDS = {
    "groq": "groq",
}

# Local fallback pricing (per 1M tokens) for local tools or models missing from genai-prices.
FALLBACK_PRICING = {
    "groq": {
        "openai/gpt-oss-20b": {"input": 0.029, "output": 0.130},
        "openai/gpt-oss-120b": {"input": 0.039, "output": 0.180},
        "llama-3.1-70b-versatile": {"input": 0.59, "output": 0.79},
        "llama-3.1-8b-instant": {"input": 0.02, "output": 0.03},
        "default": {"input": 0.15, "output": 0.60},
    },
    "embedding": {
        "sentence-transformers/all-MiniLM-L6-v2 + Qdrant/bm25": {"input": 0.0, "output": 0.0},
        "local": {"input": 0.0, "output": 0.0},
    }
}

def calculate_cost_details(model_provider: str, model_name: str, input_tokens: int, output_tokens: int) -> dict:
    """Calculate usage cost and preserve the pricing source for auditability."""
    safe_input = max(0, int(input_tokens or 0))
    safe_output = max(0, int(output_tokens or 0))
    provider_id = GENAI_PROVIDER_IDS.get(model_provider)

    if provider_id and GenAIUsage and genai_calc_price:
        try:
            price = genai_calc_price(
                GenAIUsage(input_tokens=safe_input, output_tokens=safe_output),
                model_name,
                provider_id=provider_id,
            )
            return {
                "cost_usd": float(price.total_price),
                "input_cost_usd": float(price.input_price),
                "output_cost_usd": float(price.output_price),
                "pricing_source": "genai-prices",
                "pricing_provider_id": provider_id,
            }
        except Exception as exc:
            pricing_error = f"{type(exc).__name__}: {exc}"
    else:
        pricing_error = "genai-prices unavailable" if provider_id else "local/free provider"

    provider_pricing = FALLBACK_PRICING.get(model_provider, {})
    model_pricing = provider_pricing.get(model_name, provider_pricing.get("default", {"input": 0, "output": 0}))
    input_cost = (safe_input / 1_000_000) * model_pricing.get("input", 0)
    output_cost = (safe_output / 1_000_000) * model_pricing.get("output", 0)
    return {
        "cost_usd": input_cost + output_cost,
        "input_cost_usd": input_cost,
        "output_cost_usd": output_cost,
        "pricing_source": "fallback",
        "pricing_provider_id": model_provider,
        "pricing_error": pricing_error,
    }

def calculate_cost(model_provider: str, model_name: str, input_tokens: int, output_tokens: int) -> float:
    """Return only the total cost for existing UI call sites."""
    return calculate_cost_details(model_provider, model_name, input_tokens, output_tokens)["cost_usd"]

def log_usage(
    conversation_id: str,
    user_id: str,
    username: str,
    model_provider: str,
    model_name: str,
    input_tokens: int,
    output_tokens: int,
    tool_calls: list,
    tool_results: list,
    latency_ms: int,
    step_type: str,
    message_id: str = None,
    trace_data: dict = None
):
    """Log usage to MongoDB for monitoring."""
    cost_details = calculate_cost_details(model_provider, model_name, input_tokens, output_tokens)
    
    doc = {
        "conversation_id": conversation_id,
        "message_id": message_id or conversation_id,
        "user_id": user_id,
        "username": username,

        "model_provider": model_provider,
        "model_name": model_name,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,

        "cost_usd": cost_details["cost_usd"],
        "input_cost_usd": cost_details.get("input_cost_usd", 0.0),
        "output_cost_usd": cost_details.get("output_cost_usd", 0.0),
        "pricing_source": cost_details.get("pricing_source", "unknown"),
        "pricing_provider_id": cost_details.get("pricing_provider_id", model_provider),
        "pricing_error": cost_details.get("pricing_error"),
        "tool_calls": tool_calls,
        "tool_results": tool_results,

        "latency_ms": latency_ms,
        "step_type": step_type,  # "llm_call", "tool_call", "embedding"
        "trace_data": trace_data or {},
        "timestamp": datetime.now(UTC_PLUS_3),
    }
    usage_logs.insert_one(doc)
    return doc

def save_lead(**kw) -> str:
    kw["level"] = kw.get("level") or "مبتدئ"
    kw["created_at"] = datetime.now(UTC_PLUS_3)
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

Fallback when knowledge base has no answer:
Arabic: "للأسف ما عندي معلومات كافية عن هذا الموضوع الآن، لكن فريق كيف سيجيبك بشكل مباشر — تواصل معهم على info@kayfa.io أو عبر واتساب، وسأساعدك في التواصل الآن."
English: "I don't have enough information on that right now, but the Kayfa team can help you directly. You can reach them at info@kayfa.io — would you like me to connect you?"
Then offer to capture their contact info and create a support ticket.


# Output Style
- Render markdown properly — use line breaks and formatting to make responses easy to read
- Keep responses concise (3–5 sentences for general questions, more when explaining a specific product)
- Use bullet points only for course features, comparisons, or step-by-step instructions — not for casual conversation
- Use light emoji (1–2 max per message) to keep the tone warm, never excessive
- Never write walls of text — break up longer answers with spacing


# Sales Approach
1. **LISTEN** — Understand the visitor's intent: are they browsing, comparing, price-sensitive, hesitant, or ready to enroll?
2. **RECOMMEND** — Map their goal to real Kayfa products using the knowledge base. Right product, right level, right price.
3. **PERSUADE** — Frame value using real social proof (instructors, partners, accreditation). Handle objections honestly.
4. **UPSELL** — Start where they're comfortable (free content or individual courses). Guide warm leads upward toward tracks and live diplomas where it genuinely fits.
5. **CAPTURE** — When buying signals appear, pivot naturally to collecting missing conversation details, then call capture_lead().


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


# WHAT TO PASS TO capture_lead()
- Match the visitor's language: if they speak Arabic → fill ALL fields in Arabic; if English → in English
- You MUST collect BOTH name AND phone before calling capture_lead():
- **name**: their full name — ask naturally e.g. "بأي اسم أناديك؟" if arabic or "What's your name?" if english 
- **phone**: their phone number with country code — ask naturally e.g. "ما رقم تواصلك؟" if arabic or "What's the best number to reach you?" if english
- **products**: the specific course, track, or diploma they are interested in
- **goal**: their motivation or what they want to achieve
- **level**: their current skill level (مبتدئ / متوسط / متقدم) if arabic or (beginner / intermediate / advanced) if english
- **language**: the language the user is speaking (e.g. "العربية", "English") — detect it from the conversation


## Rules - READ CAREFULLY:
- Collect name and phone conversationally — never ask for both at once, weave them into the conversation naturally
- Do NOT ask name/phone until you detect at least 1 buying signal
- NEVER call capture_lead() with only name and no phone, or only phone and no name — both are required together, no exceptions 
- If you have buying signals but missing fields, gather them one at a time in a natural flow
- ❌ NEVER call capture_lead() without name
- ❌ NEVER call capture_lead() without phone
- ❌ NEVER call capture_lead() with name only and phone missing
- ❌ NEVER call capture_lead() with phone only and name missing
- ✅ ONLY call capture_lead() when you have BOTH name AND phone confirmed by the user
- If the user refuses to give name or phone, do NOT create a ticket — just continue helping them normally
- If you are not 100% sure you have both name and phone, do NOT call capture_lead()


# Handling Off-Topic Questions
If the visitor asks something unrelated to Kayfa's educational offerings, gently redirect:

Visitor: "ممكن تساعدني في كتابة كود Python؟"
Agent: "أنا هنا خصيصاً لأساعدك تختار المسار التعليمي المناسب في كيف 😊 — لو مهتم تتعلم Python باحترافية، عندنا مسارات ممتازة تناسبك. تبي أعرّفك عليها؟"

Always try to bridge back to a relevant Kayfa product when possible.


# Brand Voice
- Professional, warm, and helpful — you're a trusted guide, not a pushy salesperson
- Persuasive with facts and value statements, never with pressure or manipulation
- Use real social proof: "هذه الدبلومة معتمدة من جامعة ديلاوير وليدز أكاديمي" or "أكثر من ١٥,٠٠٠ متعلم يثقون في كيف"
- Be honest if a product doesn't fit the visitor — recommend what's best for them
- NEVER break character — you are a Kayfa sales agent, nothing else
- If asked off-topic questions, politely redirect to Kayfa's educational offerings
"""

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

    def search_with_trace(self, query: str, limit: int = 5):
        results = self.hybrid_search(query, k=limit)
        sources = []
        previews = []
        for r in results:
            meta = r.get("metadata", {}) or {}
            source = meta.get("source") or meta.get("name") or meta.get("file") or meta.get("type")
            if source and source not in sources:
                sources.append(source)
            content = (r.get("content") or "").strip()
            if content:
                previews.append({
                    "source": source or "unknown",
                    "text": content[:350],
                    "score": r.get("score"),
                })
        return self.format_context(results), sources, len(results), previews

rag_service = RAGService(client=client, dense_embedder=dense_model, sparse_embedder=sparse_model)

@dataclass
class Dependencies:
    rag: any
    conversation: list = None

model = GroqModel("openai/gpt-oss-20b", provider=GroqProvider(api_key=groq_api_key))
model_strong = GroqModel("openai/gpt-oss-120b", provider=GroqProvider(api_key=groq_api_key))


def search_kayfa_knowledge_base(ctx: RunContext[Dependencies], query: str, limit: int=5) -> str:
    start_time = time.time()
    context, sources, result_count, result_preview = ctx.deps.rag.search_with_trace(query, limit=limit)
    latency_ms = int((time.time() - start_time) * 1000)
    sid = st.session_state.get("current_session", "")
    message_id = st.session_state.get("current_message_id", sid)
    current_user = st.session_state.get("user", {})
    log_usage(
        conversation_id=sid,
        message_id=message_id,
        user_id=current_user.get("id") or current_user.get("username", "unknown"),
        username=current_user.get("name", "unknown"),
        model_provider="embedding",
        model_name="sentence-transformers/all-MiniLM-L6-v2 + Qdrant/bm25",
        input_tokens=approximate_tokens(query),
        output_tokens=0,
        tool_calls=[{"tool": "search_kayfa_knowledge_base", "args": {"query": query, "limit": limit}}],
        tool_results=[{"tool": "search_kayfa_knowledge_base", "result": {"chunks": result_count, "sources": sources}}],
        latency_ms=latency_ms,
        step_type="embedding_retrieval",
        trace_data={
            "query": query,
            "sources": sources,
            "result_shape": f"{result_count} chunks returned",
            "result_preview": result_preview,
        },
    )
    return context


_LANGUAGE_MAP = {
    "english": "الإنجليزية",
    "arabic": "العربية",
    "English": "الإنجليزية",
    "Arabic": "العربية",
}


def _ensure_language_arabic(language: str) -> str:
    """Convert language name to Arabic using the mapping."""
    return _LANGUAGE_MAP.get(language, language)

def _generate_summary(messages: list, sid: str = "") -> str:
    """Summarize the chat conversation using Groq (Arabic).

    Falls back to loading messages from MongoDB when the in-memory list is
    empty (e.g. when called from inside a background thread where
    st.session_state is unavailable).
    """
    # --- Fallback: load from DB when in-memory list is empty --------------------
    if not messages and sid:
        try:
            msgs_cursor = mongo_client.kayfa.messages.find(
                {"session_id": sid}
            ).sort("timestamp", ASCENDING)
            messages = [{"role": doc["role"], "content": doc["content"]} for doc in msgs_cursor]
        except Exception:
            pass

    if not messages:
        return ""

    prompt_lines = [
        "لخص المحادثة التالية بين العميل ووكيل مبيعات باللغة العربية في سطر واحد فقط."
    ]

    for m in messages:
        role = "العميل" if m.get("role") == "user" else "وكيل المبيعات"
        prompt_lines.append(f"{role}: {m.get('content', '')}")

    if not groq_api_key:
        return ""

    try:
        groq_http = __import__("groq", fromlist=["Groq"]).Groq(api_key=groq_api_key)
        response = groq_http.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": "أنت مساعد يقوم بتلخيص محادثات خدمة العملاء باللغة العربية في جملة واحدة فقط."
                },
                {
                    "role": "user",
                    "content": "\n".join(prompt_lines)
                }
            ],
            temperature=0.2,
            max_tokens=60,
        )
        return response.choices[0].message.content.strip()

    except Exception as _e:
        print(f"[_generate_summary] error: {_e}")
        return ""


def capture_lead(ctx: RunContext[Dependencies], name: str = "", phone: str = "", 
                country: str = "", language: str = "", products: str = "", 
                 goal: str = "", level: str = "", 
                 summary: str = "") -> str:
    start_time = time.time()

    name = name.strip() if name else ""
    phone = phone.strip() if phone else ""
    sid = st.session_state.get("current_session", "")
    message_id = st.session_state.get("current_message_id", sid)
    user = st.session_state.get("user", {})

    def _log_capture_tool(result_text: str, blocked: bool = False):
        log_usage(
            conversation_id=sid,
            message_id=message_id,
            user_id=user.get("id") or user.get("username", "unknown"),
            username=user.get("name", "unknown"),
            model_provider="tool",
            model_name="capture_lead",
            input_tokens=approximate_tokens(" ".join(str(v or "") for v in [name, phone, country, language, products, goal, level, summary])),
            output_tokens=approximate_tokens(result_text),
            tool_calls=[{"tool": "capture_lead", "args": {"name": name, "phone": phone, "country": country, "language": language, "products": products, "goal": goal, "level": level}}],
            tool_results=[{"tool": "capture_lead", "result": result_text[:500]}],
            latency_ms=int((time.time() - start_time) * 1000),
            step_type="tool_call",
            trace_data={
                "tool_calls": [{"tool": "capture_lead", "blocked": blocked}],
                "tool_results": [{"tool": "capture_lead", "result": result_text[:500]}],
            },
        )

    if not name or name in ("—", "-", "unknown", "غير معروف", "null", "none", ""):
        result_text = "BLOCKED. Name is missing. Do NOT call this function again until the user provides their full name. Ask them now."
        _log_capture_tool(result_text, blocked=True)
        return result_text

    if not phone or phone in ("—", "-", "unknown", "غير معروف", "null", "none", ""):
        result_text = f"BLOCKED. You have the name ({name}) but phone is missing. Do NOT call this function again until the user provides their phone number. Ask them now."
        _log_capture_tool(result_text, blocked=True)
        return result_text

    digits = phone.replace("+", "").replace(" ", "").replace("-", "")
    if not digits.isdigit() or len(digits) < 7:
        result_text = f"BLOCKED. Phone number ({phone}) is invalid. Ask the user for a valid phone number with country code."
        _log_capture_tool(result_text, blocked=True)
        return result_text

    user = st.session_state.get("user", {})
    if not country:
        country = user.get("country", "")

    language = language.strip() if language else ""
    language = _ensure_language_arabic(language)

    groq_summary = _generate_summary(ctx.deps.conversation or [], sid=sid)
    final_summary = groq_summary or summary

    result_text = save_lead(
        user_id=_current_user_id(), username=_current_user(),
        name=name, phone=phone, country=country,
        language=language,
        products=products, goal=goal, level=level,
        summary=final_summary
    )
    _log_capture_tool(result_text)
    return result_text


_agent_tools = [search_kayfa_knowledge_base, capture_lead]

agent = Agent(model, deps_type=Dependencies, system_prompt=SYSTEM_PROMPT, tools=_agent_tools)
agent_strong = Agent(model_strong, deps_type=Dependencies, system_prompt=SYSTEM_PROMPT, tools=_agent_tools)
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
    if has_dashboard_access:
        st.session_state.page = "dashboard"
    elif has_crm_access and not has_chat_access:
        st.session_state.page = "crm"
    else:
        st.session_state.page = "chat"

if st.session_state.page == "chat" and not has_chat_access:
    st.session_state.page = "crm" if has_crm_access else "dashboard"
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
    if has_dashboard_access:
        if st.button("📊  Dashboard", key="nav_dashboard", use_container_width=True):
            st.session_state.page = "dashboard"
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
            if sid == st.session_state.current_session:
                label = f"> {label}"
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

# ── Page Routing (uses modular files) ──
if st.session_state.page == "dashboard" and has_dashboard_access:
    render_monitoring_dashboard(usage_logs)

elif st.session_state.page == "crm" and has_crm_access:
    render_crm_page(crm_coll, icon_b64)

else:
    render_chat_page(
        agents={"openai/gpt-oss-20b": agent, "openai/gpt-oss-120b": agent_strong},
        deps=deps,
        save_turn=save_turn,
        rename_session=rename_session,
        dir_class=dir_class,
        esc=esc,
        log_usage=log_usage,
        approximate_tokens=approximate_tokens,
        calculate_cost=calculate_cost,
        SYSTEM_PROMPT=SYSTEM_PROMPT,
        assistant_avatar=_kayfa_chat_icon,
        semantic_cache=semantic_cache,
        route_query=route_query,
    )
