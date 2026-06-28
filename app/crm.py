from datetime import timezone, datetime
import streamlit as st
from pymongo import DESCENDING


def render_crm_page(crm_coll, icon_b64):
    """CRM lead management page — إدارة العملاء المحتملين."""

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

        temp_filter = st.selectbox("📊Filter by level of enthusiasm", ["All", "Very Interested", "Interested", "Asking"])
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
            level = lead.get("level") or "مبتدئ"
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
                <div class="lead-field"><span class="lead-label">📍 الدولة</span><span class="lead-value" style="color:white">{lead.get('country', '—')}</span></div>
                <div class="lead-field"><span class="lead-label">🗣️ اللغة </span><span class="lead-value" style="color:white">{lead.get('language', '—')}</span></div>
                <div class="lead-field"><span class="lead-label">🎯 الهدف</span><span class="lead-value" style="color:white">{lead.get('goal', '—')}</span></div>
                <div class="lead-field"><span class="lead-label">📊 المستوى الحالي</span><span class="lead-value" style="color:white">{level}</span></div>
                <div class="lead-summary">
                    <p><strong>📝 ملخص المحادثة:</strong> {lead.get('summary', '—')}</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
    st.markdown('<div style="text-align:center;padding:30px 0 10px;font-size:13px;color:#6b7280;">All Tickets are stored in Database</div>', unsafe_allow_html=True)
