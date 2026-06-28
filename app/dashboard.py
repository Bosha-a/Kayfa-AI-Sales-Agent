import streamlit as st
from pymongo import ASCENDING
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime


# ── Plotly shared layout defaults ──────────────────────────────────
_PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color="#c8cce0"),
    margin=dict(l=40, r=20, t=40, b=40),
    legend=dict(
        bgcolor="rgba(0,0,0,0)",
        bordercolor="rgba(200,204,224,0.1)",
        borderwidth=1,
        font=dict(size=11),
    ),
    xaxis=dict(gridcolor="rgba(200,204,224,0.08)", zerolinecolor="rgba(200,204,224,0.08)"),
    yaxis=dict(gridcolor="rgba(200,204,224,0.08)", zerolinecolor="rgba(200,204,224,0.08)"),
)

# Brand-matched color palette
_COLORS = [
    "#2D3BE0", "#6C5CE7", "#00CEC9", "#FD79A8", "#FDCB6E",
    "#55EFC4", "#74B9FF", "#E17055", "#A29BFE", "#FF7675",
    "#81ECEC", "#FAB1A0", "#DFE6E9", "#636E72", "#B2BEC3",
]

def _short_id(value, size=12):
    value = str(value or "-")
    return value[:size] + "..." if len(value) > size else value

def _build_cost_over_time_chart(usage_logs):
    pipeline = [
        {"$group": {
            "_id": {"$dateToString": {"format": "%Y-%m-%d", "date": "$timestamp"}},
            "cost": {"$sum": {"$ifNull": ["$cost_usd", 0]}},
        }},
        {"$sort": {"_id": 1}},
    ]
    rows = list(usage_logs.aggregate(pipeline))
    if not rows:
        st.info("No data for cost-over-time chart yet.")
        return
    df = pd.DataFrame([{"date": r["_id"], "cost": r["cost"]} for r in rows])
    df["date"] = pd.to_datetime(df["date"])
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["cost"],
        name="Total", mode="lines+markers",
        line=dict(color="#FDCB6E", width=3),
        marker=dict(size=5),
    ))
    fig.update_layout(
        **_PLOTLY_LAYOUT,
        title=dict(text="Cost Over Time", font=dict(size=16, color="#ffffff")),
        xaxis_title="Date", yaxis_title="Cost (USD)",
        yaxis_tickprefix="$", hovermode="x unified", height=420,
    )
    st.plotly_chart(fig, use_container_width=True)


def _build_cost_per_user_chart(user_rollup):
    if not user_rollup:
        st.info("No per-user data for chart yet.")
        return
    df = pd.DataFrame(user_rollup)
    df = df.sort_values("cost_usd", ascending=True)
    fig = go.Figure(go.Bar(
        x=df["cost_usd"], y=df["username"], orientation="h",
        marker=dict(
            color=df["cost_usd"],
            colorscale=[[0, "#6C5CE7"], [0.5, "#2D3BE0"], [1, "#00CEC9"]],
            line=dict(width=0), cornerradius=4,
        ),
        text=df["cost_usd"].apply(lambda v: f"${v:.6f}"),
        textposition="outside", textfont=dict(size=11, color="#c8cce0"),
        hovertemplate="<b>%{y}</b><br>Cost: $%{x:.6f}<extra></extra>",
    ))
    fig.update_layout(
        **_PLOTLY_LAYOUT,
        title=dict(text="Cost Per User", font=dict(size=16, color="#ffffff")),
        xaxis_title="Cost (USD)", xaxis_tickprefix="$", yaxis_title="",
        height=max(300, len(df) * 40 + 100),
    )
    st.plotly_chart(fig, use_container_width=True)


def _build_llm_vs_embedding_chart(usage_logs):
    pipeline = [
        {"$group": {
            "_id": {
                "date": {"$dateToString": {"format": "%Y-%m-%d", "date": "$timestamp"}},
                "step_type": {"$ifNull": ["$step_type", "other"]},
            },
            "cost": {"$sum": {"$ifNull": ["$cost_usd", 0]}},
        }},
        {"$sort": {"_id.date": 1}},
    ]
    rows = list(usage_logs.aggregate(pipeline))
    if not rows:
        st.info("No data for LLM vs Embedding chart yet.")
        return
    df = pd.DataFrame([{"date": r["_id"]["date"], "step_type": r["_id"]["step_type"], "cost": r["cost"]} for r in rows])
    df["date"] = pd.to_datetime(df["date"])
    type_map = {"llm_call": "LLM", "embedding_retrieval": "Embedding", "embedding": "Embedding"}
    df["label"] = df["step_type"].map(lambda s: type_map.get(s, "Other"))
    color_map = {"LLM": "#2D3BE0", "Embedding": "#00CEC9", "Other": "#636E72"}
    fig = go.Figure()
    for label in ["LLM", "Embedding", "Other"]:
        sub = df[df["label"] == label]
        if sub.empty:
            continue
        agg = sub.groupby("date")["cost"].sum().reset_index()
        fig.add_trace(go.Bar(
            x=agg["date"], y=agg["cost"], name=label,
            marker_color=color_map.get(label, "#636E72"),
            hovertemplate=f"<b>{label}</b><br>Date: %{{x|%Y-%m-%d}}<br>Cost: $%{{y:.6f}}<extra></extra>",
        ))
    fig.update_layout(
        **_PLOTLY_LAYOUT, barmode="stack",
        title=dict(text="LLM Cost vs Embedding Cost Per Day", font=dict(size=16, color="#ffffff")),
        xaxis_title="Date", yaxis_title="Cost (USD)",
        yaxis_tickprefix="$", hovermode="x unified", height=420,
    )
    st.plotly_chart(fig, use_container_width=True)


def _build_latency_vs_cost_scatter(usage_logs):
    pipeline = [
        {"$match": {"latency_ms": {"$gt": 0}, "username": {"$nin": [None, "", "unknown"]}}},
        {"$project": {
            "_id": 0,
            "message_id": {"$ifNull": ["$message_id", {"$toString": "$_id"}]},
            "username": 1,
            "cost_usd": {"$ifNull": ["$cost_usd", 0]},
            "latency_ms": {"$ifNull": ["$latency_ms", 0]},
            "step_type": {"$ifNull": ["$step_type", "other"]},
        }},
        {"$sort": {"cost_usd": -1}},
        {"$limit": 500},
    ]
    rows = list(usage_logs.aggregate(pipeline))
    if not rows:
        st.info("No data for latency-vs-cost scatter yet.")
        return
    df = pd.DataFrame(rows)
    users = sorted(df["username"].unique())
    fig = go.Figure()
    for i, user in enumerate(users):
        udf = df[df["username"] == user]
        fig.add_trace(go.Scatter(
            x=udf["latency_ms"], y=udf["cost_usd"], mode="markers", name=user,
            marker=dict(color=_COLORS[i % len(_COLORS)], size=8, opacity=0.75,
                        line=dict(width=1, color="rgba(255,255,255,0.15)")),
            customdata=udf[["message_id", "step_type"]].values,
            hovertemplate=(
                "<b>%{customdata[1]}</b><br>"
                "Latency: %{x:,} ms<br>Cost: $%{y:.6f}<br>"
                "Message: %{customdata[0]}<extra></extra>"
            ),
        ))
    fig.update_layout(
        **_PLOTLY_LAYOUT,
        title=dict(text="Latency vs Cost Per Message", font=dict(size=16, color="#ffffff")),
        xaxis_title="Latency (ms)", yaxis_title="Cost (USD)",
        yaxis_tickprefix="$", height=450,
    )
    st.plotly_chart(fig, use_container_width=True)


def render_monitoring_dashboard(usage_logs):
    st.title("Monitoring Dashboard")
    st.markdown("<p style='color:#9ca3af;font-size:14px;'>Cost, tokens, latency, and response traces from MongoDB.</p>", unsafe_allow_html=True)

    if st.button("🔄 Refresh Data", key="dash_refresh"):
        st.rerun()

    total_doc = next(usage_logs.aggregate([
        {"$group": {
            "_id": None,
            "cost_usd": {"$sum": {"$ifNull": ["$cost_usd", 0]}},
            "total_tokens": {"$sum": {"$ifNull": ["$total_tokens", 0]}},
            "calls": {"$sum": 1},
            "avg_latency_ms": {"$avg": {"$ifNull": ["$latency_ms", 0]}},
            "last_updated": {"$max": "$timestamp"},
        }}
    ]), None)

    llm_count = usage_logs.count_documents({"step_type": "llm_call"})
    cache_count = usage_logs.count_documents({"step_type": "cache_hit"})

    if not total_doc:
        st.info("No usage logs yet. Start a chat conversation, then return here to inspect cost and traces.")
        return

    st.markdown("""
    <style>
    .stats-box {
        background: #1e2130; border: 1px solid #e8ecf2; border-radius: 14px;
        padding: 20px 24px; text-align: center;
    }
    .stats-number { font-size: 32px; font-weight: 700; color: white; }
    .stats-label { font-size: 13px; color: #6b7280; margin-top: 4px; }
    </style>
    """, unsafe_allow_html=True)

    cols = st.columns(4)
    with cols[0]:
        st.markdown(f'<div class="stats-box"><div class="stats-number">${float(total_doc.get("cost_usd", 0) or 0):.6f}</div><div class="stats-label">Total Cost</div></div>', unsafe_allow_html=True)
    with cols[1]:
        st.markdown(f'<div class="stats-box"><div class="stats-number">{int(total_doc.get("total_tokens", 0) or 0):,}</div><div class="stats-label">Total Tokens</div></div>', unsafe_allow_html=True)
    with cols[2]:
        st.markdown(f'<div class="stats-box"><div class="stats-number">{int(total_doc.get("calls", 0) or 0):,}</div><div class="stats-label">Model Calls</div></div>', unsafe_allow_html=True)
    with cols[3]:
        st.markdown(f'<div class="stats-box"><div class="stats-number">{int(total_doc.get("avg_latency_ms", 0) or 0):,} ms</div><div class="stats-label">Avg Latency</div></div>', unsafe_allow_html=True)

    st.caption(f"📄 {int(total_doc.get('calls', 0)):,} total logs · {llm_count:,} LLM calls · {cache_count:,} cache hits")

    tab_charts, tab_user, tab_conversation, tab_message, tab_consumption, tab_trace, tab_opt = st.tabs([
        "Charts", "Per User", "Per Conversation", "Per Message", "Consumption", "Trace consumption", "Optimization"
    ])

    user_rollup = list(usage_logs.aggregate([
        {"$match": {"username": {"$nin": [None, "", "unknown"]}}},
        {"$group": {
            "_id": {"user_id": "$user_id", "username": "$username"},
            "cost_usd": {"$sum": {"$ifNull": ["$cost_usd", 0]}},
            "total_tokens": {"$sum": {"$ifNull": ["$total_tokens", 0]}},
            "calls": {"$sum": 1},
            "avg_latency_ms": {"$avg": {"$ifNull": ["$latency_ms", 0]}},
        }},
        {"$project": {
            "_id": 0,
            "user_id": "$_id.user_id", "username": "$_id.username",
            "cost_usd": 1, "total_tokens": 1, "calls": 1,
            "avg_latency_ms": {"$round": ["$avg_latency_ms", 0]},
        }},
        {"$sort": {"cost_usd": -1}},
        {"$limit": 100},
    ]))

    conversation_rollup = list(usage_logs.aggregate([
        {"$match": {"username": {"$nin": [None, "", "unknown"]}}},
        {"$group": {
            "_id": {"conversation_id": "$conversation_id", "username": "$username"},
            "cost_usd": {"$sum": {"$ifNull": ["$cost_usd", 0]}},
            "total_tokens": {"$sum": {"$ifNull": ["$total_tokens", 0]}},
            "calls": {"$sum": 1},
            "avg_latency_ms": {"$avg": {"$ifNull": ["$latency_ms", 0]}},
            "last_seen": {"$max": "$timestamp"},
        }},
        {"$project": {
            "_id": 0,
            "conversation_id": "$_id.conversation_id", "username": "$_id.username",
            "cost_usd": 1, "total_tokens": 1, "calls": 1,
            "avg_latency_ms": {"$round": ["$avg_latency_ms", 0]},
            "last_seen": 1,
        }},
        {"$sort": {"last_seen": -1}},
        {"$limit": 100},
    ]))

    message_rollup = list(usage_logs.aggregate([
        {"$match": {"username": {"$nin": [None, "", "unknown"]}}},
        {"$group": {
            "_id": {
                "message_id": {"$ifNull": ["$message_id", {"$toString": "$_id"}]},
                "conversation_id": "$conversation_id",
                "username": "$username",
            },
            "cost_usd": {"$sum": {"$ifNull": ["$cost_usd", 0]}},
            "total_tokens": {"$sum": {"$ifNull": ["$total_tokens", 0]}},
            "input_tokens": {"$sum": {"$ifNull": ["$input_tokens", 0]}},
            "output_tokens": {"$sum": {"$ifNull": ["$output_tokens", 0]}},
            "calls": {"$sum": 1},
            "latency_ms": {"$sum": {"$ifNull": ["$latency_ms", 0]}},
            "last_seen": {"$max": "$timestamp"},
        }},
        {"$project": {
            "_id": 0,
            "message_id": "$_id.message_id", "conversation_id": "$_id.conversation_id",
            "username": "$_id.username",
            "cost_usd": 1, "total_tokens": 1, "input_tokens": 1, "output_tokens": 1,
            "calls": 1, "latency_ms": 1, "last_seen": 1,
        }},
        {"$sort": {"last_seen": -1}},
        {"$limit": 100},
    ]))

    operation_rollup = list(usage_logs.aggregate([
        {"$match": {"username": {"$nin": [None, "", "unknown"]}}},
        {"$group": {
            "_id": {
                "step_type": {"$ifNull": ["$step_type", "other"]},
                "provider": {"$ifNull": ["$model_provider", "unknown"]},
                "model": {"$ifNull": ["$model_name", "unknown"]},
                "pricing_source": {"$ifNull": ["$pricing_source", "unknown"]},
            },
            "cost_usd": {"$sum": {"$ifNull": ["$cost_usd", 0]}},
            "input_cost_usd": {"$sum": {"$ifNull": ["$input_cost_usd", 0]}},
            "output_cost_usd": {"$sum": {"$ifNull": ["$output_cost_usd", 0]}},
            "input_tokens": {"$sum": {"$ifNull": ["$input_tokens", 0]}},
            "output_tokens": {"$sum": {"$ifNull": ["$output_tokens", 0]}},
            "calls": {"$sum": 1},
            "avg_latency_ms": {"$avg": {"$ifNull": ["$latency_ms", 0]}},
        }},
        {"$project": {
            "_id": 0,
            "operation": "$_id.step_type", "provider": "$_id.provider",
            "model": "$_id.model", "pricing_source": "$_id.pricing_source",
            "cost_usd": 1, "input_cost_usd": 1, "output_cost_usd": 1,
            "input_tokens": 1, "output_tokens": 1, "calls": 1,
            "avg_latency_ms": {"$round": ["$avg_latency_ms", 0]},
        }},
        {"$sort": {"cost_usd": -1, "calls": -1}},
        {"$limit": 100},
    ]))

    tool_operation_rollup = list(usage_logs.aggregate([
        {"$match": {"tool_calls.0": {"$exists": True}}},
        {"$unwind": "$tool_calls"},
        {"$group": {
            "_id": {
                "tool": {"$ifNull": ["$tool_calls.tool", "unknown"]},
                "step_type": {"$ifNull": ["$step_type", "other"]},
            },
            "cost_usd": {"$sum": {"$ifNull": ["$cost_usd", 0]}},
            "input_tokens": {"$sum": {"$ifNull": ["$input_tokens", 0]}},
            "output_tokens": {"$sum": {"$ifNull": ["$output_tokens", 0]}},
            "calls": {"$sum": 1},
            "avg_latency_ms": {"$avg": {"$ifNull": ["$latency_ms", 0]}},
        }},
        {"$project": {
            "_id": 0,
            "tool": "$_id.tool", "operation": "$_id.step_type",
            "cost_usd": 1, "input_tokens": 1, "output_tokens": 1, "calls": 1,
            "avg_latency_ms": {"$round": ["$avg_latency_ms", 0]},
        }},
        {"$sort": {"calls": -1, "cost_usd": -1}},
        {"$limit": 100},
    ]))

    # ── Shared CSS ──
    _timeline_css = """
    <style>
    .tl-container { position: relative; padding-left: 32px; }
    .tl-container::before {
        content: ""; position: absolute; left: 11px; top: 0; bottom: 0; width: 2px;
        background: linear-gradient(180deg, #2D3BE0 0%, #6C5CE7 50%, #00CEC9 100%);
        border-radius: 2px;
    }
    .tl-card {
        position: relative;
        background: linear-gradient(135deg, #1e2130 0%, #282b40 100%);
        border: 1px solid rgba(108, 92, 231, 0.2); border-radius: 14px;
        padding: 18px 20px; margin-bottom: 16px;
        transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
    }
    .tl-card:hover { transform: translateX(4px); box-shadow: 0 6px 28px rgba(45,59,224,0.2); border-color: rgba(108,92,231,0.45); }
    .tl-dot { position: absolute; left: -27px; top: 22px; width: 12px; height: 12px; border-radius: 50%; background: #6C5CE7; border: 2px solid #1e2130; box-shadow: 0 0 8px rgba(108,92,231,0.5); }
    .tl-header { display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 8px; margin-bottom: 12px; }
    .tl-title { font-size: 14px; font-weight: 600; color: #ffffff; display: flex; align-items: center; gap: 8px; }
    .tl-id { font-family: monospace; font-size: 12px; color: #a29bfe; background: rgba(108,92,231,0.12); padding: 2px 8px; border-radius: 6px; }
    .tl-badge { font-size: 11px; color: #9ca3af; background: rgba(255,255,255,0.05); padding: 3px 10px; border-radius: 20px; white-space: nowrap; }
    .tl-stats { display: flex; flex-wrap: wrap; gap: 8px; }
    .tl-pill { display: inline-flex; align-items: center; gap: 5px; background: rgba(255,255,255,0.04); border-radius: 8px; padding: 6px 12px; font-size: 12px; }
    .tl-pill-value { font-weight: 700; color: #ffffff; }
    .tl-pill-label { color: #6b7280; }
    .tl-user-tag { font-size: 11px; color: #00CEC9; background: rgba(0,206,201,0.1); padding: 2px 8px; border-radius: 6px; font-weight: 500; }

    /* ── Trace message bubbles ── */
    .trace-bubble-user {
        background: rgba(45,59,224,0.12);
        border: 1px solid rgba(45,59,224,0.25);
        border-radius: 12px 12px 12px 4px;
        padding: 14px 18px;
        margin-bottom: 6px;
        font-size: 14px;
        color: #e8ecf2;
        white-space: pre-wrap;
        word-break: break-word;
    }
    .trace-bubble-assistant {
        background: rgba(108,92,231,0.1);
        border: 1px solid rgba(108,92,231,0.2);
        border-radius: 12px 12px 4px 12px;
        padding: 14px 18px;
        margin-bottom: 6px;
        font-size: 14px;
        color: #e8ecf2;
        white-space: pre-wrap;
        word-break: break-word;
    }
    .trace-bubble-label {
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        margin-bottom: 6px;
        color: #6b7280;
    }
    .trace-section-divider {
        border: none;
        border-top: 1px solid rgba(200,204,224,0.08);
        margin: 16px 0;
    }
    .trace-step-header {
        font-size: 11px;
        color: #6b7280;
        background: rgba(255,255,255,0.03);
        border-radius: 8px;
        padding: 6px 12px;
        margin-bottom: 8px;
        display: inline-block;
    }
    </style>
    """

    with tab_charts:
        st.markdown("<p style='color:#9ca3af;font-size:13px;margin-bottom:4px;'>Interactive visualizations of cost, usage, and performance trends.</p>", unsafe_allow_html=True)
        _build_cost_over_time_chart(usage_logs)
        _build_cost_per_user_chart(user_rollup)
        _build_llm_vs_embedding_chart(usage_logs)
        _build_latency_vs_cost_scatter(usage_logs)

    with tab_user:
        if not user_rollup:
            st.info("No per-user data yet.")
        else:
            st.markdown("""
            <style>
            .user-card { background: linear-gradient(135deg, #1e2130 0%, #282b40 100%); border: 1px solid rgba(108,92,231,0.25); border-radius: 16px; padding: 24px 20px 18px 20px; text-align: center; transition: transform 0.22s ease, box-shadow 0.22s ease, border-color 0.22s ease; position: relative; overflow: hidden; }
            .user-card::before { content: ""; position: absolute; top: 0; left: 0; right: 0; height: 4px; background: linear-gradient(90deg, #2D3BE0, #6C5CE7, #00CEC9); border-radius: 16px 16px 0 0; }
            .user-card:hover { transform: translateY(-4px); box-shadow: 0 8px 32px rgba(45,59,224,0.25); border-color: rgba(108,92,231,0.5); }
            .user-card-avatar { width: 52px; height: 52px; border-radius: 50%; background: linear-gradient(135deg, #2D3BE0 0%, #6C5CE7 100%); display: flex; align-items: center; justify-content: center; margin: 0 auto 12px auto; font-size: 22px; font-weight: 700; color: #fff; letter-spacing: 1px; box-shadow: 0 4px 14px rgba(45,59,224,0.35); }
            .user-card-name { font-size: 16px; font-weight: 600; color: #ffffff; margin-bottom: 14px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
            .user-card-stats { display: grid; grid-template-columns: 1fr 1fr; gap: 10px 8px; }
            .user-card-stat { background: rgba(255,255,255,0.04); border-radius: 10px; padding: 10px 6px; }
            .user-card-stat-value { font-size: 15px; font-weight: 700; color: #ffffff; }
            .user-card-stat-label { font-size: 11px; color: #6b7280; margin-top: 2px; }
            </style>
            """, unsafe_allow_html=True)
            card_cols_count = 3
            for row_start in range(0, len(user_rollup), card_cols_count):
                row_items = user_rollup[row_start : row_start + card_cols_count]
                card_cols = st.columns(card_cols_count)
                for idx, user in enumerate(row_items):
                    with card_cols[idx]:
                        uname = user.get("username", "")
                        initials = "".join(w[0] for w in uname.split()[:2]).upper() or "?"
                        cost = float(user.get("cost_usd", 0) or 0)
                        tokens = int(user.get("total_tokens", 0) or 0)
                        calls = int(user.get("calls", 0) or 0)
                        latency = int(user.get("avg_latency_ms", 0) or 0)
                        st.markdown(f"""
                        <div class="user-card">
                            <div class="user-card-avatar">{initials}</div>
                            <div class="user-card-name" title="{uname}">{uname}</div>
                            <div class="user-card-stats">
                                <div class="user-card-stat"><div class="user-card-stat-value">${cost:.6f}</div><div class="user-card-stat-label">Cost (USD)</div></div>
                                <div class="user-card-stat"><div class="user-card-stat-value">{tokens:,}</div><div class="user-card-stat-label">Tokens</div></div>
                                <div class="user-card-stat"><div class="user-card-stat-value">{calls:,}</div><div class="user-card-stat-label">Model Calls</div></div>
                                <div class="user-card-stat"><div class="user-card-stat-value">{latency:,} ms</div><div class="user-card-stat-label">Avg Latency</div></div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

    with tab_conversation:
        if not conversation_rollup:
            st.info("No conversation data yet.")
        else:
            st.markdown(_timeline_css, unsafe_allow_html=True)
            st.markdown('<div class="tl-container">', unsafe_allow_html=True)
            for conv in conversation_rollup:
                conv_id = conv.get("conversation_id", "—")
                short_id = conv_id[:12] + "…" if len(conv_id) > 12 else conv_id
                uname = conv.get("username", "")
                cost = float(conv.get("cost_usd", 0) or 0)
                tokens = int(conv.get("total_tokens", 0) or 0)
                calls = int(conv.get("calls", 0) or 0)
                latency = int(conv.get("avg_latency_ms", 0) or 0)
                last_seen = conv.get("last_seen")
                time_str = last_seen.strftime("%b %d, %Y · %H:%M") if hasattr(last_seen, "strftime") else str(last_seen or "—")
                st.markdown(f"""
                <div class="tl-card">
                    <div class="tl-dot"></div>
                    <div class="tl-header">
                        <div class="tl-title">
                            <span>💬</span>
                            <span class="tl-id" title="{conv_id}">{short_id}</span>
                            <span class="tl-user-tag">@{uname}</span>
                        </div>
                        <div class="tl-badge">🕐 {time_str}</div>
                    </div>
                    <div class="tl-stats">
                        <div class="tl-pill"><span class="tl-pill-value">${cost:.6f}</span><span class="tl-pill-label">cost</span></div>
                        <div class="tl-pill"><span class="tl-pill-value">{tokens:,}</span><span class="tl-pill-label">tokens</span></div>
                        <div class="tl-pill"><span class="tl-pill-value">{calls}</span><span class="tl-pill-label">calls</span></div>
                        <div class="tl-pill"><span class="tl-pill-value">{latency:,} ms</span><span class="tl-pill-label">latency</span></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

    with tab_message:
        if not message_rollup:
            st.info("No message data yet.")
        else:
            st.markdown(_timeline_css, unsafe_allow_html=True)
            st.markdown('<div class="tl-container">', unsafe_allow_html=True)
            for msg in message_rollup:
                msg_id = msg.get("message_id", "—")
                short_msg_id = msg_id[:12] + "…" if len(msg_id) > 12 else msg_id
                conv_id = msg.get("conversation_id", "—")
                short_conv_id = conv_id[:10] + "…" if len(conv_id) > 10 else conv_id
                uname = msg.get("username", "")
                cost = float(msg.get("cost_usd", 0) or 0)
                tokens = int(msg.get("total_tokens", 0) or 0)
                input_tokens = int(msg.get("input_tokens", 0) or 0)
                output_tokens = int(msg.get("output_tokens", 0) or 0)
                calls = int(msg.get("calls", 0) or 0)
                latency = int(msg.get("latency_ms", 0) or 0)
                last_seen = msg.get("last_seen")
                time_str = last_seen.strftime("%b %d, %Y · %H:%M") if hasattr(last_seen, "strftime") else str(last_seen or "—")
                st.markdown(f"""
                <div class="tl-card">
                    <div class="tl-dot"></div>
                    <div class="tl-header">
                        <div class="tl-title">
                            <span>✉️</span>
                            <span class="tl-id" title="{msg_id}">{short_msg_id}</span>
                            <span class="tl-user-tag">@{uname}</span>
                            <span class="tl-badge" style="padding:2px 6px;font-size:10px;">conv: {short_conv_id}</span>
                        </div>
                        <div class="tl-badge">🕐 {time_str}</div>
                    </div>
                    <div class="tl-stats">
                        <div class="tl-pill"><span class="tl-pill-value">${cost:.6f}</span><span class="tl-pill-label">cost</span></div>
                        <div class="tl-pill"><span class="tl-pill-value">{tokens:,}</span><span class="tl-pill-label">tokens</span></div>
                        <div class="tl-pill"><span class="tl-pill-value">{input_tokens:,}</span><span class="tl-pill-label">input</span></div>
                        <div class="tl-pill"><span class="tl-pill-value">{output_tokens:,}</span><span class="tl-pill-label">output</span></div>
                        <div class="tl-pill"><span class="tl-pill-value">{calls}</span><span class="tl-pill-label">calls</span></div>
                        <div class="tl-pill"><span class="tl-pill-value">{latency:,} ms</span><span class="tl-pill-label">latency</span></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

    with tab_consumption:
        st.markdown("<p style='color:#9ca3af;font-size:13px;margin-bottom:4px;'>Token and cost consumption by operation, model, and tool.</p>", unsafe_allow_html=True)
        if operation_rollup:
            st.dataframe(
                pd.DataFrame(operation_rollup), use_container_width=True, hide_index=True,
                column_config={
                    "cost_usd": st.column_config.NumberColumn("Cost", format="$%.6f"),
                    "input_cost_usd": st.column_config.NumberColumn("Input Cost", format="$%.6f"),
                    "output_cost_usd": st.column_config.NumberColumn("Output Cost", format="$%.6f"),
                    "input_tokens": st.column_config.NumberColumn("Input Tokens", format="%d"),
                    "output_tokens": st.column_config.NumberColumn("Output Tokens", format="%d"),
                    "avg_latency_ms": st.column_config.NumberColumn("Avg Latency", format="%d ms"),
                },
            )
        else:
            st.info("No operation consumption data yet.")
        st.markdown("##### Tool Operations")
        if tool_operation_rollup:
            st.dataframe(
                pd.DataFrame(tool_operation_rollup), use_container_width=True, hide_index=True,
                column_config={
                    "cost_usd": st.column_config.NumberColumn("Cost", format="$%.6f"),
                    "input_tokens": st.column_config.NumberColumn("Input Tokens", format="%d"),
                    "output_tokens": st.column_config.NumberColumn("Output Tokens", format="%d"),
                    "avg_latency_ms": st.column_config.NumberColumn("Avg Latency", format="%d ms"),
                },
            )
        else:
            st.info("No tool calls have been logged yet.")

    # ── Trace Replay tab — full message content ──
    with tab_trace:
        st.markdown(_timeline_css, unsafe_allow_html=True)
        conv_options = conversation_rollup if conversation_rollup else []
        if not conv_options:
            st.info("No conversation data yet.")
        else:
            selected_row = st.selectbox(
                "Select a conversation to replay",
                conv_options,
                format_func=lambda row: (
                    f"@{row.get('username', '')} | {_short_id(row.get('conversation_id'))} | "
                    f"{int(row.get('total_tokens', 0) or 0):,} tokens | "
                    f"{row.get('last_seen').strftime('%b %d %H:%M') if hasattr(row.get('last_seen'), 'strftime') else ''}"
                ),
            )
            selected_conv = selected_row["conversation_id"]
            trace_logs = list(usage_logs.find(
                {"conversation_id": selected_conv}
            ).sort("timestamp", ASCENDING))

            # Group steps by message_id
            messages_by_id = {}
            for doc in trace_logs:
                mid = doc.get("message_id") or str(doc["_id"])
                if mid not in messages_by_id:
                    messages_by_id[mid] = []
                messages_by_id[mid].append(doc)

            st.markdown(f"**{len(messages_by_id)} message(s)** in this conversation")
            st.markdown("---")

            for msg_idx, (mid, steps) in enumerate(messages_by_id.items(), start=1):
                # ── Collect everything from this message's steps ──
                prompt_text = ""
                response_text = ""
                msg_cost = 0.0
                msg_input_tokens = 0
                msg_output_tokens = 0
                msg_latency = 0
                routed_model = ""
                is_cache = False
                cached_query = ""
                cached_score = 0.0
                tool_calls_all = []
                tool_results_all = []
                rag_sources = []
                rag_previews = []
                optimization_data = {}

                for s in steps:
                    trace = s.get("trace_data", {}) or {}
                    if trace.get("prompt") and not prompt_text:
                        prompt_text = trace["prompt"]
                    if trace.get("response") and not response_text:
                        response_text = trace["response"]
                    if trace.get("routed_model") and not routed_model:
                        routed_model = trace["routed_model"]
                    if s.get("step_type") == "cache_hit":
                        is_cache = True
                        cached_query = trace.get("cached_query", "")
                        cached_score = trace.get("similarity_score", 0)
                    if trace.get("sources"):
                        for src in trace["sources"]:
                            if src not in rag_sources:
                                rag_sources.append(src)
                    if trace.get("result_preview"):
                        rag_previews.extend(trace["result_preview"])
                    if trace.get("tool_calls"):
                        tool_calls_all.extend(trace["tool_calls"])
                    if s.get("tool_results"):
                        tool_results_all.extend(s["tool_results"])
                    if trace.get("optimization"):
                        optimization_data = trace["optimization"]
                    msg_cost += float(s.get("cost_usd", 0) or 0)
                    msg_input_tokens += int(s.get("input_tokens", 0) or 0)
                    msg_output_tokens += int(s.get("output_tokens", 0) or 0)
                    msg_latency += int(s.get("latency_ms", 0) or 0)

                # ── Expander title ──
                cache_tag = " 💾 CACHED" if is_cache else ""
                prompt_preview = (prompt_text[:70] + "…") if len(prompt_text) > 70 else prompt_text
                expander_title = f"#{msg_idx}  {prompt_preview}{cache_tag}"

                with st.expander(expander_title, expanded=(msg_idx <= 2)):

                    # ── Stats pills row ──
                    model_label = routed_model or ("semantic_cache" if is_cache else "—")
                    st.markdown(f"""
                    <div style="display:flex;flex-wrap:wrap;gap:8px;margin-bottom:16px;">
                        <div class="tl-pill"><span class="tl-pill-value">${msg_cost:.6f}</span><span class="tl-pill-label">cost</span></div>
                        <div class="tl-pill"><span class="tl-pill-value">{msg_input_tokens:,}</span><span class="tl-pill-label">in tokens</span></div>
                        <div class="tl-pill"><span class="tl-pill-value">{msg_output_tokens:,}</span><span class="tl-pill-label">out tokens</span></div>
                        <div class="tl-pill"><span class="tl-pill-value">{msg_latency:,} ms</span><span class="tl-pill-label">latency</span></div>
                        <div class="tl-pill"><span class="tl-pill-value">{model_label}</span><span class="tl-pill-label">model</span></div>
                        <div class="tl-pill"><span class="tl-pill-value">{len(steps)}</span><span class="tl-pill-label">steps</span></div>
                    </div>
                    """, unsafe_allow_html=True)

                    # ── USER INPUT ──
                    if prompt_text:
                        st.markdown('<div class="trace-bubble-label">👤 User Input</div>', unsafe_allow_html=True)
                        import html as _html
                        st.markdown(
                            f'<div class="trace-bubble-user">{_html.escape(prompt_text)}</div>',
                            unsafe_allow_html=True,
                        )

                    # ── CACHE HIT info ──
                    if is_cache and cached_query:
                        st.markdown('<hr class="trace-section-divider">', unsafe_allow_html=True)
                        st.markdown(f'<div class="trace-step-header">💾 Cache Hit · similarity {cached_score:.3f} · matched: <em>{_html.escape(cached_query[:80])}</em></div>', unsafe_allow_html=True)

                    # ── RAG RETRIEVAL ──
                    if rag_sources or rag_previews:
                        st.markdown('<hr class="trace-section-divider">', unsafe_allow_html=True)
                        st.markdown(f'<div class="trace-step-header">🔍 RAG Retrieval · {len(rag_previews)} chunk(s) · sources: {", ".join(rag_sources) if rag_sources else "—"}</div>', unsafe_allow_html=True)
                        if rag_previews:
                            with st.expander("Retrieved chunks", expanded=False):
                                for chunk in rag_previews:
                                    score = chunk.get("score", 0)
                                    src = chunk.get("source", "")
                                    text = chunk.get("text", "")
                                    st.markdown(f"**{src}** · score `{score:.4f}`")
                                    st.caption(text)
                                    st.markdown("---")

                    # ── TOOL CALLS ──
                    if tool_calls_all:
                        st.markdown('<hr class="trace-section-divider">', unsafe_allow_html=True)
                        st.markdown(f'<div class="trace-step-header">🔧 Tool Calls ({len(tool_calls_all)})</div>', unsafe_allow_html=True)
                        with st.expander("Tool calls & results", expanded=False):
                            for i, tc in enumerate(tool_calls_all):
                                st.markdown(f"**Call {i+1}: `{tc.get('tool', '?')}`**")
                                args = tc.get("args", {})
                                if args:
                                    st.json(args)
                            if tool_results_all:
                                st.markdown("**Results:**")
                                for tr in tool_results_all:
                                    st.markdown(f"• `{tr.get('tool', '?')}` → {str(tr.get('result', ''))[:300]}")

                    # ── ASSISTANT RESPONSE ──
                    if response_text:
                        st.markdown('<hr class="trace-section-divider">', unsafe_allow_html=True)
                        st.markdown('<div class="trace-bubble-label">🤖 Assistant Response</div>', unsafe_allow_html=True)
                        st.markdown(
                            f'<div class="trace-bubble-assistant">{_html.escape(response_text)}</div>',
                            unsafe_allow_html=True,
                        )

                    # ── OPTIMIZATION ──
                    if optimization_data:
                        st.markdown('<hr class="trace-section-divider">', unsafe_allow_html=True)
                        saved = int(optimization_data.get("saved_input_tokens_estimate", 0))
                        baseline_cost = float(optimization_data.get("baseline_cost_usd_no_history_cap", 0))
                        actual_cost = float(optimization_data.get("actual_cost_usd", 0))
                        st.markdown(
                            f'<div class="trace-step-header">⚡ Optimization · saved ~{saved:,} tokens · '
                            f'baseline ${baseline_cost:.6f} → actual ${actual_cost:.6f}</div>',
                            unsafe_allow_html=True,
                        )

    with tab_opt:
        st.markdown("**Applied optimizations**")
        st.markdown("##### Semantic Cache")
        cache_stats = next(usage_logs.aggregate([
            {"$match": {"step_type": "cache_hit"}},
            {"$group": {"_id": None, "hits": {"$sum": 1}, "total_calls": {"$sum": 1}}},
        ]), None)
        total_llm_calls = next(usage_logs.aggregate([
            {"$match": {"step_type": "llm_call"}},
            {"$group": {"_id": None, "calls": {"$sum": 1}}},
        ]), None)
        all_calls = (cache_stats or {}).get("hits", 0) + (total_llm_calls or {}).get("calls", 0)
        cache_hits = (cache_stats or {}).get("hits", 0)
        cache_hit_rate = (cache_hits / all_calls * 100) if all_calls > 0 else 0

        cache_cols = st.columns(3)
        with cache_cols[0]:
            st.markdown(f'<div class="stats-box"><div class="stats-number">{cache_hits:,}</div><div class="stats-label">Cache Hits</div></div>', unsafe_allow_html=True)
        with cache_cols[1]:
            st.markdown(f'<div class="stats-box"><div class="stats-number">{cache_hit_rate:.1f}%</div><div class="stats-label">Hit Rate</div></div>', unsafe_allow_html=True)
        with cache_cols[2]:
            st.markdown(f'<div class="stats-box"><div class="stats-number">${cache_hits * 0.0001:.6f}</div><div class="stats-label">Est. Cost Saved</div></div>', unsafe_allow_html=True)

        if cache_hits > 0:
            st.caption("Cache hits bypass the LLM entirely — zero cost, near-instant response.")
        else:
            st.info("No cache hits yet. Ask the same question twice to see semantic caching in action.")
        st.markdown("---")