import streamlit as st
from pymongo import ASCENDING
from bson.objectid import ObjectId
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime, timedelta


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

def _build_cost_over_time_chart(usage_logs):
    """Chart 1 — Line chart: cost over time, total only."""
    pipeline = [
        {"$group": {
            # Group only by date string
            "_id": {"$dateToString": {"format": "%Y-%m-%d", "date": "$timestamp"}},
            "cost": {"$sum": {"$ifNull": ["$cost_usd", 0]}},
        }},
        # Sort by the date string directly
        {"$sort": {"_id": 1}},
    ]
    rows = list(usage_logs.aggregate(pipeline))
    if not rows:
        st.info("No data for cost-over-time chart yet.")
        return

    # Map the simplified query results directly to the DataFrame
    df = pd.DataFrame([{
        "date": r["_id"],
        "cost": r["cost"],
    } for r in rows])
    df["date"] = pd.to_datetime(df["date"])

    fig = go.Figure()

    # Total line uses the pre-aggregated df directly (no Pandas groupby needed)
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["cost"],
        name="Total", mode="lines+markers",
        line=dict(color="#FDCB6E", width=3),
        marker=dict(size=5),
    ))

    fig.update_layout(
        **_PLOTLY_LAYOUT,
        title=dict(text="Cost Over Time", font=dict(size=16, color="#ffffff")),
        xaxis_title="Date",
        yaxis_title="Cost (USD)",
        yaxis_tickprefix="$",
        hovermode="x unified",
        height=420,
    )
    st.plotly_chart(fig, use_container_width=True)


def _build_cost_per_user_chart(user_rollup):
    """Chart 2 — Horizontal bar chart: cost per user, sorted descending."""
    if not user_rollup:
        st.info("No per-user data for chart yet.")
        return

    df = pd.DataFrame(user_rollup)
    df = df.sort_values("cost_usd", ascending=True)  # ascending for horizontal bars (bottom = highest)

    fig = go.Figure(go.Bar(
        x=df["cost_usd"],
        y=df["username"],
        orientation="h",
        marker=dict(
            color=df["cost_usd"],
            colorscale=[[0, "#6C5CE7"], [0.5, "#2D3BE0"], [1, "#00CEC9"]],
            line=dict(width=0),
            cornerradius=4,
        ),
        text=df["cost_usd"].apply(lambda v: f"${v:.6f}"),
        textposition="outside",
        textfont=dict(size=11, color="#c8cce0"),
        hovertemplate="<b>%{y}</b><br>Cost: $%{x:.6f}<extra></extra>",
    ))

    fig.update_layout(
        **_PLOTLY_LAYOUT,
        title=dict(text="Cost Per User", font=dict(size=16, color="#ffffff")),
        xaxis_title="Cost (USD)",
        xaxis_tickprefix="$",
        yaxis_title="",
        height=max(300, len(df) * 40 + 100),
    )
    st.plotly_chart(fig, use_container_width=True)


def _build_llm_vs_embedding_chart(usage_logs):
    """Chart 3 — Stacked bar: LLM cost vs embedding cost per day."""
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

    df = pd.DataFrame([{
        "date": r["_id"]["date"],
        "step_type": r["_id"]["step_type"],
        "cost": r["cost"],
    } for r in rows])
    df["date"] = pd.to_datetime(df["date"])

    # Map step_type to friendly labels
    type_map = {
        "llm_call": "LLM",
        "embedding_retrieval": "Embedding",
        "embedding": "Embedding",
    }
    df["label"] = df["step_type"].map(lambda s: type_map.get(s, "Other"))

    color_map = {"LLM": "#2D3BE0", "Embedding": "#00CEC9", "Other": "#636E72"}

    fig = go.Figure()
    for label in ["LLM", "Embedding", "Other"]:
        sub = df[df["label"] == label]
        if sub.empty:
            continue
        agg = sub.groupby("date")["cost"].sum().reset_index()
        fig.add_trace(go.Bar(
            x=agg["date"], y=agg["cost"],
            name=label,
            marker_color=color_map.get(label, "#636E72"),
            hovertemplate=f"<b>{label}</b><br>Date: %{{x|%Y-%m-%d}}<br>Cost: $%{{y:.6f}}<extra></extra>",
        ))

    fig.update_layout(
        **_PLOTLY_LAYOUT,
        barmode="stack",
        title=dict(text="LLM Cost vs Embedding Cost Per Day", font=dict(size=16, color="#ffffff")),
        xaxis_title="Date",
        yaxis_title="Cost (USD)",
        yaxis_tickprefix="$",
        hovermode="x unified",
        height=420,
    )
    st.plotly_chart(fig, use_container_width=True)


def _build_latency_vs_cost_scatter(usage_logs):
    """Chart 4 — Scatter: latency vs cost per message."""
    pipeline = [
        {"$match": {
            "latency_ms": {"$gt": 0},
            "username": {"$nin": [None, "", "unknown"]},
        }},
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
            x=udf["latency_ms"],
            y=udf["cost_usd"],
            mode="markers",
            name=user,
            marker=dict(
                color=_COLORS[i % len(_COLORS)],
                size=8,
                opacity=0.75,
                line=dict(width=1, color="rgba(255,255,255,0.15)"),
            ),
            customdata=udf[["message_id", "step_type"]].values,
            hovertemplate=(
                "<b>%{customdata[1]}</b><br>"
                "Latency: %{x:,} ms<br>"
                "Cost: $%{y:.6f}<br>"
                "Message: %{customdata[0]}<extra></extra>"
            ),
        ))

    fig.update_layout(
        **_PLOTLY_LAYOUT,
        title=dict(text="Latency vs Cost Per Message", font=dict(size=16, color="#ffffff")),
        xaxis_title="Latency (ms)",
        yaxis_title="Cost (USD)",
        yaxis_tickprefix="$",
        height=450,
    )
    st.plotly_chart(fig, use_container_width=True)


def render_monitoring_dashboard(usage_logs):
    """Monitoring dashboard page — cost, tokens, latency, and response traces from MongoDB."""
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

    last_updated = total_doc.get("last_updated")
    if last_updated:
        time_str = last_updated.strftime("%Y-%m-%d %H:%M:%S") if hasattr(last_updated, "strftime") else str(last_updated)
        st.caption(f"Last updated: {time_str}")

    tab_charts, tab_user, tab_conversation, tab_message, tab_trace, tab_opt = st.tabs([
        "Charts", "Per User", "Per Conversation", "Per Message", "Trace Replay", "Optimization"
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
            "user_id": "$_id.user_id",
            "username": "$_id.username",
            "cost_usd": 1,
            "total_tokens": 1,
            "calls": 1,
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
            "conversation_id": "$_id.conversation_id",
            "username": "$_id.username",
            "cost_usd": 1,
            "total_tokens": 1,
            "calls": 1,
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
            "calls": {"$sum": 1},
            "latency_ms": {"$sum": {"$ifNull": ["$latency_ms", 0]}},
            "last_seen": {"$max": "$timestamp"},
        }},
        {"$project": {
            "_id": 0,
            "message_id": "$_id.message_id",
            "conversation_id": "$_id.conversation_id",
            "username": "$_id.username",
            "cost_usd": 1,
            "total_tokens": 1,
            "calls": 1,
            "latency_ms": 1,
            "last_seen": 1,
        }},
        {"$sort": {"last_seen": -1}},
        {"$limit": 100},
    ]))

    # ── Charts tab ──
    with tab_charts:
        st.markdown(
            "<p style='color:#9ca3af;font-size:13px;margin-bottom:4px;'>"
            "Interactive visualizations of cost, usage, and performance trends.</p>",
            unsafe_allow_html=True,
        )

        # Row 1 — Cost over time (full width)
        _build_cost_over_time_chart(usage_logs)

        # Row 2 — Cost per user | LLM vs embedding side-by-side

        _build_cost_per_user_chart(user_rollup)
        
        # Row 3 — LLM vs Embedding (full width)
        _build_llm_vs_embedding_chart(usage_logs)

        # Row 4 — Scatter (full width)
        _build_latency_vs_cost_scatter(usage_logs)

    with tab_user:
        if not user_rollup:
            st.info("No per-user data yet.")
        else:
            # Inject user-card CSS once
            st.markdown("""
            <style>
            .user-card {
                background: linear-gradient(135deg, #1e2130 0%, #282b40 100%);
                border: 1px solid rgba(108, 92, 231, 0.25);
                border-radius: 16px;
                padding: 24px 20px 18px 20px;
                text-align: center;
                transition: transform 0.22s ease, box-shadow 0.22s ease, border-color 0.22s ease;
                position: relative;
                overflow: hidden;
            }
            .user-card::before {
                content: "";
                position: absolute;
                top: 0; left: 0; right: 0;
                height: 4px;
                background: linear-gradient(90deg, #2D3BE0, #6C5CE7, #00CEC9);
                border-radius: 16px 16px 0 0;
            }
            .user-card:hover {
                transform: translateY(-4px);
                box-shadow: 0 8px 32px rgba(45, 59, 224, 0.25);
                border-color: rgba(108, 92, 231, 0.5);
            }
            .user-card-avatar {
                width: 52px; height: 52px;
                border-radius: 50%;
                background: linear-gradient(135deg, #2D3BE0 0%, #6C5CE7 100%);
                display: flex; align-items: center; justify-content: center;
                margin: 0 auto 12px auto;
                font-size: 22px; font-weight: 700; color: #fff;
                letter-spacing: 1px;
                box-shadow: 0 4px 14px rgba(45, 59, 224, 0.35);
            }
            .user-card-name {
                font-size: 16px; font-weight: 600; color: #ffffff;
                margin-bottom: 14px;
                white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
            }
            .user-card-stats {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 10px 8px;
            }
            .user-card-stat {
                background: rgba(255,255,255,0.04);
                border-radius: 10px;
                padding: 10px 6px;
            }
            .user-card-stat-value {
                font-size: 15px; font-weight: 700; color: #ffffff;
            }
            .user-card-stat-label {
                font-size: 11px; color: #6b7280; margin-top: 2px;
            }
            </style>
            """, unsafe_allow_html=True)

            # Render cards in a responsive grid (3 per row)
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
                                <div class="user-card-stat">
                                    <div class="user-card-stat-value">${cost:.6f}</div>
                                    <div class="user-card-stat-label">Cost (USD)</div>
                                </div>
                                <div class="user-card-stat">
                                    <div class="user-card-stat-value">{tokens:,}</div>
                                    <div class="user-card-stat-label">Tokens</div>
                                </div>
                                <div class="user-card-stat">
                                    <div class="user-card-stat-value">{calls:,}</div>
                                    <div class="user-card-stat-label">Model Calls</div>
                                </div>
                                <div class="user-card-stat">
                                    <div class="user-card-stat-value">{latency:,} ms</div>
                                    <div class="user-card-stat-label">Avg Latency</div>
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

    # ── Shared timeline CSS (injected once for both tabs) ──
    _timeline_css = """
    <style>
    .tl-container {
        position: relative;
        padding-left: 32px;
    }
    .tl-container::before {
        content: "";
        position: absolute;
        left: 11px; top: 0; bottom: 0;
        width: 2px;
        background: linear-gradient(180deg, #2D3BE0 0%, #6C5CE7 50%, #00CEC9 100%);
        border-radius: 2px;
    }
    .tl-card {
        position: relative;
        background: linear-gradient(135deg, #1e2130 0%, #282b40 100%);
        border: 1px solid rgba(108, 92, 231, 0.2);
        border-radius: 14px;
        padding: 18px 20px;
        margin-bottom: 16px;
        transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
    }
    .tl-card:hover {
        transform: translateX(4px);
        box-shadow: 0 6px 28px rgba(45, 59, 224, 0.2);
        border-color: rgba(108, 92, 231, 0.45);
    }
    .tl-dot {
        position: absolute;
        left: -27px; top: 22px;
        width: 12px; height: 12px;
        border-radius: 50%;
        background: #6C5CE7;
        border: 2px solid #1e2130;
        box-shadow: 0 0 8px rgba(108, 92, 231, 0.5);
    }
    .tl-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 8px;
        margin-bottom: 12px;
    }
    .tl-title {
        font-size: 14px; font-weight: 600; color: #ffffff;
        display: flex; align-items: center; gap: 8px;
    }
    .tl-title-icon {
        font-size: 16px;
    }
    .tl-id {
        font-family: 'JetBrains Mono', monospace;
        font-size: 12px;
        color: #a29bfe;
        background: rgba(108, 92, 231, 0.12);
        padding: 2px 8px;
        border-radius: 6px;
    }
    .tl-badge {
        font-size: 11px;
        color: #9ca3af;
        background: rgba(255,255,255,0.05);
        padding: 3px 10px;
        border-radius: 20px;
        white-space: nowrap;
    }
    .tl-stats {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
    }
    .tl-pill {
        display: inline-flex;
        align-items: center;
        gap: 5px;
        background: rgba(255,255,255,0.04);
        border-radius: 8px;
        padding: 6px 12px;
        font-size: 12px;
    }
    .tl-pill-value {
        font-weight: 700;
        color: #ffffff;
    }
    .tl-pill-label {
        color: #6b7280;
    }
    .tl-user-tag {
        font-size: 11px;
        color: #00CEC9;
        background: rgba(0, 206, 201, 0.1);
        padding: 2px 8px;
        border-radius: 6px;
        font-weight: 500;
    }
    </style>
    """

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
                            <span class="tl-title-icon">💬</span>
                            <span class="tl-id" title="{conv_id}">{short_id}</span>
                            <span class="tl-user-tag">@{uname}</span>
                        </div>
                        <div class="tl-badge">🕐 {time_str}</div>
                    </div>
                    <div class="tl-stats">
                        <div class="tl-pill">
                            <span class="tl-pill-value">${cost:.6f}</span>
                            <span class="tl-pill-label">cost</span>
                        </div>
                        <div class="tl-pill">
                            <span class="tl-pill-value">{tokens:,}</span>
                            <span class="tl-pill-label">tokens</span>
                        </div>
                        <div class="tl-pill">
                            <span class="tl-pill-value">{calls}</span>
                            <span class="tl-pill-label">calls</span>
                        </div>
                        <div class="tl-pill">
                            <span class="tl-pill-value">{latency:,} ms</span>
                            <span class="tl-pill-label">latency</span>
                        </div>
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
                calls = int(msg.get("calls", 0) or 0)
                latency = int(msg.get("latency_ms", 0) or 0)
                last_seen = msg.get("last_seen")
                time_str = last_seen.strftime("%b %d, %Y · %H:%M") if hasattr(last_seen, "strftime") else str(last_seen or "—")
                st.markdown(f"""
                <div class="tl-card">
                    <div class="tl-dot"></div>
                    <div class="tl-header">
                        <div class="tl-title">
                            <span class="tl-title-icon">✉️</span>
                            <span class="tl-id" title="{msg_id}">{short_msg_id}</span>
                            <span class="tl-user-tag">@{uname}</span>
                            <span class="tl-badge" style="padding:2px 6px;font-size:10px;">conv: {short_conv_id}</span>
                        </div>
                        <div class="tl-badge">🕐 {time_str}</div>
                    </div>
                    <div class="tl-stats">
                        <div class="tl-pill">
                            <span class="tl-pill-value">${cost:.6f}</span>
                            <span class="tl-pill-label">cost</span>
                        </div>
                        <div class="tl-pill">
                            <span class="tl-pill-value">{tokens:,}</span>
                            <span class="tl-pill-label">tokens</span>
                        </div>
                        <div class="tl-pill">
                            <span class="tl-pill-value">{calls}</span>
                            <span class="tl-pill-label">calls</span>
                        </div>
                        <div class="tl-pill">
                            <span class="tl-pill-value">{latency:,} ms</span>
                            <span class="tl-pill-label">latency</span>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

    with tab_trace:
        message_options = [row["message_id"] for row in message_rollup]
        selected_message = st.selectbox("Message trace", message_options)
        trace_query = {"message_id": selected_message}
        if ObjectId.is_valid(selected_message):
            trace_query = {"$or": [{"message_id": selected_message}, {"_id": ObjectId(selected_message)}]}
        trace_logs = list(usage_logs.find(trace_query).sort("timestamp", ASCENDING))
        for i, doc in enumerate(trace_logs, start=1):
            trace = doc.get("trace_data", {}) or {}
            title = f"{i}. {doc.get('step_type', 'step')} · {doc.get('model_provider', '')}/{doc.get('model_name', '')}"
            with st.expander(title, expanded=True):
                st.caption(
                    f"tokens {doc.get('input_tokens', 0)} in / {doc.get('output_tokens', 0)} out · "
                    f"latency {doc.get('latency_ms', 0)} ms · cost ${float(doc.get('cost_usd', 0) or 0):.6f}"
                )
                if trace.get("prompt"):
                    st.markdown("**User prompt**")
                    st.write(trace["prompt"])
                if trace.get("query"):
                    st.markdown("**Tool query**")
                    st.code(trace["query"], language="text")
                if trace.get("tool_calls"):
                    st.markdown("**Tool calls**")
                    st.json(trace["tool_calls"])
                elif doc.get("tool_calls"):
                    st.markdown("**Tool calls**")
                    st.json(doc.get("tool_calls"))
                if trace.get("reasoning_summary"):
                    st.markdown("**Think**")
                    st.write(trace["reasoning_summary"])
                if doc.get("tool_results"):
                    st.markdown("**Tool results**")
                    st.json(doc.get("tool_results"))
                if trace.get("sources"):
                    st.markdown("**Sources**")
                    st.write(", ".join(trace["sources"]))
                if trace.get("result_shape"):
                    st.markdown("**Result shape**")
                    st.write(trace["result_shape"])
                if trace.get("result_preview"):
                    st.markdown("**Result preview**")
                    st.json(trace["result_preview"])
                if trace.get("optimization"):
                    st.markdown("**Cost optimization data**")
                    st.json(trace["optimization"])
                if trace.get("response"):
                    st.markdown("**Final response**")
                    st.write(trace["response"])

    with tab_opt:
        st.markdown("**Applied optimizations**")

        st.markdown("##### Semantic Cache")
        cache_stats = next(usage_logs.aggregate([
            {"$match": {"step_type": "cache_hit"}},
            {"$group": {
                "_id": None,
                "hits": {"$sum": 1},
                "total_calls": {"$sum": 1},
            }},
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
        st.markdown("##### Message History Window")
        st.write(
            "The chat runtime keeps only the latest five model messages in `message_history` instead of resending "
            "the whole conversation on every turn. Each LLM log stores the actual billed input tokens and an "
            "estimated uncapped-history baseline for comparison."
        )
        opt_doc = next(usage_logs.aggregate([
            {"$match": {
                "step_type": "llm_call",
                "trace_data.optimization.baseline_input_tokens_no_history_cap": {"$exists": True},
            }},
            {"$group": {
                "_id": None,
                "turns": {"$sum": 1},
                "actual_input_tokens": {"$sum": {"$ifNull": ["$input_tokens", 0]}},
                "baseline_input_tokens": {"$sum": {"$ifNull": ["$trace_data.optimization.baseline_input_tokens_no_history_cap", 0]}},
                "actual_cost": {"$sum": {"$ifNull": ["$cost_usd", 0]}},
                "baseline_cost": {"$sum": {"$ifNull": ["$trace_data.optimization.baseline_cost_usd_no_history_cap", 0]}},
            }},
            {"$project": {
                "_id": 0,
                "turns": 1,
                "actual_input_tokens": 1,
                "baseline_input_tokens": 1,
                "saved_input_tokens": {"$subtract": ["$baseline_input_tokens", "$actual_input_tokens"]},
                "actual_cost": 1,
                "baseline_cost": 1,
                "estimated_saved_cost": {"$subtract": ["$baseline_cost", "$actual_cost"]},
            }},
        ]), None)
        if opt_doc:
            opt_cols = st.columns(4)
            with opt_cols[0]:
                st.markdown(f'<div class="stats-box"><div class="stats-number">{int(opt_doc.get("turns", 0)):,}</div><div class="stats-label">Measured Turns</div></div>', unsafe_allow_html=True)
            with opt_cols[1]:
                st.markdown(f'<div class="stats-box"><div class="stats-number">{max(0, int(opt_doc.get("saved_input_tokens", 0) or 0)):,}</div><div class="stats-label">Input Tokens Saved</div></div>', unsafe_allow_html=True)
            with opt_cols[2]:
                st.markdown(f'<div class="stats-box"><div class="stats-number">${float(opt_doc.get("actual_cost", 0) or 0):.6f}</div><div class="stats-label">Actual LLM Cost</div></div>', unsafe_allow_html=True)
            with opt_cols[3]:
                st.markdown(f'<div class="stats-box"><div class="stats-number">${max(0, float(opt_doc.get("estimated_saved_cost", 0) or 0)):.6f}</div><div class="stats-label">Estimated Cost Saved</div></div>', unsafe_allow_html=True)
            st.caption("Baseline is estimated from the full stored chat transcript plus the system prompt; actual cost comes from provider usage.")
        else:
            st.info("No optimization comparison yet. Send a chat message, then return here to see before/after estimates.")
