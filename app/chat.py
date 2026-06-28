import asyncio
import time
import streamlit as st
from bson.objectid import ObjectId


def render_chat_page(agents, deps, save_turn, rename_session, dir_class, esc,
                     log_usage, approximate_tokens, calculate_cost, SYSTEM_PROMPT,
                     assistant_avatar=None, semantic_cache=None, route_query=None):
    """Chat page — AI sales agent interface."""

    st.title("Kayfa Agent")
    sid = st.session_state.current_session
    chat_messages = st.session_state.sessions[sid]["messages"]

    for msg in chat_messages:
        avatar = assistant_avatar if msg["role"] == "assistant" else None
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask me anything about Kayfa's courses, tracks, and diplomas..."):
        message_id = str(ObjectId())
        st.session_state.current_message_id = message_id
        chat_messages.append({"role": "user", "content": prompt})
        save_turn(sid, "user", prompt)
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant", avatar=assistant_avatar):
            cached = None
            if semantic_cache:
                cached = semantic_cache.find(prompt)

            if cached:
                current_user = st.session_state.get("user", {})
                log_usage(
                    conversation_id=sid,
                    user_id=current_user.get("id") or current_user.get("username", "unknown"),
                    username=current_user.get("name", "unknown"),
                    model_provider="cache",
                    model_name="semantic_cache",
                    input_tokens=0,
                    output_tokens=0,
                    tool_calls=[],
                    tool_results=[],
                    latency_ms=0,
                    step_type="cache_hit",
                    message_id=message_id,
                    trace_data={
                        "prompt": prompt,
                        "cached_query": cached["query"],
                        "similarity_score": cached["score"],
                        "response": cached["response"],
                    },
                )
                st.markdown(cached["response"])
                result_output = cached["response"]
            else:
                with st.spinner("Kayfa AI is thinking..."):
                    start_time = time.time()
                    model_history = st.session_state.sessions[sid].get("model_history", [])

                    selected_model = route_query(prompt, model_history) if route_query else "openai/gpt-oss-20b"
                    active_agent = agents[selected_model]

                    # Run the async agent in a dedicated thread so it gets a clean,
                    # unpatched event loop.  This avoids the nest_asyncio / Tornado
                    # interaction that breaks on Streamlit Cloud.
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as _pool:
                        _future = _pool.submit(
                            asyncio.run,
                            active_agent.run(prompt, deps=deps, message_history=model_history),
                        )
                        result = _future.result()

                    latency_ms = int((time.time() - start_time) * 1000)

                # --- Safe token extraction -------------------------------------------
                # pydantic-ai has renamed Usage fields across versions:
                #   input_tokens  (current)   vs  request_tokens  (older)
                #   output_tokens (current)   vs  response_tokens (older)
                # .usage() may also return None on some provider/version combos.
                # Using getattr with fallback ensures we never store None in MongoDB.
                try:
                    usage = result.usage()
                    print(f"[DEBUG chat] usage={usage}, type={type(usage)}")
                    if usage is None:
                        input_tokens = output_tokens = 0
                    else:
                        input_tokens = int(
                            getattr(usage, "input_tokens", None)
                            or getattr(usage, "request_tokens", None)
                            or 0
                        )
                        output_tokens = int(
                            getattr(usage, "output_tokens", None)
                            or getattr(usage, "response_tokens", None)
                            or 0
                        )
                except Exception as _e:
                    print(f"[DEBUG chat] result.usage() raised {type(_e).__name__}: {_e}")
                    input_tokens = output_tokens = 0

                print(f"[DEBUG chat] input_tokens={input_tokens}, output_tokens={output_tokens}, model={selected_model}")

                full_history_text = "\n".join(
                    f"{msg.get('role', '')}: {msg.get('content', '')}" for msg in chat_messages
                )
                baseline_input_tokens = approximate_tokens(SYSTEM_PROMPT) + approximate_tokens(full_history_text)
                # Fix: use selected_model (not hardcoded "openai/gpt-oss-20b") so the
                # pricing lookup is correct when the strong model is routed.
                baseline_cost = calculate_cost("groq", selected_model, baseline_input_tokens, output_tokens)

                tool_calls = []
                tool_results = []
                try:
                    for msg in result.all_messages():
                        if hasattr(msg, 'parts'):
                            for part in msg.parts:
                                if hasattr(part, 'tool_name'):
                                    tool_calls.append({
                                        "tool": part.tool_name,
                                        "args": getattr(part, 'args', {})
                                    })
                                if hasattr(part, 'tool_name') and hasattr(part, 'result'):
                                    tool_results.append({
                                        "tool": part.tool_name,
                                        "result": str(part.result)[:500]
                                    })
                except Exception:
                    pass

                current_user = st.session_state.get("user", {})
                log_usage(
                    conversation_id=sid,
                    user_id=current_user.get("id") or current_user.get("username", "unknown"),
                    username=current_user.get("name", "unknown"),
                    model_provider="groq",
                    model_name=selected_model,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    tool_calls=tool_calls,
                    tool_results=tool_results,
                    latency_ms=latency_ms,
                    step_type="llm_call",
                    message_id=message_id,
                    trace_data={
                        "prompt": prompt,
                        "routed_model": selected_model,
                        "reasoning_summary": "Classify the user's intent, retrieve Kayfa knowledge when needed, answer from sources, and capture a lead only after confirmed buying signals plus name and phone.",
                        "response": result.output,
                        "tool_calls": tool_calls,
                        "tool_results": tool_results,
                        "optimization": {
                            "applied_fix": "Capped pydantic-ai message_history to the latest five model messages.",
                            "baseline_input_tokens_no_history_cap": baseline_input_tokens,
                            "actual_input_tokens": input_tokens,
                            "saved_input_tokens_estimate": max(0, baseline_input_tokens - input_tokens),
                            "baseline_cost_usd_no_history_cap": baseline_cost,
                            "actual_cost_usd": calculate_cost("groq", selected_model, input_tokens, output_tokens),
                        },
                    }
                )
                st.session_state.pop("current_message_id", None)

                # Serialize ModelMessage objects to plain dicts before storing in
                # session state.  Streamlit Cloud may use a cross-process session
                # store that requires JSON-serializable values; pydantic-ai
                # ModelMessage objects are pydantic models, not plain dicts.
                raw_history = result.all_messages()[-5:]
                safe_history = []
                for _m in raw_history:
                    try:
                        if hasattr(_m, "model_dump"):
                            safe_history.append(_m.model_dump())
                        elif hasattr(_m, "dict"):
                            safe_history.append(_m.dict())
                        else:
                            safe_history.append(_m)
                    except Exception:
                        safe_history.append(_m)
                st.session_state.sessions[sid]["model_history"] = safe_history

                st.markdown(result.output)
                result_output = result.output

                if semantic_cache:
                    semantic_cache.store(prompt, result.output, selected_model)

        chat_messages.append({"role": "assistant", "content": result_output})
        save_turn(sid, "assistant", result_output)
        sdata = st.session_state.sessions[sid]
        if sdata["name"] in ("New Chat", "Session 1"):
            new_name = prompt[:40] + ("..." if len(prompt) > 40 else "")
            sdata["name"] = new_name
            rename_session(sid, new_name)
