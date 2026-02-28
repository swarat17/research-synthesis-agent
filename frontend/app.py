import os

import httpx
import pandas as pd
import streamlit as st

from helpers import (
    format_confidence,
    format_cost,
    identify_expensive_nodes,
    severity_badge,
)

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Research Synthesis Agent",
    page_icon="🔬",
    layout="wide",
)

# ── Sidebar ────────────────────────────────────────────────────────────────
st.sidebar.title("🔬 Research Agent")
page = st.sidebar.radio(
    "Navigate",
    ["🔍 Research Query", "📊 Cost Dashboard"],
)


# ── API helpers ────────────────────────────────────────────────────────────

def _post(endpoint: str, payload: dict) -> dict | None:
    try:
        with httpx.Client(timeout=120.0) as client:
            resp = client.post(f"{API_URL}{endpoint}", json=payload)
        if resp.status_code != 200:
            st.warning(f"API returned {resp.status_code}: {resp.json().get('detail', resp.text)[:300]}")
            return None
        return resp.json()
    except httpx.ConnectError:
        st.error("Cannot reach the API. Make sure the server is running at " + API_URL)
        return None
    except Exception:
        st.error("An unexpected error occurred while calling the API.")
        return None


def _get(endpoint: str) -> dict | None:
    try:
        with httpx.Client(timeout=30.0) as client:
            resp = client.get(f"{API_URL}{endpoint}")
        if resp.status_code != 200:
            st.warning(f"API returned {resp.status_code}: {resp.text[:200]}")
            return None
        return resp.json()
    except httpx.ConnectError:
        st.error("Cannot reach the API. Make sure the server is running at " + API_URL)
        return None
    except Exception:
        st.error("An unexpected error occurred while calling the API.")
        return None


# ══════════════════════════════════════════════════════════════════════════
# PAGE 1 — Research Query
# ══════════════════════════════════════════════════════════════════════════
if page == "🔍 Research Query":
    st.title("🔬 Research Synthesis Agent")
    st.caption("Enter a scientific topic to fetch papers, synthesize findings, detect contradictions, and generate novel hypotheses.")

    query = st.text_area(
        "Research query",
        placeholder="e.g. transformer attention mechanisms in NLP",
        height=100,
    )
    max_papers = st.slider("Max papers to fetch", min_value=4, max_value=20, value=10, step=2)

    analyze_clicked = st.button("🔍 Analyze", type="primary")

    if analyze_clicked:
        with st.spinner("Fetching papers, synthesizing, generating hypotheses…"):
            result = _post("/analyze", {"query": query.strip(), "max_papers": max_papers})
        if result:
            for err in result.get("errors", []):
                st.warning(f"⚠️ Pipeline warning: {err}")
            st.session_state["last_result"] = result

    result = st.session_state.get("last_result")
    if not result:
        st.stop()

    # ── Papers ──────────────────────────────────────────────────────────
    with st.expander(f"📄 Papers ({len(result.get('papers', []))} fetched)", expanded=False):
        papers = result.get("papers", [])
        if papers:
            rows = []
            for p in papers:
                authors = p.get("authors", [])
                rows.append({
                    "Title": p.get("title", ""),
                    "Authors": ", ".join(authors[:2]) + (" et al." if len(authors) > 2 else ""),
                    "Year": p.get("year", ""),
                    "Source": p.get("source", ""),
                    "Citations": p.get("citation_count", 0),
                    "URL": p.get("url", ""),
                })
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("No papers in response.")

    # ── Synthesis ────────────────────────────────────────────────────────
    with st.expander("📝 Synthesis", expanded=True):
        synthesis = result.get("synthesis", "")
        if synthesis:
            st.markdown(synthesis)
            st.divider()
            st.caption("📋 Copy-friendly version:")
            st.code(synthesis, language="markdown")
        else:
            st.info("No synthesis was generated.")

    # ── Contradictions ────────────────────────────────────────────────────
    contradictions = result.get("contradictions", [])
    with st.expander(f"⚔️ Contradictions ({len(contradictions)} found)", expanded=False):
        if not contradictions:
            st.info("No contradictions detected across the fetched papers.")
        else:
            for c in contradictions:
                sev = c.get("severity", "low")
                badge = severity_badge(sev) if sev in {"high", "medium", "low"} else "❓"
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**{badge} {c.get('paper_a_title', 'Paper A')}**")
                    st.write(c.get("claim_a", ""))
                with col2:
                    st.markdown(f"**{c.get('paper_b_title', 'Paper B')}**")
                    st.write(c.get("claim_b", ""))
                st.caption(f"Topic: {c.get('topic', '—')}  |  Severity: {sev.upper()}")
                st.divider()

    # ── Hypotheses ────────────────────────────────────────────────────────
    hypotheses = result.get("hypotheses", [])
    with st.expander(f"💡 Hypotheses ({len(hypotheses)} generated)", expanded=True):
        if not hypotheses:
            st.info("No hypotheses were generated.")
        else:
            for i, h in enumerate(hypotheses, 1):
                st.markdown(f"**Hypothesis {i}:** {h.get('hypothesis', '')}")
                confidence = float(h.get("confidence", 0.0))
                st.progress(confidence, text=f"Confidence: {format_confidence(confidence)}")
                novelty = h.get("novelty", "")
                col_a, col_b = st.columns([1, 3])
                col_a.metric("Novelty", novelty.capitalize() if novelty else "—")
                col_b.markdown(f"**Suggested method:** {h.get('suggested_method', '—')}")
                st.markdown(f"*Rationale:* {h.get('rationale', '')}")
                supporting = h.get("supporting_papers", [])
                if supporting:
                    st.caption("Supporting papers: " + ", ".join(supporting))
                if i < len(hypotheses):
                    st.divider()

    # ── Cost Report ────────────────────────────────────────────────────────
    with st.expander("💰 Cost Report", expanded=False):
        cost_report = result.get("cost_report", {})
        if cost_report:
            c1, c2 = st.columns(2)
            c1.metric("Total cost", format_cost(cost_report.get("total_cost_usd", 0.0)))
            c2.metric("Total latency", f"{cost_report.get('total_latency_ms', 0.0) / 1000:.1f}s")

            breakdown = cost_report.get("breakdown", [])
            if breakdown:
                df = pd.DataFrame(breakdown)
                df_display = df[["node_name", "cost_usd", "latency_ms"]].copy()
                df_display.columns = ["Node", "Cost (USD)", "Latency (ms)"]

                st.bar_chart(df_display.set_index("Node")["Cost (USD)"])
                st.dataframe(df_display, use_container_width=True, hide_index=True)

                expensive = identify_expensive_nodes(breakdown)
                if expensive:
                    st.warning(f"High-cost nodes (>40% of total): {', '.join(expensive)}")
        else:
            st.info("No cost data available.")


# ══════════════════════════════════════════════════════════════════════════
# PAGE 2 — Cost Dashboard
# ══════════════════════════════════════════════════════════════════════════
elif page == "📊 Cost Dashboard":
    st.title("📊 Cost Dashboard")
    st.caption("Live statistics from Supabase — updated after every query.")

    if st.button("🔄 Refresh"):
        st.rerun()

    stats = _get("/stats")
    if not stats:
        st.stop()

    if stats.get("error"):
        st.warning(f"Stats partially unavailable: {stats['error']}")

    # ── Top metrics ────────────────────────────────────────────────────
    c1, c2, c3 = st.columns(3)
    c1.metric("Total queries", stats.get("total_queries", 0))
    c2.metric("Avg cost / query", format_cost(stats.get("avg_cost_usd", 0.0)))
    c3.metric("Avg latency", f"{stats.get('avg_latency_ms', 0.0) / 1000:.1f}s")

    rows = stats.get("recent_queries", [])
    if not rows:
        st.info("No queries logged yet. Run an analysis first.")
        st.stop()

    df = pd.DataFrame(rows)

    # ── Cost over time ─────────────────────────────────────────────────
    if "timestamp" in df.columns and "total_cost_usd" in df.columns:
        st.subheader("Cost over time")
        chart_df = df[["timestamp", "total_cost_usd"]].copy()
        chart_df = chart_df.sort_values("timestamp")
        st.line_chart(chart_df.set_index("timestamp")["total_cost_usd"])

    # ── Avg cost per node ──────────────────────────────────────────────
    node_costs: dict[str, list[float]] = {}
    for row in rows:
        for node in row.get("node_breakdown") or []:
            name = node.get("node_name", "unknown")
            node_costs.setdefault(name, []).append(node.get("cost_usd", 0.0))
    if node_costs:
        st.subheader("Avg cost per node")
        avg_costs = {k: sum(v) / len(v) for k, v in node_costs.items()}
        st.bar_chart(avg_costs)

    # ── Recent queries table ───────────────────────────────────────────
    st.subheader("Recent queries")
    display_cols = [
        c for c in [
            "timestamp", "query", "total_cost_usd",
            "total_latency_ms", "num_papers", "num_contradictions", "num_hypotheses",
        ]
        if c in df.columns
    ]
    st.dataframe(df[display_cols], use_container_width=True, hide_index=True)


