"""
InsightForge — AI-Powered Business Intelligence Assistant (Streamlit UI)

Run:  streamlit run app.py
"""
import os

import matplotlib.pyplot as plt
import streamlit as st

from insightforge import visualizations as viz
from insightforge.data_analysis import explore, full_summary, load_data

st.set_page_config(page_title="InsightForge", page_icon="📊", layout="wide")
DEFAULT_DATA = os.path.join(os.path.dirname(__file__), "data", "sales_data.csv")
PDF_FOLDER = os.path.join(os.path.dirname(__file__), "data", "pdfs")


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.title("📊 InsightForge")
    st.caption("AI-Powered Business Intelligence Assistant")
    api_key = st.text_input("OpenAI API key", type="password", value=os.getenv("OPENAI_API_KEY", ""))
    model = st.selectbox("Model", ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini", "gpt-4.1"], index=0)
    uploaded = st.file_uploader("Upload sales data (CSV)", type="csv",
                                help="Columns: Date, Product, Region, Sales, Customer_Age, "
                                     "Customer_Gender, Customer_Satisfaction")
    use_faiss = st.toggle("Semantic search (FAISS)", value=True,
                          help="Adds vector similarity search on top of the statistics retriever")
    use_pdfs = st.toggle("Use research PDFs", value=True,
                         help="Adds passages from the BI research papers in data/pdfs to strategy questions")


@st.cache_data(show_spinner=False)
def get_data(file_bytes: bytes | None):
    import io
    return load_data(io.BytesIO(file_bytes) if file_bytes else DEFAULT_DATA)


try:
    df = get_data(uploaded.getvalue() if uploaded else None)
except ValueError as e:
    st.error(str(e))
    st.stop()

data_key = f"{uploaded.name if uploaded else 'default'}|{model}|{use_faiss}|{use_pdfs}|{bool(api_key)}"


def get_assistant():
    """Build the RAG assistant once per data/model combination and keep it (and its memory) in the session."""
    if st.session_state.get("assistant_key") != data_key:
        from insightforge.rag import InsightForgeAssistant
        with st.spinner("Building knowledge base…"):
            st.session_state.assistant = InsightForgeAssistant(
                df, api_key=api_key, model=model, use_vectorstore=use_faiss,
                pdf_folder=PDF_FOLDER if use_pdfs and os.path.isdir(PDF_FOLDER) else None)
        st.session_state.assistant_key = data_key
        st.session_state.messages = []
    return st.session_state.assistant


st.session_state.setdefault("messages", [])
tab_chat, tab_dash, tab_data, tab_eval = st.tabs(["💬 Ask InsightForge", "📈 Dashboard", "🗂️ Data & Stats", "✅ Evaluation"])

# ---------------------------------------------------------------- chat
with tab_chat:
    st.subheader("Ask a question about your business data")
    if not api_key:
        st.info("Add your OpenAI API key in the sidebar to chat. The Dashboard and Data tabs work without it.")
    else:
        examples = ["Which product is our best performer and why?",
                    "How are sales trending over time?",
                    "How can we use AI to grow sales, based on the research?",
                    "What's the median and standard deviation of sales?"]
        cols = st.columns(len(examples))
        clicked = next((ex for c, ex in zip(cols, examples) if c.button(ex, use_container_width=True)), None)

        for m in st.session_state.messages:
            with st.chat_message(m["role"]):
                st.markdown(m["content"])
                if m.get("sources"):
                    st.caption("Sources: " + " · ".join(m["sources"]))

        question = st.chat_input("e.g. How did Widget A do in the North in 2023?") or clicked
        if question:
            assistant = get_assistant()
            st.session_state.messages.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)
            with st.chat_message("assistant"):
                with st.spinner("Retrieving statistics and analysing…"):
                    try:
                        res = assistant.ask(question)
                    except Exception as e:  # surface API errors (bad key, rate limit) instead of crashing
                        st.error(f"Request failed: {e}")
                        st.stop()
                st.markdown(res["answer"])
                st.caption("Sources: " + " · ".join(res["sources"]))
                with st.expander("How this answer was built"):
                    st.markdown(f"**Standalone question (after memory):** {res['standalone_question']}")
                    st.markdown("**Extracted facts (prompt 1 of chain):**")
                    st.markdown(res["facts"])
                    st.markdown("**Retrieved context:**")
                    st.code(res["context"], language="text")
            st.session_state.messages.append({"role": "assistant", "content": res["answer"],
                                              "sources": res["sources"]})

        if st.session_state.messages and st.button("🧹 Clear conversation memory"):
            st.session_state.messages = []
            if "assistant" in st.session_state:
                st.session_state.assistant.clear_memory()
            st.rerun()

# ---------------------------------------------------------------- dashboard
with tab_dash:
    k = full_summary(df)["kpis"]
    c = st.columns(5)
    c[0].metric("Total sales", f"{k['total_sales']:,}")
    c[1].metric("Transactions", f"{k['transactions']:,}")
    c[2].metric("Average sale", f"{k['avg_sale']:,.2f}", help=f"Median {k['median_sale']}")
    c[3].metric(f"{k['latest_year']} growth", f"{k['latest_yoy_growth_%']}%",
                help="Latest complete year vs the year before" +
                     (f"; {', '.join(map(str, k['partial_years']))} is a partial year" if k['partial_years'] else ""))
    c[4].metric("Avg satisfaction", f"{k['avg_satisfaction']} / 5")
    st.caption(f"Top product: **{k['top_product']}** · Top region: **{k['top_region']}**")

    f1, f2 = st.columns(2)
    products = f1.multiselect("Filter products", sorted(df["Product"].unique()))
    regions = f2.multiselect("Filter regions", sorted(df["Region"].unique()))
    mask = df["Product"].isin(products or df["Product"].unique()) & df["Region"].isin(regions or df["Region"].unique())
    view = df[mask]
    if view.empty:
        st.warning("No data for that filter.")
    else:
        for name, fn in viz.ALL_CHARTS.items():
            fig = fn(view)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

# ---------------------------------------------------------------- data & stats
with tab_data:
    info = explore(df)
    st.markdown(f"**{info['rows']:,} rows** · {info['date_range'][0]} → {info['date_range'][1]} · "
                f"Products: {', '.join(info['products'])} · Regions: {', '.join(info['regions'])}")
    s = full_summary(df)
    st.markdown("#### Statistical measures"); st.dataframe(s["statistics"], use_container_width=True)
    a, b = st.columns(2)
    a.markdown("#### By product"); a.dataframe(s["products"], use_container_width=True)
    b.markdown("#### By region"); b.dataframe(s["regions"], use_container_width=True)
    a.markdown("#### By year"); a.dataframe(s["yearly"], use_container_width=True)
    b.markdown("#### By age group"); b.dataframe(s["age_groups"], use_container_width=True)
    with st.expander("Raw data"):
        st.dataframe(df, use_container_width=True)

# ---------------------------------------------------------------- evaluation
with tab_eval:
    st.subheader("Model evaluation (QAEvalChain)")
    st.write("Each test question has a reference answer computed directly with pandas. "
             "The assistant answers it through the full RAG pipeline, and an LLM grader (QAEvalChain) "
             "marks each answer CORRECT or INCORRECT.")
    from insightforge.evaluation import build_test_set
    st.dataframe(build_test_set(df), use_container_width=True)
    if not api_key:
        st.info("Add your OpenAI API key to run the evaluation.")
    elif st.button("▶️ Run evaluation"):
        from insightforge.evaluation import evaluate_assistant, summarize_eval
        with st.spinner("Running test questions and grading…"):
            assistant = get_assistant()
            results = evaluate_assistant(assistant, assistant.llm)
        summary = summarize_eval(results)
        m = st.columns(3)
        m[0].metric("Accuracy", f"{summary['accuracy_%']}%")
        m[1].metric("Correct", f"{summary['correct']} / {summary['total']}")
        m[2].metric("Avg latency", f"{summary['avg_latency_s']} s")
        st.dataframe(results, use_container_width=True)
        st.download_button("Download results (CSV)", results.to_csv(index=False), "evaluation_results.csv")
