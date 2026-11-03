"""
Step 2 (Knowledge base creation) + Step 3 (custom retriever for RAG).

The knowledge base is a set of text "documents", each holding one computed
statistical summary (e.g. "Sales by Region"). Each has a `topic` tag so the
custom retriever can route a question to the right statistics.

Two retrieval paths are combined (hybrid retrieval):
  1. Keyword/entity routing  -> exact pandas statistics (deterministic, precise)
  2. Vector similarity (FAISS) -> semantic match for loosely worded questions
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import pandas as pd

from .data_analysis import full_summary


@dataclass
class KBEntry:
    topic: str
    title: str
    text: str
    metadata: dict = field(default_factory=dict)


def _table(df: pd.DataFrame) -> str:
    return df.to_string()


def build_knowledge_base(df: pd.DataFrame) -> list[KBEntry]:
    """Organise the data into structured text chunks suitable for retrieval."""
    s = full_summary(df)
    k = s["kpis"]
    entries = [
        KBEntry("overview", "Dataset overview and headline KPIs",
                f"The dataset has {k['transactions']} transactions from {df['Date'].min().date()} to "
                f"{df['Date'].max().date()}. Total sales: {k['total_sales']:,}. Average sale: {k['avg_sale']}. "
                f"Median sale: {k['median_sale']}. Average customer satisfaction: {k['avg_satisfaction']} (1-5 scale). "
                f"Top product by total sales: {k['top_product']}. Top region by total sales: {k['top_region']}. "
                f"Sales growth in {k['latest_year']} (latest complete year) vs prior year: {k['latest_yoy_growth_%']}%. "
                f"Partial years (not full 12 months, compare on average daily sales): {k['partial_years'] or 'none'}. "
                f"Products: {', '.join(sorted(df['Product'].unique()))}. Regions: {', '.join(sorted(df['Region'].unique()))}."),
        KBEntry("time", "Sales performance by year", _table(s["yearly"])),
        KBEntry("time", "Sales performance by quarter", _table(s["quarterly"])),
        KBEntry("time", "Sales performance by month", _table(s["monthly"])),
        KBEntry("time", "Seasonality: average monthly sales by calendar month",
                _table(df.groupby(df["Date"].dt.month_name(), sort=False)["Sales"].agg(["sum", "mean"]).round(2))),
        KBEntry("product", "Product performance", _table(s["products"])),
        KBEntry("region", "Regional performance", _table(s["regions"])),
        KBEntry("product_region", "Total sales by product and region", _table(s["product_region"])),
        KBEntry("customer", "Customer segmentation by gender", _table(s["gender"])),
        KBEntry("customer", "Customer segmentation by age group", _table(s["age_groups"])),
        KBEntry("customer", "Customer segmentation by age group and gender", _table(s["age_gender"])),
        KBEntry("customer", "Best-selling product for each age group", _table(s["top_product_per_age"])),
        KBEntry("statistics", "Statistical measures (mean, median, standard deviation, quartiles, skew)",
                _table(s["statistics"])),
        KBEntry("statistics", "Correlation between sales, customer age and satisfaction", _table(s["correlations"])),
        KBEntry("satisfaction", "Customer satisfaction by product and region",
                _table(df.pivot_table(index="Product", columns="Region",
                                      values="Customer_Satisfaction", aggfunc="mean").round(2))),
    ]
    return entries


# ---------------------------------------------------------------- custom retriever logic
TOPIC_KEYWORDS = {
    "time": ["trend", "time", "month", "monthly", "year", "yearly", "annual", "quarter", "q1", "q2", "q3", "q4",
             "growth", "season", "over time", "period", "when", "202"],
    "product": ["product", "widget", "item", "best-selling", "best selling", "top selling", "portfolio"],
    "region": ["region", "regional", "north", "south", "east", "west", "geograph", "market", "where"],
    "product_region": ["product and region", "by region and product", "each region", "which product in"],
    "customer": ["customer", "demographic", "age", "gender", "male", "female", "segment", "men", "women",
                 "young", "old", "audience"],
    "statistics": ["median", "standard deviation", "std", "variance", "mean", "average", "statistic",
                   "distribution", "quartile", "correlat", "skew", "spread", "range"],
    "satisfaction": ["satisf", "happy", "rating", "experience", "nps"],
}


def route_query(query: str) -> list[str]:
    """Return the topics a question is about (keyword routing)."""
    q = query.lower()
    topics = [t for t, words in TOPIC_KEYWORDS.items() if any(w in q for w in words)]
    return topics or ["overview"]


def entity_statistics(df: pd.DataFrame, query: str) -> list[KBEntry]:
    """Compute on-demand statistics for specific products/regions/years/genders named in the question."""
    q = query.lower()
    out: list[KBEntry] = []
    filters: dict[str, list] = {}
    for col in ["Product", "Region", "Customer_Gender"]:
        hits = [v for v in df[col].unique() if re.search(rf"\b{re.escape(str(v).lower())}\b", q)]
        if hits:
            filters[col] = hits
    years = [int(y) for y in re.findall(r"\b(20\d{2})\b", q) if int(y) in set(df["Year"])]
    if years:
        filters["Year"] = years

    if not filters:
        return out
    sub = df.copy()
    for col, vals in filters.items():
        sub = sub[sub[col].isin(vals)]
    label = "; ".join(f"{c}={', '.join(map(str, v))}" for c, v in filters.items())
    if sub.empty:
        return [KBEntry("filtered", f"Statistics for {label}", "No records match this filter.")]

    stats = sub["Sales"].agg(["sum", "count", "mean", "median", "std", "min", "max"]).round(2)
    out.append(KBEntry(
        "filtered", f"Statistics for {label}",
        f"Filter: {label}\nTotal sales: {stats['sum']:,.0f}\nTransactions: {int(stats['count'])}\n"
        f"Mean sale: {stats['mean']}\nMedian sale: {stats['median']}\nStd dev: {stats['std']}\n"
        f"Min: {stats['min']}  Max: {stats['max']}\n"
        f"Avg satisfaction: {sub['Customer_Satisfaction'].mean():.2f}\n"
        f"Share of all sales: {sub['Sales'].sum() / df['Sales'].sum() * 100:.2f}%",
        {"filters": label},
    ))
    for dim in ("Product", "Region"):          # e.g. "best product IN the West" -> products within West only
        if dim not in filters:
            br = sub.groupby(dim).agg(Total_Sales=("Sales", "sum"), Avg_Sale=("Sales", "mean"),
                                      Transactions=("Sales", "count"),
                                      Avg_Satisfaction=("Customer_Satisfaction", "mean")).round(2)
            br[f"Share_within_filter_%"] = (br["Total_Sales"] / br["Total_Sales"].sum() * 100).round(2)
            out.append(KBEntry("filtered", f"Sales by {dim} within {label}",
                               br.sort_values("Total_Sales", ascending=False).to_string()))
    if len(filters) == 1 and "Year" not in filters:
        out.append(KBEntry("filtered", f"Yearly trend for {label}",
                           sub.groupby("Year")["Sales"].agg(["sum", "mean", "count"]).round(2).to_string()))
    return out


def retrieve(df: pd.DataFrame, kb: list[KBEntry], query: str, max_entries: int = 6) -> list[KBEntry]:
    """Deterministic part of the custom retriever: entity stats first, then routed topic summaries."""
    results = entity_statistics(df, query)
    results.append(next(e for e in kb if e.topic == "overview"))   # always give the LLM headline context
    # round-robin across matched topics so a multi-topic question gets a slot for each topic
    buckets = [[e for e in kb if e.topic == t] for t in route_query(query)]
    for i in range(max((len(b) for b in buckets), default=0)):
        results.extend(b[i] for b in buckets if i < len(b))
    seen, unique = set(), []
    for e in results:
        if e.title not in seen:
            seen.add(e.title)
            unique.append(e)
    return unique[:max_entries]


# ---------------------------------------------------------------- research documents (PDF folder)
RESEARCH_KEYWORDS = [
    "research", "study", "studies", "literature", "paper", "best practice", "strategy", "strategies",
    "approach", "business intelligence", " bi ", "bi?", "ai ", "artificial intelligence", "machine learning",
    "forecast", "predict", "walmart", "industry", "business model", "innovation", "big data", "iot",
    "time series", "how can", "how should", "recommend", "improve", "increase", "grow",
]


def load_pdf_chunks(folder: str, chunk_size: int = 1200, overlap: int = 150) -> list[KBEntry]:
    """Load every PDF in `folder` and split it into overlapping text chunks (with source + page metadata)."""
    import glob
    import os

    from pypdf import PdfReader

    chunks: list[KBEntry] = []
    for path in sorted(glob.glob(os.path.join(folder, "*.pdf"))):
        source = os.path.splitext(os.path.basename(path))[0]
        for page_no, page in enumerate(PdfReader(path).pages, start=1):
            text = re.sub(r"\s+", " ", page.extract_text() or "").strip()
            if len(text) < 200:          # skip blank / cover / reference-only fragments
                continue
            start, part = 0, 1
            while start < len(text):
                piece = text[start:start + chunk_size]
                chunks.append(KBEntry("research", f"{source} (p. {page_no}, part {part})", piece,
                                      {"source": source, "page": page_no}))
                start += chunk_size - overlap
                part += 1
    return chunks


def is_research_question(query: str) -> bool:
    q = f" {query.lower()} "
    return any(w in q for w in RESEARCH_KEYWORDS)


def keyword_search(chunks: list[KBEntry], query: str, k: int = 3) -> list[KBEntry]:
    """Lightweight TF-IDF-style fallback ranking, used when no vector store is available."""
    import math
    from collections import Counter

    stop = {"the", "a", "an", "of", "and", "or", "to", "in", "on", "for", "is", "are", "what", "how", "which",
            "do", "does", "we", "our", "should", "can", "with", "by", "be", "it", "that", "this", "from"}
    terms = [t for t in re.findall(r"[a-z]{3,}", query.lower()) if t not in stop]
    if not terms or not chunks:
        return []
    docs = [Counter(re.findall(r"[a-z]{3,}", c.text.lower())) for c in chunks]
    n = len(docs)
    idf = {t: math.log((n + 1) / (1 + sum(t in d for d in docs))) + 1 for t in terms}
    scores = [sum(math.log1p(d[t]) * idf[t] for t in terms) for d in docs]
    ranked = sorted(range(n), key=lambda i: scores[i], reverse=True)
    return [chunks[i] for i in ranked[:k] if scores[i] > 0]
