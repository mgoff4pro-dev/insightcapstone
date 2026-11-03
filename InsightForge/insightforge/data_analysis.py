"""
Steps 1-3: Data preparation, knowledge base creation, and advanced data summary.

Pure pandas — no LangChain or API key needed. Everything the LLM is told
comes from these functions, so the numbers it reports are computed, not guessed.
"""
from __future__ import annotations

import pandas as pd

REQUIRED_COLUMNS = [
    "Date", "Product", "Region", "Sales",
    "Customer_Age", "Customer_Gender", "Customer_Satisfaction",
]
AGE_BINS = [17, 25, 35, 45, 55, 70]
AGE_LABELS = ["18-25", "26-35", "36-45", "46-55", "56-69"]


# ---------------------------------------------------------------- Step 1 & 2
def load_data(path_or_buffer) -> pd.DataFrame:
    """Load the (pre-cleaned) sales dataset and add derived columns used for analysis."""
    df = pd.read_csv(path_or_buffer)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    df["Date"] = pd.to_datetime(df["Date"])
    df["Year"] = df["Date"].dt.year
    df["Quarter"] = df["Date"].dt.to_period("Q").astype(str)
    df["Month"] = df["Date"].dt.to_period("M").astype(str)
    df["Age_Group"] = pd.cut(df["Customer_Age"], bins=AGE_BINS, labels=AGE_LABELS)
    return df


def explore(df: pd.DataFrame) -> dict:
    """Quick exploration of the dataset (shape, columns, ranges)."""
    return {
        "rows": len(df),
        "columns": list(df.columns),
        "date_range": (df["Date"].min().date().isoformat(), df["Date"].max().date().isoformat()),
        "products": sorted(df["Product"].unique().tolist()),
        "regions": sorted(df["Region"].unique().tolist()),
        "dtypes": df.dtypes.astype(str).to_dict(),
    }


# ---------------------------------------------------------------- Step 3 summaries
def _agg(df: pd.DataFrame, by) -> pd.DataFrame:
    out = df.groupby(by, observed=True).agg(
        Total_Sales=("Sales", "sum"),
        Avg_Sale=("Sales", "mean"),
        Transactions=("Sales", "count"),
        Avg_Satisfaction=("Customer_Satisfaction", "mean"),
    ).round(2)
    out["Share_of_Sales_%"] = (out["Total_Sales"] / out["Total_Sales"].sum() * 100).round(2)
    return out


def full_years(df: pd.DataFrame) -> list[int]:
    """Calendar years fully covered by the data (a partial first/last year would distort growth figures)."""
    lo, hi = df["Date"].min(), df["Date"].max()
    return [int(y) for y in sorted(df["Year"].unique())
            if lo <= pd.Timestamp(y, 1, 1) and hi >= pd.Timestamp(y, 12, 31)]


def sales_by_time(df: pd.DataFrame, period: str = "Month") -> pd.DataFrame:
    """1. Sales performance by time period (Year / Quarter / Month)."""
    out = _agg(df, period)
    out["Growth_%"] = (out["Total_Sales"].pct_change() * 100).round(2)
    if period in ("Month", "Quarter"):
        # a period only partly covered by the data (e.g. Nov 1-4, 2028) would look like a sales crash
        lo, hi = df["Date"].min().normalize(), df["Date"].max().normalize()
        freq = "M" if period == "Month" else "Q"
        complete = [pd.Period(p, freq=freq).start_time >= lo and pd.Period(p, freq=freq).end_time.normalize() <= hi
                    for p in out.index]
        out["Complete_Period"] = complete
        out.loc[~out["Complete_Period"], "Growth_%"] = float("nan")
    if period == "Year":
        days = df.groupby("Year")["Date"].agg(lambda d: (d.max() - d.min()).days + 1)
        out["Days_Covered"] = days
        out["Complete_Year"] = out.index.isin(full_years(df))
        out["Avg_Daily_Sales"] = (out["Total_Sales"] / out["Days_Covered"]).round(2)
        # growth vs a partial year is misleading: compare it on average daily sales instead
        out.loc[~out["Complete_Year"], "Growth_%"] = float("nan")
        out["Daily_Sales_Growth_%"] = (out["Avg_Daily_Sales"].pct_change() * 100).round(2)
    return out


def product_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """2a. Product performance."""
    return _agg(df, "Product").sort_values("Total_Sales", ascending=False)


def regional_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """2b. Regional performance."""
    return _agg(df, "Region").sort_values("Total_Sales", ascending=False)


def product_region_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """2c. Product x Region total sales."""
    return df.pivot_table(index="Product", columns="Region", values="Sales", aggfunc="sum")


def customer_segmentation(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """3. Customer segmentation by demographics."""
    return {
        "by_gender": _agg(df, "Customer_Gender"),
        "by_age_group": _agg(df, "Age_Group"),
        "by_age_and_gender": _agg(df, ["Age_Group", "Customer_Gender"]),
        "top_product_per_age_group": (
            df.groupby(["Age_Group", "Product"], observed=True)["Sales"].sum()
            .reset_index().sort_values("Sales", ascending=False)
            .drop_duplicates("Age_Group").set_index("Age_Group").sort_index()
        ),
    }


def statistical_measures(df: pd.DataFrame) -> pd.DataFrame:
    """4. Statistical measures (mean, median, std, quartiles, etc.)."""
    cols = ["Sales", "Customer_Age", "Customer_Satisfaction"]
    stats = df[cols].describe().T
    stats["median"] = df[cols].median()
    stats["variance"] = df[cols].var()
    stats["skew"] = df[cols].skew()
    stats = stats.rename(columns={"50%": "Q2", "25%": "Q1", "75%": "Q3"})
    return stats[["count", "mean", "median", "std", "variance", "min", "Q1", "Q3", "max", "skew"]].round(2)


def correlations(df: pd.DataFrame) -> pd.DataFrame:
    return df[["Sales", "Customer_Age", "Customer_Satisfaction"]].corr().round(3)


def kpis(df: pd.DataFrame) -> dict:
    """Headline numbers for the dashboard."""
    yearly = df.groupby("Year")["Sales"].sum()
    fy = full_years(df)
    yearly_full = yearly.loc[fy] if len(fy) else yearly
    yoy = (yearly_full.iloc[-1] / yearly_full.iloc[-2] - 1) * 100 if len(yearly_full) > 1 else float("nan")
    partial = [int(y) for y in yearly.index if y not in fy]
    return {
        "total_sales": int(df["Sales"].sum()),
        "transactions": int(len(df)),
        "avg_sale": round(float(df["Sales"].mean()), 2),
        "median_sale": round(float(df["Sales"].median()), 2),
        "avg_satisfaction": round(float(df["Customer_Satisfaction"].mean()), 2),
        "top_product": df.groupby("Product")["Sales"].sum().idxmax(),
        "top_region": df.groupby("Region")["Sales"].sum().idxmax(),
        "latest_year": int(yearly_full.index[-1]),          # latest COMPLETE year
        "latest_yoy_growth_%": round(float(yoy), 2),
        "partial_years": partial,
    }


def full_summary(df: pd.DataFrame) -> dict:
    """Every summary the knowledge base is built from."""
    seg = customer_segmentation(df)
    return {
        "kpis": kpis(df),
        "yearly": sales_by_time(df, "Year"),
        "quarterly": sales_by_time(df, "Quarter"),
        "monthly": sales_by_time(df, "Month"),
        "products": product_analysis(df),
        "regions": regional_analysis(df),
        "product_region": product_region_matrix(df),
        "gender": seg["by_gender"],
        "age_groups": seg["by_age_group"],
        "age_gender": seg["by_age_and_gender"],
        "top_product_per_age": seg["top_product_per_age_group"],
        "statistics": statistical_measures(df),
        "correlations": correlations(df),
    }
