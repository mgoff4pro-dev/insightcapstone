"""
Step 7b: Data visualizations.
  1. Sales trends over time
  2. Product performance comparisons
  3. Regional analysis
  4. Customer demographics and segmentation
Each function returns a matplotlib Figure (works in Jupyter and in Streamlit via st.pyplot).
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import pandas as pd
import seaborn as sns

PALETTE = ["#2563EB", "#F59E0B", "#10B981", "#EF4444", "#8B5CF6"]
sns.set_theme(style="whitegrid", palette=PALETTE)
_thousands = mtick.FuncFormatter(lambda x, _: f"{x / 1000:,.0f}k")


def _finish(fig, title):
    fig.suptitle(title, fontsize=14, fontweight="bold", x=0.01, ha="left")
    fig.tight_layout()
    return fig


# 1 ---------------------------------------------------------------- time
def sales_trend(df: pd.DataFrame) -> plt.Figure:
    monthly = df.set_index("Date")["Sales"].resample("MS").sum()
    # drop incomplete first/last months so a few days of data don't look like a sales crash
    lo, hi = df["Date"].min(), df["Date"].max()
    if lo.day != 1:
        monthly = monthly.iloc[1:]
    if hi != hi + pd.offsets.MonthEnd(0):
        monthly = monthly.iloc[:-1]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), gridspec_kw={"width_ratios": [2.2, 1]})
    axes[0].plot(monthly.index, monthly.values, color=PALETTE[0], lw=1.4, alpha=0.8, label="Monthly sales")
    axes[0].plot(monthly.index, monthly.rolling(3).mean(), color=PALETTE[1], lw=2.4, label="3-month average")
    axes[0].yaxis.set_major_formatter(_thousands)
    axes[0].set(title="Monthly sales (complete months only)", xlabel="", ylabel="Sales")
    axes[0].set_ylim(monthly.min() * 0.95, monthly.max() * 1.12)
    axes[0].legend(frameon=False, loc="upper center", ncol=2)

    # yearly comparison on AVERAGE DAILY sales so a partial year is comparable with full years
    days = df.groupby("Year")["Date"].agg(lambda d: (d.max() - d.min()).days + 1)
    lo_y, hi_y = lo.year, hi.year
    full = {y: (pd.Timestamp(y, 1, 1) >= lo) and (pd.Timestamp(y, 12, 31) <= hi) for y in days.index}
    daily = df.groupby("Year")["Sales"].sum() / days
    bars = axes[1].bar([str(y) for y in daily.index], daily.values,
                       color=[PALETTE[0] if full[y] else "white" for y in daily.index],
                       edgecolor=PALETTE[0], hatch=["" if full[y] else "///" for y in daily.index])
    for b, y, v in zip(bars, daily.index, daily.values):
        axes[1].text(b.get_x() + b.get_width() / 2, v, f"{v:,.0f}" + ("" if full[y] else "*"),
                     ha="center", va="bottom", fontsize=8)
    axes[1].set(title="Average daily sales by year", ylim=(daily.min() * 0.85, daily.max() * 1.06))
    axes[1].tick_params(axis="x", rotation=45)
    if not all(full.values()):
        axes[1].set_xlabel("* partial year (compare on daily average)", fontsize=8)
    return _finish(fig, "1. Sales trends over time")


# 2 ---------------------------------------------------------------- products
def product_performance(df: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    p = df.groupby("Product")["Sales"].sum().sort_values()
    axes[0].barh(p.index, p.values, color=PALETTE[0])
    axes[0].xaxis.set_major_formatter(_thousands)
    axes[0].set(title="Total sales by product")
    for i, v in enumerate(p.values):
        axes[0].text(v, i, f" {v / 1000:,.0f}k", va="center", fontsize=9)
    sns.boxplot(data=df, x="Product", y="Sales", ax=axes[1], order=sorted(df["Product"].unique()),
                hue="Product", legend=False)
    axes[1].set(title="Sale amount distribution", xlabel="")
    py = df.groupby(["Year", "Product"])["Sales"].sum().unstack()
    py.plot(ax=axes[2], marker="o", lw=2)
    axes[2].yaxis.set_major_formatter(_thousands)
    axes[2].set(title="Product sales by year", xlabel="")
    axes[2].legend(frameon=False, fontsize=8)
    axes[2].set_xticks(py.index)
    return _finish(fig, "2. Product performance comparison")


# 3 ---------------------------------------------------------------- regions
def regional_analysis(df: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    r = df.groupby("Region")["Sales"].sum().sort_values(ascending=False)
    axes[0].bar(r.index, r.values, color=PALETTE[2])
    axes[0].yaxis.set_major_formatter(_thousands)
    axes[0].set(title="Total sales by region")
    heat = df.pivot_table(index="Product", columns="Region", values="Sales", aggfunc="sum") / 1000
    sns.heatmap(heat, annot=True, fmt=".0f", cmap="Blues", ax=axes[1], cbar_kws={"label": "Sales (k)"})
    axes[1].set(title="Sales by product × region (k)", xlabel="", ylabel="")
    axes[1].tick_params(axis="y", rotation=0)
    sat = df.groupby("Region")["Customer_Satisfaction"].mean().reindex(r.index)
    axes[2].bar(sat.index, sat.values, color=PALETTE[1])
    axes[2].set(title="Avg. satisfaction by region", ylim=(sat.min() - 0.3, sat.max() + 0.2))
    for i, v in enumerate(sat.values):
        axes[2].text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    return _finish(fig, "3. Regional analysis")


# 4 ---------------------------------------------------------------- customers
def customer_segmentation(df: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    ag = df.groupby(["Age_Group", "Customer_Gender"], observed=True)["Sales"].sum().unstack()
    ag.plot(kind="bar", ax=axes[0], color=PALETTE[:2], width=0.8)
    axes[0].yaxis.set_major_formatter(_thousands)
    axes[0].set(title="Sales by age group and gender", xlabel="Age group")
    axes[0].tick_params(axis="x", rotation=0)
    axes[0].legend(title="", frameon=False)
    g = df.groupby("Customer_Gender")["Sales"].sum()
    axes[1].pie(g.values, labels=g.index, autopct="%1.1f%%", colors=PALETTE[:2], startangle=90, pctdistance=0.78,
                textprops={"color": "black"},
                wedgeprops={"width": 0.45, "edgecolor": "white"})
    axes[1].set(title="Share of sales by gender")
    sns.histplot(data=df, x="Customer_Age", bins=13, ax=axes[2], color=PALETTE[4])
    axes[2].axvline(df["Customer_Age"].median(), color="black", ls="--", lw=1)
    axes[2].text(df["Customer_Age"].median(), axes[2].get_ylim()[1] * 0.95,
                 f" median {df['Customer_Age'].median():.0f}", fontsize=9)
    axes[2].set(title="Customer age distribution", xlabel="Age")
    return _finish(fig, "4. Customer demographics and segmentation")


def satisfaction_vs_sales(df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 4.2))
    sns.scatterplot(data=df.sample(min(800, len(df)), random_state=1), x="Customer_Satisfaction", y="Sales",
                    hue="Product", alpha=0.6, s=18, ax=ax)
    ax.set(title="Satisfaction vs. sale amount (sample)")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return fig


ALL_CHARTS = {
    "sales_trend": sales_trend,
    "product_performance": product_performance,
    "regional_analysis": regional_analysis,
    "customer_segmentation": customer_segmentation,
    "satisfaction_vs_sales": satisfaction_vs_sales,
}


def save_all(df: pd.DataFrame, out_dir: str = "outputs/charts") -> list[str]:
    import os
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for name, fn in ALL_CHARTS.items():
        fig = fn(df)
        path = f"{out_dir}/{name}.png"
        fig.savefig(path, dpi=130, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
    return paths
