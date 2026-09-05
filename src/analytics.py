"""
analytics.py
------------
ALL deterministic number-crunching lives here. Nothing in this file calls
Gemini or any LLM. This is intentional: the project's core guarantee is that
"facts" shown to the user/Gemini come from pandas/numpy arithmetic, not from
model generation.

Gemini (ai_engine.py) is only ever handed the OUTPUT of this module as
read-only "evidence" - it reasons over these numbers, it does not produce them.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import difflib

import numpy as np
import pandas as pd

from . import rules


# =============================================================================
# Basic filtering helpers
# =============================================================================

def filter_df(df: pd.DataFrame,
              product_id: Optional[str] = None,
              store_id: Optional[str] = None,
              category: Optional[str] = None) -> pd.DataFrame:
    out = df
    if product_id is not None:
        out = out[out["product_id"] == str(product_id)]
    if store_id is not None:
        out = out[out["store_id"] == str(store_id)]
    if category is not None:
        out = out[out["category"].str.lower() == str(category).lower()]
    return out


# =============================================================================
# Core metric primitives
# =============================================================================

def total_revenue(df: pd.DataFrame) -> float:
    if df.empty:
        return 0.0
    return float(df["revenue"].sum())


def total_units_sold(df: pd.DataFrame) -> float:
    if df.empty:
        return 0.0
    return float(df["units_sold"].sum())


def average_daily_sales(df: pd.DataFrame, total_days: Optional[int] = None) -> Optional[float]:
    """
    Average units sold per day across the FULL span of the dataset for this
    slice. `total_days` should be the number of calendar days spanned by the
    whole dataset (not just days that appear in this slice) so that days
    with zero sales are correctly counted as zero, not omitted.
    """
    if df.empty:
        return 0.0
    if total_days is None or total_days <= 0:
        span = (df["date"].max() - df["date"].min()).days + 1
        total_days = max(span, 1)
    avg = total_units_sold(df) / total_days
    return round(float(avg), 4)


def recent_daily_sales(df: pd.DataFrame, as_of: pd.Timestamp,
                        window_days: int = rules.RECENT_WINDOW_DAYS) -> Optional[float]:
    """Average units sold per day over the most recent `window_days` days present in the dataset."""
    if df.empty:
        return None
    window_start = as_of - pd.Timedelta(days=window_days - 1)
    recent = df[(df["date"] >= window_start) & (df["date"] <= as_of)]
    if recent.empty:
        return 0.0
    return round(float(recent["units_sold"].sum()) / window_days, 4)


def baseline_daily_sales(df: pd.DataFrame, as_of: pd.Timestamp,
                          window_days: int = rules.BASELINE_WINDOW_DAYS) -> Optional[float]:
    """Average daily sales over a longer trailing window, used as a trend baseline."""
    if df.empty:
        return None
    window_start = as_of - pd.Timedelta(days=window_days - 1)
    window = df[(df["date"] >= window_start) & (df["date"] <= as_of)]
    if window.empty:
        return 0.0
    return round(float(window["units_sold"].sum()) / window_days, 4)


def current_stock(df: pd.DataFrame) -> Optional[float]:
    """
    Current stock = current_stock value on the MOST RECENT date on record
    for this slice. If multiple rows share the max date (e.g. multiple
    stores rolled into one product-level slice), they are summed.
    """
    if df.empty:
        return None
    max_date = df["date"].max()
    latest_rows = df[df["date"] == max_date]
    return float(latest_rows["current_stock"].sum())


def reorder_level(df: pd.DataFrame, avg_daily_sales_value: Optional[float]) -> Tuple[Optional[float], bool]:
    """
    Returns (reorder_level_value, is_derived).
    Prefers an explicit reorder_level column from the source data. Falls back
    to a rules-based default (clearly marked as derived/assumed) only if the
    data doesn't supply one, and only if sales velocity is non-zero.
    """
    if df.empty:
        return None, False
    if "reorder_level" in df.columns:
        vals = df["reorder_level"].dropna()
        if not vals.empty:
            max_date = df["date"].max()
            latest = df[df["date"] == max_date]["reorder_level"].dropna()
            if not latest.empty:
                return float(latest.iloc[-1]), False
            return float(vals.iloc[-1]), False
    derived = rules.default_reorder_level(avg_daily_sales_value)
    return derived, derived is not None


def days_of_inventory_cover(stock: Optional[float], avg_daily_sales_value: Optional[float]) -> Optional[float]:
    """
    Days of cover = current_stock / average_daily_sales.
    Returns None (never a fabricated number) if avg_daily_sales is zero,
    None, or stock is None - per the explicit project requirement.
    """
    if stock is None or avg_daily_sales_value is None:
        return None
    if avg_daily_sales_value <= 0:
        return None
    return round(float(stock) / float(avg_daily_sales_value), 2)


# =============================================================================
# Product-level composite metrics
# =============================================================================

def product_metrics(df: pd.DataFrame, product_id: str, store_id: Optional[str] = None) -> Dict:
    """
    Builds the full deterministic metric set for one product (optionally
    scoped to one store). This is the atomic "evidence unit" fed to Gemini.
    """
    full_span_days = int((df["date"].max() - df["date"].min()).days) + 1
    as_of = df["date"].max()

    scoped = filter_df(df, product_id=product_id, store_id=store_id)
    if scoped.empty:
        return {
            "product_id": product_id,
            "store_id": store_id,
            "found": False,
            "message": "No rows found for this product/store combination in the loaded dataset.",
        }

    name = scoped["product_name"].iloc[-1]
    category = scoped["category"].iloc[-1]

    revenue = total_revenue(scoped)
    units = total_units_sold(scoped)
    avg_daily = average_daily_sales(scoped, total_days=full_span_days)
    recent_avg = recent_daily_sales(scoped, as_of=as_of)
    baseline_avg = baseline_daily_sales(scoped, as_of=as_of)
    stock = current_stock(scoped)
    reorder_lvl, reorder_is_derived = reorder_level(scoped, avg_daily)
    days_cover = days_of_inventory_cover(stock, avg_daily)
    stock_risk = rules.classify_stock_risk(days_cover)
    trend = rules.classify_trend(recent_avg, baseline_avg)

    below_reorder = None
    if reorder_lvl is not None and stock is not None:
        below_reorder = bool(stock <= reorder_lvl)

    non_moving = is_non_moving(scoped, as_of=as_of)

    return {
        "product_id": product_id,
        "product_name": name,
        "category": category,
        "store_id": store_id,
        "found": True,
        "dataset_span_days": full_span_days,
        "total_revenue": round(revenue, 2),
        "total_units_sold": round(units, 2),
        "average_daily_sales": avg_daily,
        "recent_daily_sales_last_7d": recent_avg,
        "baseline_daily_sales_last_30d": baseline_avg,
        "current_stock": stock,
        "reorder_level": reorder_lvl,
        "reorder_level_is_derived_estimate": reorder_is_derived,
        "is_below_reorder_level": below_reorder,
        "days_of_inventory_cover": days_cover,
        "stock_risk": stock_risk,
        "sales_trend": trend,
        "is_non_moving": non_moving,
    }


def is_non_moving(df: pd.DataFrame, as_of: pd.Timestamp,
                   window_days: int = rules.NON_MOVING_WINDOW_DAYS) -> Optional[bool]:
    if df.empty:
        return None
    window_start = as_of - pd.Timedelta(days=window_days - 1)
    recent = df[(df["date"] >= window_start) & (df["date"] <= as_of)]
    units_in_window = recent["units_sold"].sum() if not recent.empty else 0
    stock_now = current_stock(df)
    if stock_now is None:
        return None
    return bool(units_in_window == 0 and stock_now > 0)


# =============================================================================
# Dataset-wide detections and rankings
# =============================================================================

def detect_non_moving_products(df: pd.DataFrame,
                                window_days: int = rules.NON_MOVING_WINDOW_DAYS,
                                limit: int = 25) -> List[Dict]:
    as_of = df["date"].max()
    results = []
    for (pid, sid), group in df.groupby(["product_id", "store_id"]):
        if is_non_moving(group, as_of=as_of, window_days=window_days):
            results.append({
                "product_id": pid,
                "product_name": group["product_name"].iloc[-1],
                "store_id": sid,
                "store_name": group["store_name"].iloc[-1],
                "category": group["category"].iloc[-1],
                "current_stock": current_stock(group),
                "days_checked": window_days,
            })
    return results[:limit]


def detect_stock_risks(df: pd.DataFrame, risk_levels: Optional[List[str]] = None,
                        limit: int = 50) -> List[Dict]:
    """Scans every (product, store) pair and returns those matching the requested risk levels."""
    full_span_days = int((df["date"].max() - df["date"].min()).days) + 1
    as_of = df["date"].max()
    if risk_levels is None:
        risk_levels = [rules.STOCK_RISK_CRITICAL, rules.STOCK_RISK_HIGH]

    results = []
    for (pid, sid), group in df.groupby(["product_id", "store_id"]):
        avg_daily = average_daily_sales(group, total_days=full_span_days)
        stock = current_stock(group)
        days_cover = days_of_inventory_cover(stock, avg_daily)
        risk = rules.classify_stock_risk(days_cover)
        if risk in risk_levels:
            results.append({
                "product_id": pid,
                "product_name": group["product_name"].iloc[-1],
                "store_id": sid,
                "store_name": group["store_name"].iloc[-1],
                "category": group["category"].iloc[-1],
                "current_stock": stock,
                "average_daily_sales": avg_daily,
                "days_of_inventory_cover": days_cover,
                "stock_risk": risk,
            })
    # Sort ascending by days_of_inventory_cover (None / unknown last)
    results.sort(key=lambda r: (r["days_of_inventory_cover"] is None, r["days_of_inventory_cover"] or 0))
    return results[:limit]


def detect_sales_trends(df: pd.DataFrame, limit: int = 25) -> Dict[str, List[Dict]]:
    as_of = df["date"].max()
    spikes, drops = [], []
    for (pid, sid), group in df.groupby(["product_id", "store_id"]):
        recent_avg = recent_daily_sales(group, as_of=as_of)
        baseline_avg = baseline_daily_sales(group, as_of=as_of)
        trend = rules.classify_trend(recent_avg, baseline_avg)
        record = {
            "product_id": pid,
            "product_name": group["product_name"].iloc[-1],
            "store_id": sid,
            "category": group["category"].iloc[-1],
            "recent_daily_sales_last_7d": recent_avg,
            "baseline_daily_sales_last_30d": baseline_avg,
            "trend": trend,
        }
        if trend == rules.TREND_SPIKE:
            spikes.append(record)
        elif trend == rules.TREND_DROP:
            drops.append(record)
    spikes.sort(key=lambda r: r["recent_daily_sales_last_7d"] or 0, reverse=True)
    drops.sort(key=lambda r: r["recent_daily_sales_last_7d"] or 0)
    return {"spikes": spikes[:limit], "drops": drops[:limit]}


def rank_products(df: pd.DataFrame, by: str = "revenue", top_n: int = 10,
                   ascending: bool = False) -> List[Dict]:
    if by not in ("revenue", "units_sold"):
        by = "revenue"
    grouped = (
        df.groupby(["product_id", "product_name", "category"])
        .agg(total_revenue=("revenue", "sum"), total_units_sold=("units_sold", "sum"))
        .reset_index()
    )
    sort_col = "total_revenue" if by == "revenue" else "total_units_sold"
    grouped = grouped.sort_values(sort_col, ascending=ascending).head(top_n)
    grouped["total_revenue"] = grouped["total_revenue"].round(2)
    return grouped.to_dict(orient="records")


def category_analysis(df: pd.DataFrame) -> List[Dict]:
    grouped = (
        df.groupby("category")
        .agg(total_revenue=("revenue", "sum"),
             total_units_sold=("units_sold", "sum"),
             n_products=("product_id", "nunique"))
        .reset_index()
        .sort_values("total_revenue", ascending=False)
    )
    grouped["total_revenue"] = grouped["total_revenue"].round(2)
    return grouped.to_dict(orient="records")


def store_analysis(df: pd.DataFrame) -> List[Dict]:
    grouped = (
        df.groupby(["store_id", "store_name"])
        .agg(total_revenue=("revenue", "sum"),
             total_units_sold=("units_sold", "sum"),
             n_products=("product_id", "nunique"))
        .reset_index()
        .sort_values("total_revenue", ascending=False)
    )
    grouped["total_revenue"] = grouped["total_revenue"].round(2)
    return grouped.to_dict(orient="records")


def overall_summary(df: pd.DataFrame) -> Dict:
    full_span_days = int((df["date"].max() - df["date"].min()).days) + 1
    return {
        "total_revenue": round(total_revenue(df), 2),
        "total_units_sold": round(total_units_sold(df), 2),
        "dataset_span_days": full_span_days,
        "n_products": int(df["product_id"].nunique()),
        "n_stores": int(df["store_id"].nunique()),
        "n_categories": int(df["category"].nunique()),
        "top_products_by_revenue": rank_products(df, by="revenue", top_n=5),
        "critical_and_high_risk_count": len(detect_stock_risks(
            df, risk_levels=[rules.STOCK_RISK_CRITICAL, rules.STOCK_RISK_HIGH], limit=10_000)),
        "non_moving_count": len(detect_non_moving_products(df, limit=10_000)),
    }


# =============================================================================
# Natural-language query intent detection (deterministic, keyword-based)
# =============================================================================

INTENT_STOCK_RISK = "STOCK_RISK"
INTENT_NON_MOVING = "NON_MOVING_STOCK"
INTENT_TOP_SELLERS = "TOP_SELLERS"
INTENT_SALES_TREND = "SALES_TREND"
INTENT_PRODUCT_LOOKUP = "PRODUCT_LOOKUP"
INTENT_CATEGORY_ANALYSIS = "CATEGORY_ANALYSIS"
INTENT_STORE_ANALYSIS = "STORE_ANALYSIS"
INTENT_REVENUE_SUMMARY = "REVENUE_SUMMARY"
INTENT_GENERAL = "GENERAL_SUMMARY"

_INTENT_KEYWORDS = {
    INTENT_STOCK_RISK: ["stock out", "stockout", "run out", "reorder", "restock", "risk", "low stock", "cover"],
    INTENT_NON_MOVING: ["non-moving", "non moving", "dead stock", "not selling", "slow moving", "stale"],
    INTENT_TOP_SELLERS: ["top selling", "best seller", "top product", "highest revenue", "top revenue", "rank"],
    INTENT_SALES_TREND: ["spike", "drop", "trend", "increase", "decrease", "surge", "decline"],
    INTENT_CATEGORY_ANALYSIS: ["category", "categories"],
    INTENT_STORE_ANALYSIS: ["store", "stores", "branch", "location"],
    INTENT_REVENUE_SUMMARY: ["revenue", "sales total", "how much did we sell", "total sales"],
}


def _fuzzy_match(term: str, candidates: List[str], cutoff: float = 0.72) -> Optional[str]:
    if not term or not candidates:
        return None
    matches = difflib.get_close_matches(term.lower(), [c.lower() for c in candidates], n=1, cutoff=cutoff)
    if not matches:
        return None
    for c in candidates:
        if c.lower() == matches[0]:
            return c
    return None


def extract_entities(question: str, engine) -> Dict[str, Optional[str]]:
    """
    Best-effort deterministic entity extraction: tries to match tokens/phrases
    in the question against known product names, categories, and store ids
    from the currently loaded dataset. This is intentionally simple
    (substring + fuzzy match) - good enough to ground Gemini's evidence
    selection without needing an LLM call just to parse the question.
    """
    q_lower = question.lower()
    entities = {"product_id": None, "product_name": None, "category": None, "store_id": None}

    try:
        products = engine.list_products()
        categories = engine.list_categories()
        stores = engine.list_stores()
    except RuntimeError:
        return entities

    # Category: check substring first, then fuzzy on whole question
    for cat in categories:
        if cat and cat.lower() in q_lower:
            entities["category"] = cat
            break
    if entities["category"] is None:
        match = _fuzzy_match(question, categories)
        if match:
            entities["category"] = match

    # Product: substring match on product_name (longest match wins to avoid
    # a short product name accidentally matching inside a longer phrase)
    best_len = 0
    for p in products:
        pname = str(p.get("product_name", ""))
        if pname and pname.lower() in q_lower and len(pname) > best_len:
            entities["product_id"] = p.get("product_id")
            entities["product_name"] = pname
            best_len = len(pname)
    if entities["product_id"] is None:
        names = [p.get("product_name", "") for p in products]
        match = _fuzzy_match(question, names)
        if match:
            for p in products:
                if p.get("product_name") == match:
                    entities["product_id"] = p.get("product_id")
                    entities["product_name"] = match
                    break

    # Store: match store_id or store_name substrings
    for s in stores:
        sid = str(s.get("store_id", ""))
        sname = str(s.get("store_name", ""))
        if (sid and sid.lower() in q_lower) or (sname and sname.lower() in q_lower):
            entities["store_id"] = sid
            break

    return entities


def detect_intent(question: str, engine) -> Dict:
    """
    Returns {"intent": <str>, "entities": {...}}.
    Falls back to INTENT_GENERAL if no keyword matches, in which case Gemini
    still receives a broad evidence pack (overall_summary) rather than
    nothing.
    """
    q_lower = (question or "").lower()
    entities = extract_entities(question or "", engine)

    scored = []
    for intent, keywords in _INTENT_KEYWORDS.items():
        hits = sum(1 for kw in keywords if kw in q_lower)
        if hits > 0:
            scored.append((hits, intent))

    if entities.get("product_id") and not scored:
        intent = INTENT_PRODUCT_LOOKUP
    elif scored:
        scored.sort(reverse=True)
        intent = scored[0][1]
    elif entities.get("product_id"):
        intent = INTENT_PRODUCT_LOOKUP
    else:
        intent = INTENT_GENERAL

    return {"intent": intent, "entities": entities}


# =============================================================================
# Evidence assembly - the single bridge between analytics and ai_engine
# =============================================================================

def build_evidence(df: pd.DataFrame, intent: str, entities: Dict) -> Dict:
    """
    Builds a compact, JSON-serializable evidence pack scoped to the detected
    intent/entities. This is the ONLY numeric/factual content Gemini will
    ever see - it must be self-sufficient (Gemini should never need to infer
    a number that isn't literally present here).
    """
    evidence: Dict = {"intent": intent, "entities": entities}

    product_id = entities.get("product_id")
    category = entities.get("category")
    store_id = entities.get("store_id")

    if intent == INTENT_STOCK_RISK:
        if product_id:
            evidence["product_metrics"] = product_metrics(df, product_id, store_id)
        else:
            scoped = filter_df(df, category=category) if category else df
            evidence["at_risk_products"] = detect_stock_risks(scoped)

    elif intent == INTENT_NON_MOVING:
        scoped = filter_df(df, category=category) if category else df
        evidence["non_moving_products"] = detect_non_moving_products(scoped)

    elif intent == INTENT_TOP_SELLERS:
        scoped = filter_df(df, category=category, store_id=store_id) if (category or store_id) else df
        evidence["top_by_revenue"] = rank_products(scoped, by="revenue", top_n=10)
        evidence["top_by_units"] = rank_products(scoped, by="units_sold", top_n=10)

    elif intent == INTENT_SALES_TREND:
        scoped = filter_df(df, category=category, store_id=store_id) if (category or store_id) else df
        if product_id:
            evidence["product_metrics"] = product_metrics(df, product_id, store_id)
        else:
            evidence["trends"] = detect_sales_trends(scoped)

    elif intent == INTENT_PRODUCT_LOOKUP and product_id:
        evidence["product_metrics"] = product_metrics(df, product_id, store_id)

    elif intent == INTENT_CATEGORY_ANALYSIS:
        evidence["category_breakdown"] = category_analysis(df)
        if category:
            evidence["focused_category"] = category

    elif intent == INTENT_STORE_ANALYSIS:
        evidence["store_breakdown"] = store_analysis(df)

    elif intent == INTENT_REVENUE_SUMMARY:
        scoped = filter_df(df, category=category, store_id=store_id) if (category or store_id) else df
        evidence["revenue_summary"] = overall_summary(scoped)

    else:  # INTENT_GENERAL or unmatched
        evidence["overall_summary"] = overall_summary(df)
        if product_id:
            evidence["product_metrics"] = product_metrics(df, product_id, store_id)

    return evidence
