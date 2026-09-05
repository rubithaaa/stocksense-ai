"""
rules.py
--------
Centralized, deterministic business rules and thresholds for StockSense AI.

Nothing in this file calls Gemini or does I/O. It is pure, testable logic that
defines *how numbers are turned into judgments* (risk levels, trend labels,
default reorder points). Keeping this separate from analytics.py makes the
thresholds easy to tune for a demo without touching calculation code, and
keeps ai_engine.py from ever having to invent its own thresholds.
"""

from __future__ import annotations
from typing import Optional


# ---------------------------------------------------------------------------
# Stock-out risk thresholds (in DAYS OF INVENTORY COVER)
# ---------------------------------------------------------------------------
# CRITICAL <= 3 days
# HIGH     >3  and <=7 days
# MEDIUM   >7  and <=14 days
# LOW      >14 days
STOCK_RISK_CRITICAL_MAX_DAYS = 3
STOCK_RISK_HIGH_MAX_DAYS = 7
STOCK_RISK_MEDIUM_MAX_DAYS = 14

STOCK_RISK_CRITICAL = "CRITICAL"
STOCK_RISK_HIGH = "HIGH"
STOCK_RISK_MEDIUM = "MEDIUM"
STOCK_RISK_LOW = "LOW"
STOCK_RISK_UNKNOWN = "UNKNOWN_NO_SALES_VELOCITY"  # avg daily sales == 0 / undefined


def classify_stock_risk(days_of_cover: Optional[float]) -> str:
    """
    Classify stock-out risk from days-of-inventory-cover.

    IMPORTANT: If days_of_cover is None (which happens when average daily
    sales is zero or cannot be computed), we NEVER invent a risk level.
    We return STOCK_RISK_UNKNOWN instead, so downstream consumers (including
    Gemini) know this is a data-insufficiency case, not a LOW-risk case.
    """
    if days_of_cover is None:
        return STOCK_RISK_UNKNOWN
    if days_of_cover < 0:
        # Negative cover shouldn't happen (would imply negative stock), but
        # guard defensively rather than silently misclassifying.
        return STOCK_RISK_CRITICAL
    if days_of_cover <= STOCK_RISK_CRITICAL_MAX_DAYS:
        return STOCK_RISK_CRITICAL
    if days_of_cover <= STOCK_RISK_HIGH_MAX_DAYS:
        return STOCK_RISK_HIGH
    if days_of_cover <= STOCK_RISK_MEDIUM_MAX_DAYS:
        return STOCK_RISK_MEDIUM
    return STOCK_RISK_LOW


# ---------------------------------------------------------------------------
# Non-moving stock detection
# ---------------------------------------------------------------------------
# A product is "non-moving" if it has recorded ZERO units sold over the last
# NON_MOVING_WINDOW_DAYS days of the dataset AND it currently holds stock > 0.
NON_MOVING_WINDOW_DAYS = 14


# ---------------------------------------------------------------------------
# Recent vs baseline sales trend detection (spike / drop)
# ---------------------------------------------------------------------------
RECENT_WINDOW_DAYS = 7          # "recent" window used for recent_daily_sales
BASELINE_WINDOW_DAYS = 30       # longer window used as the comparison baseline
SPIKE_THRESHOLD_PCT = 50.0      # recent avg >= baseline avg * 1.5  -> SPIKE
DROP_THRESHOLD_PCT = -50.0      # recent avg <= baseline avg * 0.5  -> DROP
MIN_BASELINE_UNITS_FOR_TREND = 1  # avoid divide-by-zero / noise on near-zero baselines

TREND_SPIKE = "SPIKE_UP"
TREND_DROP = "SPIKE_DOWN"
TREND_STABLE = "STABLE"
TREND_UNKNOWN = "UNKNOWN_INSUFFICIENT_DATA"


def classify_trend(recent_avg: Optional[float], baseline_avg: Optional[float]) -> str:
    """
    Classify a product's recent sales trend relative to its own baseline.
    Returns TREND_UNKNOWN rather than guessing when data is insufficient.
    """
    if recent_avg is None or baseline_avg is None:
        return TREND_UNKNOWN
    if baseline_avg < MIN_BASELINE_UNITS_FOR_TREND:
        return TREND_UNKNOWN

    pct_change = ((recent_avg - baseline_avg) / baseline_avg) * 100.0
    if pct_change >= SPIKE_THRESHOLD_PCT:
        return TREND_SPIKE
    if pct_change <= DROP_THRESHOLD_PCT:
        return TREND_DROP
    return TREND_STABLE


# ---------------------------------------------------------------------------
# Reorder level defaults
# ---------------------------------------------------------------------------
# If the source CSV does not supply an explicit reorder_level column, we can
# derive a conservative default from sales velocity and an assumed supplier
# lead time. This is clearly labeled as a DERIVED/ASSUMED value everywhere it
# is surfaced (never presented as a fact from the data).
DEFAULT_LEAD_TIME_DAYS = 7
DEFAULT_SAFETY_STOCK_DAYS = 3


def default_reorder_level(avg_daily_sales: Optional[float],
                           lead_time_days: int = DEFAULT_LEAD_TIME_DAYS,
                           safety_stock_days: int = DEFAULT_SAFETY_STOCK_DAYS) -> Optional[float]:
    """
    Compute a fallback reorder level = avg_daily_sales * (lead_time + safety_stock).

    Returns None if avg_daily_sales is None or 0, since a velocity-based
    reorder point is meaningless (and would be an invented number) when
    there is no observed sales velocity.
    """
    if not avg_daily_sales or avg_daily_sales <= 0:
        return None
    return round(avg_daily_sales * (lead_time_days + safety_stock_days), 2)


# ---------------------------------------------------------------------------
# Confidence heuristics used by ai_engine when Gemini itself doesn't set one
# (Gemini is asked to return its own confidence; these are only used as a
# deterministic sanity fallback / for validating that Gemini's confidence
# isn't wildly inconsistent with the evidence quality.)
# ---------------------------------------------------------------------------
MIN_DAYS_OF_HISTORY_FOR_HIGH_CONFIDENCE = 14


def suggest_confidence_ceiling(days_of_history: Optional[int]) -> str:
    """
    Suggests an upper bound on confidence based purely on how much history
    the dataset actually contains. Used as a guardrail note passed to Gemini,
    not as a hard override of Gemini's own stated confidence.
    """
    if days_of_history is None:
        return "LOW"
    if days_of_history < 7:
        return "LOW"
    if days_of_history < MIN_DAYS_OF_HISTORY_FOR_HIGH_CONFIDENCE:
        return "MEDIUM"
    return "HIGH"
