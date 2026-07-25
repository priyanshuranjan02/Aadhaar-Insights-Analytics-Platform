"""
===============================================================================
Aadhaar Insights Analytics Platform
Reusable KPI / Metric Components
===============================================================================
Author : Priyanshu Ranjan
===============================================================================
"""

from __future__ import annotations

import streamlit as st


# =============================================================================
# Single KPI Card
# =============================================================================

def metric_card(
    title: str,
    value,
    delta: str | None = None,
    help_text: str | None = None,
):
    """
    Display a single KPI metric.

    Parameters
    ----------
    title : str
        Metric title.

    value : Any
        Metric value.

    delta : str, optional
        Optional delta text.

    help_text : str, optional
        Tooltip displayed on hover.
    """

    st.metric(
        label=title,
        value=value,
        delta=delta,
        help=help_text,
    )


# =============================================================================
# KPI Row
# =============================================================================

def metric_row(metrics: list[dict]):
    """
    Display multiple KPI cards in one row.

    Example
    -------
    metrics = [
        {"title": "Total Records", "value": "3.37 M"},
        {"title": "States", "value": 36},
        {"title": "Districts", "value": 964},
        {"title": "ML Models", "value": 3},
    ]
    """

    columns = st.columns(len(metrics))

    for column, metric in zip(columns, metrics):

        with column:

            metric_card(
                title=metric["title"],
                value=metric["value"],
                delta=metric.get("delta"),
                help_text=metric.get("help"),
            )


# =============================================================================
# Section Header
# =============================================================================

def section_header(title: str):
    """
    Display a consistent section header.
    """

    st.markdown("---")
    st.subheader(title)


# =============================================================================
# Information Card
# =============================================================================

def info_card(title: str, body: str):
    """
    Display an information box.
    """

    st.markdown(f"### {title}")
    st.info(body)


# =============================================================================
# Success Card
# =============================================================================

def success_card(title: str, body: str):
    """
    Display a success box.
    """

    st.markdown(f"### {title}")
    st.success(body)


# =============================================================================
# Warning Card
# =============================================================================

def warning_card(title: str, body: str):
    """
    Display a warning box.
    """

    st.markdown(f"### {title}")
    st.warning(body)