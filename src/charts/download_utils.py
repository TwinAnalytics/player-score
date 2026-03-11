# src/charts/download_utils.py
"""
Thin wrappers around st.altair_chart / st.plotly_chart / st.pyplot
that add a "Download PNG" button below each visual.

Usage:
    from src.charts.download_utils import altair_dl, plotly_dl, pyplot_dl

    altair_dl(chart, "team_score_vs_rank", use_container_width=True)
    plotly_dl(fig, "rankings_bar", use_container_width=True)
    pyplot_dl(fig, "pizza_chart")
"""
from __future__ import annotations

from io import BytesIO

import streamlit as st

_BTN_LABEL = "⬇ Download PNG"
_BTN_STYLE = (
    "<style>"
    ".dl-btn-row {display:flex; justify-content:flex-end; margin-top:-0.2rem;}"
    "</style>"
)


def altair_dl(chart, filename: str, **kwargs) -> None:
    """Render an Altair chart + PNG download button."""
    st.altair_chart(chart, **kwargs)
    try:
        import vl_convert as vlc  # type: ignore

        png_bytes = vlc.vegalite_to_png(chart.to_dict(), scale=2)
        st.download_button(
            _BTN_LABEL,
            data=png_bytes,
            file_name=f"{filename}.png",
            mime="image/png",
            key=f"dl_alt_{filename}",
        )
    except Exception:
        pass


def plotly_dl(fig, filename: str, **kwargs) -> object:
    """Render a Plotly figure + PNG download button. Returns the chart event."""
    # Inject PNG as default camera-icon format
    cfg = kwargs.pop("config", {})
    cfg.setdefault("toImageButtonOptions", {}).update(
        {"format": "png", "filename": filename, "scale": 2}
    )
    result = st.plotly_chart(fig, config=cfg, **kwargs)

    # Also add an explicit download button (requires kaleido)
    try:
        png_bytes = fig.to_image(format="png", scale=2)
        st.download_button(
            _BTN_LABEL,
            data=png_bytes,
            file_name=f"{filename}.png",
            mime="image/png",
            key=f"dl_plt_{filename}",
        )
    except Exception:
        pass

    return result


def pyplot_dl(fig, filename: str, **kwargs) -> None:
    """Render a matplotlib figure + PNG download button."""
    st.pyplot(fig, **kwargs)
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=180, bbox_inches="tight")
    buf.seek(0)
    st.download_button(
        _BTN_LABEL,
        data=buf.getvalue(),
        file_name=f"{filename}.png",
        mime="image/png",
        key=f"dl_mpl_{filename}",
    )
