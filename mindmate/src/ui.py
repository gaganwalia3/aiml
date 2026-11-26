"""UI styling, theme tokens, charts, and HTML components for MindMate."""
from __future__ import annotations

import html
import math

import altair as alt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from .personality import TRAITS


def theme_tokens(dark: bool) -> dict[str, str]:
    """Return consistent color tokens for light or dark mode."""
    if dark:
        return {
            "bg": "#0B1211",
            "bg_alt": "#101B19",
            "surface": "#142220",
            "surface_2": "#1B2D2A",
            "border": "#243B37",
            "text": "#EDF5F3",
            "muted": "#93ADA8",
            "accent": "#46C4AC",
            "accent_2": "#6CE0C8",
            "accent_soft": "rgba(70, 196, 172, 0.14)",
            "hero_bg": "radial-gradient(circle at 12% 20%, rgba(70, 196, 172, 0.22), transparent 52%), radial-gradient(circle at 88% 75%, rgba(56, 145, 204, 0.16), transparent 50%), linear-gradient(135deg, #102421 0%, #132C28 100%)",
            "helpline_bg": "#261616",
            "helpline_border": "#5A2A27",
            "helpline_title": "#F5A39C",
            "helpline_text": "#F2D7D5",
            "shadow": "0 10px 30px rgba(0, 0, 0, 0.34)",
            "grid": "#263F3B",
        }
    return {
        "bg": "#F6F9F8",
        "bg_alt": "#ECF3F1",
        "surface": "#FFFFFF",
        "surface_2": "#EFF6F4",
        "border": "#DCE7E4",
        "text": "#122421",
        "muted": "#556F6A",
        "accent": "#1E7868",
        "accent_2": "#34A38E",
        "accent_soft": "rgba(30, 120, 104, 0.09)",
        "hero_bg": "radial-gradient(circle at 10% 18%, rgba(52, 163, 142, 0.16), transparent 52%), radial-gradient(circle at 90% 80%, rgba(58, 134, 194, 0.12), transparent 48%), linear-gradient(135deg, #FFFFFF 0%, #EEF6F4 100%)",
        "helpline_bg": "#FFF8F7",
        "helpline_border": "#F0C4BF",
        "helpline_title": "#8E241C",
        "helpline_text": "#3D2321",
        "shadow": "0 8px 28px rgba(18, 36, 33, 0.06)",
        "grid": "#DCE7E4",
    }


def inject_styles(dark: bool) -> None:
    """Inject modern typography, CSS variables, animations, and widget overrides."""
    t = theme_tokens(dark)
    css = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

:root {{
    --mm-bg: {t['bg']};
    --mm-bg-alt: {t['bg_alt']};
    --mm-surface: {t['surface']};
    --mm-surface-2: {t['surface_2']};
    --mm-border: {t['border']};
    --mm-text: {t['text']};
    --mm-muted: {t['muted']};
    --mm-accent: {t['accent']};
    --mm-accent-2: {t['accent_2']};
    --mm-accent-soft: {t['accent_soft']};
    --mm-shadow: {t['shadow']};
    --mm-helpline-bg: {t['helpline_bg']};
    --mm-helpline-border: {t['helpline_border']};
    --mm-helpline-title: {t['helpline_title']};
    --mm-helpline-text: {t['helpline_text']};
}}

html, body, [class*="css"], .stApp {{
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
    color: var(--mm-text);
}}

.stApp, [data-testid="stAppViewContainer"] {{
    background-color: var(--mm-bg) !important;
    color: var(--mm-text) !important;
}}

[data-testid="stHeader"] {{
    background-color: transparent !important;
}}

#MainMenu, footer {{
    visibility: hidden;
}}

[data-testid="stSidebar"] {{
    background-color: var(--mm-bg-alt) !important;
    border-right: 1px solid var(--mm-border);
}}

[data-testid="stSidebar"] * {{
    color: var(--mm-text);
}}

.block-container {{
    padding-top: 1rem;
    padding-bottom: 3rem;
    max-width: 1220px;
}}

.stCaption, [data-testid="stCaptionContainer"] p {{
    color: var(--mm-muted) !important;
}}

/* Top brand & navbar wrapper */
.mm-brand-row {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.4rem 0 0.75rem 0;
    margin-bottom: 0.4rem;
    border-bottom: 1px solid var(--mm-border);
}}
.mm-brand-left {{
    display: flex;
    align-items: center;
    gap: 0.65rem;
}}
.mm-logo-dot {{
    width: 36px;
    height: 36px;
    border-radius: 11px;
    background: linear-gradient(135deg, var(--mm-accent), var(--mm-accent-2));
    display: inline-flex;
    align-items: center;
    justify-content: center;
    color: #FFFFFF;
    font-size: 1.15rem;
    box-shadow: 0 4px 12px rgba(30, 120, 104, 0.25);
}}
.mm-brand-title {{
    font-size: 1.15rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    color: var(--mm-text);
    line-height: 1.1;
}}
.mm-brand-sub {{
    font-size: 0.75rem;
    color: var(--mm-muted);
}}

/* Landing Hero Card */
.mm-landing-hero {{
    background: {t['hero_bg']};
    border: 1px solid var(--mm-border);
    border-radius: 24px;
    padding: 2.2rem 2.3rem;
    box-shadow: var(--mm-shadow);
    margin-bottom: 1.25rem;
    position: relative;
    overflow: hidden;
}}
.mm-hero-grid {{
    display: grid;
    grid-template-columns: 1.45fr 1fr;
    gap: 2rem;
    align-items: center;
}}
@media (max-width: 900px) {{
    .mm-hero-grid {{
        grid-template-columns: 1fr;
    }}
}}
.mm-eyebrow {{
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: var(--mm-accent-soft);
    border: 1px solid rgba(30, 120, 104, 0.25);
    color: var(--mm-accent);
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    padding: 0.3rem 0.78rem;
    border-radius: 999px;
    margin-bottom: 0.9rem;
}}
.mm-hero-title {{
    font-size: 2.75rem;
    font-weight: 800;
    line-height: 1.1;
    letter-spacing: -0.035em;
    color: var(--mm-text);
    margin: 0 0 0.85rem 0;
}}
.mm-hero-title span {{
    background: linear-gradient(90deg, var(--mm-accent), var(--mm-accent-2));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}}
.mm-hero-desc {{
    font-size: 1.03rem;
    line-height: 1.6;
    color: var(--mm-muted);
    margin: 0 0 1.2rem 0;
    max-width: 580px;
}}
.mm-trust-row {{
    display: flex;
    flex-wrap: wrap;
    gap: 0.9rem;
    font-size: 0.8rem;
    font-weight: 600;
    color: var(--mm-muted);
}}

/* Ambient Breathing Orb Card inside Hero */
.mm-orb-panel {{
    background: var(--mm-surface);
    border: 1px solid var(--mm-border);
    border-radius: 20px;
    padding: 1.5rem 1.25rem;
    text-align: center;
    box-shadow: var(--mm-shadow);
}}
@keyframes mmPulse {{
    0%, 100% {{ transform: scale(0.94); box-shadow: 0 0 0 0 rgba(45, 168, 144, 0.32); }}
    50% {{ transform: scale(1.06); box-shadow: 0 0 0 18px rgba(45, 168, 144, 0); }}
}}
.mm-orb {{
    width: 108px;
    height: 108px;
    margin: 0.25rem auto 1rem auto;
    border-radius: 50%;
    background: radial-gradient(circle at 35% 30%, var(--mm-accent-2), var(--mm-accent));
    display: flex;
    align-items: center;
    justify-content: center;
    color: #FFFFFF;
    font-size: 2.2rem;
    animation: mmPulse 6s ease-in-out infinite;
}}
.mm-orb-title {{
    font-size: 0.96rem;
    font-weight: 700;
    color: var(--mm-text);
    margin-bottom: 0.25rem;
}}
.mm-orb-sub {{
    font-size: 0.8rem;
    color: var(--mm-muted);
    margin-bottom: 0.8rem;
}}
.mm-mini-stats {{
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 0.45rem;
    border-top: 1px solid var(--mm-border);
    padding-top: 0.75rem;
}}
.mm-mini-stat strong {{
    display: block;
    font-size: 0.92rem;
    color: var(--mm-text);
}}
.mm-mini-stat span {{
    font-size: 0.7rem;
    color: var(--mm-muted);
}}

/* Feature Bento Cards */
.mm-bento {{
    background: var(--mm-surface);
    border: 1px solid var(--mm-border);
    border-radius: 18px;
    padding: 1.25rem 1.25rem 1rem 1.25rem;
    box-shadow: var(--mm-shadow);
    margin-bottom: 0.55rem;
    min-height: 170px;
    transition: transform 0.18s ease, border-color 0.18s ease;
}}
.mm-bento:hover {{
    transform: translateY(-2px);
    border-color: var(--mm-accent);
}}
.mm-bento-top {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.7rem;
}}
.mm-bento-icon {{
    width: 40px;
    height: 40px;
    border-radius: 12px;
    background: var(--mm-accent-soft);
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-size: 1.25rem;
}}
.mm-bento h4 {{
    margin: 0 0 0.35rem 0;
    font-size: 1.02rem;
    font-weight: 750;
    color: var(--mm-text);
}}
.mm-bento p {{
    margin: 0;
    font-size: 0.86rem;
    color: var(--mm-muted);
    line-height: 1.5;
}}

/* Standard Cards */
.mm-card {{
    background: var(--mm-surface);
    border: 1px solid var(--mm-border);
    border-radius: 16px;
    padding: 1.1rem 1.25rem;
    box-shadow: var(--mm-shadow);
    margin-bottom: 0.9rem;
}}
.mm-card h4 {{
    margin: 0 0 0.35rem 0;
    color: var(--mm-text);
    font-size: 1rem;
    font-weight: 700;
}}
.mm-card p {{
    margin: 0;
    color: var(--mm-muted);
    font-size: 0.89rem;
    line-height: 1.5;
}}

/* Section title */
.mm-sec-head {{
    margin: 1.2rem 0 0.65rem 0;
}}
.mm-sec-head h3 {{
    margin: 0;
    font-size: 1.25rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    color: var(--mm-text);
}}
.mm-sec-head p {{
    margin: 0.15rem 0 0 0;
    font-size: 0.88rem;
    color: var(--mm-muted);
}}

/* Disclaimer banner */
.mm-disclaimer {{
    background: var(--mm-surface);
    border: 1px solid var(--mm-border);
    border-left: 4px solid var(--mm-accent);
    border-radius: 12px;
    padding: 0.7rem 1rem;
    font-size: 0.84rem;
    color: var(--mm-muted);
    margin-top: 1rem;
    box-shadow: var(--mm-shadow);
}}
.mm-disclaimer strong {{
    color: var(--mm-text);
}}

/* KPI cards */
.mm-kpi {{
    background: var(--mm-surface);
    border: 1px solid var(--mm-border);
    border-radius: 14px;
    padding: 0.9rem 1rem;
    box-shadow: var(--mm-shadow);
}}
.mm-kpi-label {{
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--mm-muted);
    margin-bottom: 0.25rem;
}}
.mm-kpi-val {{
    font-size: 1.55rem;
    font-weight: 800;
    color: var(--mm-text);
    line-height: 1.15;
}}
.mm-kpi-sub {{
    font-size: 0.77rem;
    color: var(--mm-muted);
    margin-top: 0.2rem;
}}

/* Pills */
.mm-pill {{
    display: inline-block;
    padding: 0.2rem 0.65rem;
    border-radius: 999px;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.03em;
}}
.mm-pill-low {{ background: rgba(34, 154, 102, 0.16); color: #1E9E66; border: 1px solid rgba(34, 154, 102, 0.35); }}
.mm-pill-moderate {{ background: rgba(217, 146, 26, 0.16); color: #C98210; border: 1px solid rgba(217, 146, 26, 0.35); }}
.mm-pill-elevated {{ background: rgba(214, 69, 58, 0.16); color: #D9483B; border: 1px solid rgba(214, 69, 58, 0.35); }}
.mm-pill-mid {{ background: rgba(45, 127, 184, 0.16); color: #368BC7; border: 1px solid rgba(45, 127, 184, 0.35); }}
.mm-pill-high {{ background: rgba(30, 120, 104, 0.16); color: var(--mm-accent); border: 1px solid rgba(30, 120, 104, 0.35); }}
.mm-pill-neutral {{ background: var(--mm-surface-2); color: var(--mm-muted); border: 1px solid var(--mm-border); }}

/* Trait cards */
.mm-trait-card {{
    background: var(--mm-surface);
    border: 1px solid var(--mm-border);
    border-radius: 13px;
    padding: 0.85rem 1rem;
    margin-bottom: 0.6rem;
    box-shadow: var(--mm-shadow);
}}
.mm-trait-top {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.45rem;
}}
.mm-trait-name {{
    font-weight: 700;
    font-size: 0.92rem;
    color: var(--mm-text);
}}
.mm-bar-bg {{
    width: 100%;
    height: 8px;
    background: var(--mm-surface-2);
    border-radius: 999px;
    overflow: hidden;
}}
.mm-bar-fill {{
    height: 100%;
    background: linear-gradient(90deg, var(--mm-accent), var(--mm-accent-2));
    border-radius: 999px;
}}

/* Helpline card */
.mm-helpline-card {{
    background: var(--mm-helpline-bg);
    border: 1px solid var(--mm-helpline-border);
    border-left: 4px solid #D14338;
    border-radius: 14px;
    padding: 0.9rem 1rem;
    margin-top: 0.5rem;
}}
.mm-helpline-card h5 {{
    margin: 0 0 0.45rem 0;
    color: var(--mm-helpline-title);
    font-size: 0.92rem;
    font-weight: 700;
}}
.mm-helpline-item {{
    font-size: 0.84rem;
    color: var(--mm-helpline-text);
    margin: 0.25rem 0;
    line-height: 1.4;
}}

/* Sidebar status box */
.mm-side-box {{
    background: var(--mm-surface);
    border: 1px solid var(--mm-border);
    border-radius: 14px;
    padding: 0.85rem 0.95rem;
    margin-bottom: 0.8rem;
    font-size: 0.85rem;
    box-shadow: var(--mm-shadow);
}}
.mm-side-title {{
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--mm-muted);
    margin-bottom: 0.5rem;
}}

/* Chat styling */
[data-testid="stChatMessage"] {{
    background: var(--mm-surface);
    border: 1px solid var(--mm-border);
    border-radius: 16px;
    padding: 0.85rem 1rem;
    margin-bottom: 0.55rem;
    box-shadow: var(--mm-shadow);
}}
.mm-chat-meta {{
    font-size: 0.74rem;
    color: var(--mm-muted);
    margin-top: 0.35rem;
    padding-top: 0.25rem;
    border-top: 1px dashed var(--mm-border);
}}

/* Buttons */
.stButton > button {{
    border-radius: 12px !important;
    font-weight: 600 !important;
    transition: all 0.16s ease !important;
    border: 1px solid var(--mm-border) !important;
    background-color: var(--mm-surface) !important;
    color: var(--mm-text) !important;
}}
.stButton > button:hover {{
    transform: translateY(-1px);
    border-color: var(--mm-accent) !important;
}}
.stButton > button[kind="primary"] {{
    background: linear-gradient(135deg, var(--mm-accent), var(--mm-accent-2)) !important;
    color: #FFFFFF !important;
    border: none !important;
    box-shadow: 0 4px 14px rgba(30, 120, 104, 0.22);
}}

/* Containers, expanders, forms, inputs, chat bar */
[data-testid="stExpander"], [data-testid="stForm"] {{
    background: var(--mm-surface);
    border: 1px solid var(--mm-border) !important;
    border-radius: 16px !important;
}}
[data-testid="stBottom"], [data-testid="stBottom"] > div {{
    background-color: var(--mm-bg) !important;
}}
[data-testid="stChatInput"] > div {{
    background-color: var(--mm-surface) !important;
    border: 1px solid var(--mm-border) !important;
    border-radius: 14px !important;
}}
[data-testid="stChatInput"] textarea,
[data-baseweb="input"] input,
[data-baseweb="textarea"] textarea {{
    color: var(--mm-text) !important;
    -webkit-text-fill-color: var(--mm-text) !important;
}}
[data-baseweb="input"],
[data-baseweb="base-input"],
[data-baseweb="textarea"],
[data-baseweb="select"] > div {{
    background-color: var(--mm-surface-2) !important;
    border-color: var(--mm-border) !important;
    color: var(--mm-text) !important;
}}
[data-baseweb="popover"] ul,
[role="listbox"] {{
    background-color: var(--mm-surface) !important;
    color: var(--mm-text) !important;
    border: 1px solid var(--mm-border) !important;
}}

/* Architecture step cards */
.mm-step {{
    background: var(--mm-surface);
    border: 1px solid var(--mm-border);
    border-radius: 14px;
    padding: 0.95rem 1rem;
    height: 100%;
    box-shadow: var(--mm-shadow);
}}
.mm-step-num {{
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 24px;
    height: 24px;
    border-radius: 999px;
    background: var(--mm-accent);
    color: #FFFFFF !important;
    font-size: 0.76rem;
    font-weight: 700;
    margin-right: 0.45rem;
}}

/* Breathing cadence visual box */
.mm-breath-box {{
    background: var(--mm-surface-2);
    border: 1px solid var(--mm-border);
    border-radius: 14px;
    padding: 0.9rem 1rem;
    text-align: center;
    margin-bottom: 0.8rem;
}}
.mm-breath-cycle {{
    display: flex;
    justify-content: space-between;
    gap: 0.4rem;
    margin-top: 0.55rem;
}}
.mm-breath-phase {{
    flex: 1;
    background: var(--mm-surface);
    border: 1px solid var(--mm-border);
    border-radius: 10px;
    padding: 0.45rem 0.35rem;
    font-size: 0.78rem;
    font-weight: 700;
    color: var(--mm-text);
}}
.mm-breath-phase span {{
    display: block;
    font-size: 0.7rem;
    font-weight: 500;
    color: var(--mm-muted);
}}
</style>
"""
    st.markdown(css, unsafe_allow_html=True)


def render_brand_header() -> None:
    st.markdown(
        """
<div class="mm-brand-row">
<div class="mm-brand-left">
<div class="mm-logo-dot">🧠</div>
<div>
<div class="mm-brand-title">MindMate</div>
<div class="mm-brand-sub">Personality-Aware Student Wellness Companion</div>
</div>
</div>
<div style="font-size:0.78rem; font-weight:600; color:var(--mm-muted);">
24×7 Crisis Helpline: <strong style="color:var(--mm-text);">Tele-MANAS 14416</strong> · Emergency <strong style="color:var(--mm-text);">112</strong>
</div>
</div>
""",
        unsafe_allow_html=True,
    )


def render_landing_hero(journal_count: int, has_profile: bool) -> None:
    profile_val = "Active" if has_profile else "Ready"
    st.markdown(
        f"""
<div class="mm-landing-hero">
<div class="mm-hero-grid">
<div>
<div class="mm-eyebrow">✨ Built for Students · Safety-First AI</div>
<h1 class="mm-hero-title">A calmer space for <span>student minds.</span></h1>
<p class="mm-hero-desc">
Talk through exam pressure, sleep, and overwhelm with a personality-aware companion, run a quick lifestyle risk check, and track your mood trends privately on your device.
</p>
<div class="mm-trust-row">
<span>🛡️ Rule-based crisis shield</span>
<span>🧩 Big-Five tailored tips</span>
<span>🔒 Local-only mood journal</span>
</div>
</div>
<div class="mm-orb-panel">
<div class="mm-orb">🌿</div>
<div class="mm-orb-title">Take one slow breath</div>
<div class="mm-orb-sub">Inhale 4s · Hold 2s · Exhale 6s</div>
<div class="mm-mini-stats">
<div class="mm-mini-stat"><strong>6</strong><span>Exercises</span></div>
<div class="mm-mini-stat"><strong>{html.escape(profile_val)}</strong><span>Personality</span></div>
<div class="mm-mini-stat"><strong>{journal_count}</strong><span>Journal logs</span></div>
</div>
</div>
</div>
</div>
""",
        unsafe_allow_html=True,
    )


def pill_html(text: str, variant: str = "neutral") -> str:
    return f'<span class="mm-pill mm-pill-{html.escape(variant)}">{html.escape(text)}</span>'


def kpi_card(label: str, value: str, sub: str = "") -> str:
    return (
        '<div class="mm-kpi">'
        f'<div class="mm-kpi-label">{html.escape(label)}</div>'
        f'<div class="mm-kpi-val">{html.escape(value)}</div>'
        f'<div class="mm-kpi-sub">{html.escape(sub)}</div>'
        "</div>"
    )


def trait_card(name: str, score_val: float, level_str: str, hint: str) -> str:
    pct = max(0.0, min(100.0, (score_val / 7.0) * 100.0))
    badge = pill_html(level_str.title(), level_str)
    return (
        '<div class="mm-trait-card">'
        '<div class="mm-trait-top">'
        f'<span class="mm-trait-name">{html.escape(name)} · {score_val:.1f}/7</span>'
        f"{badge}"
        "</div>"
        f'<div class="mm-bar-bg"><div class="mm-bar-fill" style="width:{pct:.1f}%"></div></div>'
        f'<div class="mm-kpi-sub" style="margin-top:0.4rem">{html.escape(hint)}</div>'
        "</div>"
    )


def helpline_card(helplines: list[dict], title: str = "Immediate Support & Helplines") -> str:
    items = "".join(
        f'<div class="mm-helpline-item">• <strong>{html.escape(str(h.get("name", "")))}</strong>: '
        f'{html.escape(str(h.get("phone", "")))}</div>'
        for h in helplines
    )
    return f'<div class="mm-helpline-card"><h5>🆘 {html.escape(title)}</h5>{items}</div>'


def build_personality_radar(scores: dict[str, float], dark: bool) -> plt.Figure:
    """Create a polar radar chart of the five Big Five trait scores (1-7 scale)."""
    t = theme_tokens(dark)
    labels = ["Extraversion", "Agreeableness", "Conscientiousness", "Emotional\nStability", "Openness"]
    values = [float(scores.get(k, 4.0)) for k in TRAITS]
    angles = np.linspace(0, 2 * math.pi, len(labels), endpoint=False).tolist()

    values_loop = values + values[:1]
    angles_loop = angles + angles[:1]

    fig, ax = plt.subplots(figsize=(4.4, 4.1), subplot_kw={"polar": True})
    fig.patch.set_alpha(0.0)
    ax.set_facecolor(t["surface"])
    ax.set_theta_offset(math.pi / 2)
    ax.set_theta_direction(-1)

    ax.plot(angles_loop, values_loop, color=t["accent"], linewidth=2.3)
    ax.fill(angles_loop, values_loop, color=t["accent"], alpha=0.22)
    ax.scatter(angles, values, color=t["accent"], s=42, zorder=5)

    ax.set_ylim(0, 7)
    ax.set_yticks([2, 4, 6])
    ax.set_yticklabels(["2", "4", "6"], color=t["muted"], fontsize=8)
    ax.set_xticks(angles)
    ax.set_xticklabels(labels, color=t["text"], fontsize=9, fontweight="bold")
    ax.tick_params(axis="x", pad=12)

    ax.grid(color=t["grid"], linestyle="--", linewidth=0.8)
    ax.spines["polar"].set_color(t["border"])
    fig.tight_layout()
    return fig


def build_risk_gauge(prob: float, level: str, dark: bool) -> alt.Chart:
    """Create a clean donut gauge chart for the predicted risk pattern match."""
    t = theme_tokens(dark)
    color_map = {"low": "#1E9E66", "moderate": "#D9921A", "elevated": "#D9483B"}
    active_color = color_map.get(level, t["accent"])

    df = pd.DataFrame(
        [
            {"segment": "Match", "value": max(0.0, min(1.0, prob))},
            {"segment": "Remaining", "value": max(0.0, 1.0 - max(0.0, min(1.0, prob)))},
        ]
    )
    donut = (
        alt.Chart(df)
        .mark_arc(innerRadius=58, outerRadius=78, cornerRadius=6)
        .encode(
            theta=alt.Theta("value:Q", stack=True),
            color=alt.Color(
                "segment:N",
                scale=alt.Scale(domain=["Match", "Remaining"], range=[active_color, t["grid"]]),
                legend=None,
            ),
        )
    )
    label_df = pd.DataFrame([{"text": f"{prob:.0%}", "sub": f"{level.upper()} MATCH"}])
    center_pct = (
        alt.Chart(label_df)
        .mark_text(fontSize=28, fontWeight=800, color=t["text"], dy=-6)
        .encode(text="text:N")
    )
    center_sub = (
        alt.Chart(label_df)
        .mark_text(fontSize=10, fontWeight=700, color=active_color, dy=16)
        .encode(text="sub:N")
    )
    return (
        (donut + center_pct + center_sub)
        .properties(width=185, height=185)
        .configure_view(strokeWidth=0, fill="transparent")
    )


def build_risk_factors_chart(inputs: dict, dark: bool) -> alt.Chart:
    """Visualize self-reported study & lifestyle load factors on a 0-100 intensity scale."""
    t = theme_tokens(dark)
    sleep_strain = {
        "Less than 5 hours": 95,
        "5-6 hours": 68,
        "7-8 hours": 20,
        "More than 8 hours": 15,
    }
    diet_strain = {"Unhealthy": 85, "Moderate": 45, "Healthy": 15}

    rows = [
        {"Factor": "Academic Pressure", "Strain": round((inputs.get("Academic Pressure", 3) / 5.0) * 100)},
        {"Factor": "Financial Stress", "Strain": round((inputs.get("Financial Stress", 3) / 5.0) * 100)},
        {"Factor": "Study Dissatisfaction", "Strain": round(((6 - inputs.get("Study Satisfaction", 3)) / 5.0) * 100)},
        {"Factor": "Study / Work Load", "Strain": round(min(100, (inputs.get("Work/Study Hours", 6) / 12.0) * 100))},
        {"Factor": "Sleep Deficit", "Strain": sleep_strain.get(inputs.get("Sleep Duration", "7-8 hours"), 40)},
        {"Factor": "Dietary Strain", "Strain": diet_strain.get(inputs.get("Dietary Habits", "Moderate"), 45)},
    ]
    df = pd.DataFrame(rows)
    df["Tier"] = df["Strain"].apply(lambda v: "High" if v >= 70 else ("Moderate" if v >= 45 else "Low"))

    chart = (
        alt.Chart(df)
        .mark_bar(cornerRadiusEnd=6, height=18)
        .encode(
            x=alt.X(
                "Strain:Q",
                scale=alt.Scale(domain=[0, 100]),
                title="Self-reported strain / load (0–100)",
                axis=alt.Axis(labelColor=t["muted"], titleColor=t["muted"], gridColor=t["grid"]),
            ),
            y=alt.Y(
                "Factor:N",
                sort="-x",
                title=None,
                axis=alt.Axis(labelColor=t["text"], labelLimit=160),
            ),
            color=alt.Color(
                "Tier:N",
                scale=alt.Scale(domain=["Low", "Moderate", "High"], range=["#1E9E66", "#D9921A", "#D9483B"]),
                legend=alt.Legend(title="Load level", labelColor=t["text"], titleColor=t["muted"]),
            ),
            tooltip=["Factor:N", "Strain:Q", "Tier:N"],
        )
        .properties(height=225)
        .configure_view(strokeWidth=0, fill="transparent")
        .configure(background="transparent")
    )
    return chart
