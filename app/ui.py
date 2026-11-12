"""Small helpers shared by all pages: theme, header, cards and badges."""
import streamlit as st

DISEASE_INFO = {
    "Brain Tumor": {
        "scan": "Brain MRI scan",
        "summary": "Sorts an MRI slice into glioma, meningioma, pituitary tumor or no tumor.",
    },
    "Pneumonia": {
        "scan": "Chest X-ray",
        "summary": "Checks a chest X-ray for signs of pneumonia.",
    },
    "Diabetic Retinopathy": {
        "scan": "Retinal fundus photograph",
        "summary": "Grades diabetic retinopathy from 'No DR' up to 'Proliferative DR'.",
    },
}

RISK_CLASS = {"Low": "badge-low", "Moderate": "badge-mid", "High": "badge-high"}

CSS = """
<style>
.block-container { padding-top: 2rem; max-width: 1100px; }

section[data-testid="stSidebar"] { background: #EAF2F7; }

.page-header {
    background: linear-gradient(120deg, #0A5A82 0%, #12929C 100%);
    color: white;
    padding: 1.6rem 2rem;
    border-radius: 12px;
    margin-bottom: 1.5rem;
}
.page-title { font-size: 1.9rem; font-weight: 700; margin: 0; }
.page-sub { margin: 0.3rem 0 0 0; opacity: 0.92; font-size: 1rem; }

.card {
    background: white;
    border: 1px solid #D9E4EC;
    border-top: 4px solid #0B6E99;
    border-radius: 10px;
    padding: 1.1rem 1.2rem;
    height: 100%;
    box-shadow: 0 1px 3px rgba(16, 42, 67, 0.06);
}
.card-title { font-weight: 700; color: #0A4A6B; font-size: 1.05rem; margin-bottom: 0.3rem; }
.card-text { color: #44566C; font-size: 0.92rem; line-height: 1.45; }
.card-tag { color: #12929C; font-size: 0.78rem; font-weight: 600; text-transform: uppercase; }

.badge {
    display: inline-block;
    padding: 0.25rem 0.85rem;
    border-radius: 999px;
    font-weight: 600;
    font-size: 0.85rem;
}
.badge-low  { background: #DFF5E8; color: #17794A; }
.badge-mid  { background: #FFF1D6; color: #9A6200; }
.badge-high { background: #FCE0E0; color: #B42323; }

.result-class { font-size: 1.8rem; font-weight: 700; color: #0A4A6B; margin: 0.2rem 0 0.6rem 0; }

.disclaimer {
    background: #FFF8E6;
    border-left: 4px solid #E0A100;
    padding: 0.8rem 1rem;
    border-radius: 6px;
    font-size: 0.88rem;
    color: #5A4300;
    margin-top: 1.5rem;
}

[data-testid="stMetric"] {
    background: white;
    border: 1px solid #D9E4EC;
    border-radius: 10px;
    padding: 0.8rem 1rem;
}
</style>
"""


def apply_theme():
    st.markdown(CSS, unsafe_allow_html=True)
    st.sidebar.caption("MedicalAIPlatform | For educational use only")


def page_header(title, subtitle):
    st.markdown(
        f"""
        <div class="page-header">
            <div class="page-title">{title}</div>
            <p class="page-sub">{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def card(title, text, tag=None):
    tag_html = f'<div class="card-tag">{tag}</div>' if tag else ""
    st.markdown(
        f"""
        <div class="card">
            {tag_html}
            <div class="card-title">{title}</div>
            <div class="card-text">{text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def risk_badge(risk):
    css_class = RISK_CLASS.get(risk, "badge-mid")
    return f'<span class="badge {css_class}">{risk} risk</span>'


def disclaimer(text):
    st.markdown(f'<div class="disclaimer"><b>Note:</b> {text}</div>', unsafe_allow_html=True)


def error_text(error):
    # our own exceptions carry a short message, the full str() also has file and line
    return getattr(error, "message", str(error))
