"""
ui/app.py — MemeIQ Streamlit Interface

Run:
    streamlit run ui/app.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import time
import numpy as np
import streamlit as st
from PIL import Image

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="MemeIQ",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
    background-color: #0A0A0F;
    color: #E8E8F0;
}

.miq-header {
    text-align: center;
    padding: 3rem 0 2rem;
}
.miq-title {
    font-size: 4rem;
    font-weight: 800;
    letter-spacing: -2px;
    background: linear-gradient(135deg, #C8FF00 0%, #00FFCC 60%, #FF6BFF 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
    line-height: 1;
}
.miq-subtitle {
    color: #666680;
    font-size: 1rem;
    font-family: 'DM Mono', monospace;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-top: 0.5rem;
}

.miq-card {
    background: #0F0F1A;
    border: 1px solid #1E1E2E;
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    position: relative;
    overflow: hidden;
}
.miq-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #C8FF00, #00FFCC);
}
.miq-card.danger::before {
    background: linear-gradient(90deg, #FF4444, #FF8800);
}
.miq-card.warning::before {
    background: linear-gradient(90deg, #FF8800, #FFCC00);
}
.miq-card.safe::before {
    background: linear-gradient(90deg, #00CC66, #C8FF00);
}

.card-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #555570;
    margin-bottom: 0.5rem;
}
.card-value {
    font-size: 1.8rem;
    font-weight: 800;
    letter-spacing: -1px;
    color: #E8E8F0;
    line-height: 1.1;
}
.card-sub {
    font-family: 'DM Mono', monospace;
    font-size: 0.8rem;
    color: #666680;
    margin-top: 0.3rem;
}

.score-bar-wrap { margin: 0.5rem 0; }
.score-bar-track {
    background: #1A1A2A;
    border-radius: 99px;
    height: 8px;
    width: 100%;
    overflow: hidden;
}
.score-bar-fill {
    height: 100%;
    border-radius: 99px;
    transition: width 0.6s ease;
}

.ocr-text-box {
    background: #0A0A14;
    border: 1px solid #1E1E2E;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    font-family: 'DM Mono', monospace;
    font-size: 0.9rem;
    line-height: 1.7;
    color: #AAAACC;
    white-space: pre-wrap;
    word-break: break-word;
}

.badge {
    display: inline-block;
    padding: 0.25rem 0.75rem;
    border-radius: 99px;
    font-size: 0.75rem;
    font-family: 'DM Mono', monospace;
    font-weight: 500;
    letter-spacing: 0.05em;
}
.badge-green  { background: #0D2E1A; color: #00CC66; border: 1px solid #00CC66; }
.badge-red    { background: #2E0D0D; color: #FF4444; border: 1px solid #FF4444; }
.badge-yellow { background: #2E1E00; color: #FFCC00; border: 1px solid #FFCC00; }
.badge-blue   { background: #0D1A2E; color: #4488FF; border: 1px solid #4488FF; }
.badge-lime   { background: #1A2E00; color: #C8FF00; border: 1px solid #C8FF00; }

.section-header {
    font-size: 0.7rem;
    font-family: 'DM Mono', monospace;
    letter-spacing: 0.25em;
    text-transform: uppercase;
    color: #444460;
    border-bottom: 1px solid #1A1A2A;
    padding-bottom: 0.5rem;
    margin: 2rem 0 1rem;
}

.stFileUploader > div { background: transparent !important; }
.stButton > button {
    background: linear-gradient(135deg, #C8FF00, #00FFCC) !important;
    color: #0A0A0F !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.75rem 2rem !important;
    width: 100%;
    letter-spacing: 0.05em;
}
.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 8px 24px rgba(200,255,0,0.2) !important;
}
.stSpinner > div { color: #C8FF00 !important; }
.stCheckbox > label { color: #888899 !important; font-size: 0.85rem; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_models(use_clip: bool):
    from main import get_models
    return get_models(use_clip=use_clip)


def score_bar(score: float, color: str = "#C8FF00") -> str:
    pct = max(0, min(100, int(score * 100)))
    return f"""
    <div class="score-bar-wrap">
      <div class="score-bar-track">
        <div class="score-bar-fill" style="width:{pct}%; background:{color};"></div>
      </div>
    </div>
    """


def sentiment_color(label: str) -> str:
    return {
        "positive": "#00CC66",
        "neutral":  "#4488FF",
        "negative": "#FF4444",
    }.get(label, "#888899")


def safe_remove_numpy(obj):
    """Recursively convert numpy types and remove arrays."""
    if isinstance(obj, dict):
        return {
            k: safe_remove_numpy(v)
            for k, v in obj.items()
            if not isinstance(v, np.ndarray)
        }
    if isinstance(obj, list):
        return [safe_remove_numpy(v) for v in obj]
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def safe_json_dumps(obj: dict) -> str:
    """JSON serialize with numpy safety and str fallback."""
    cleaned = safe_remove_numpy(obj)
    try:
        return json.dumps(cleaned, indent=2)
    except TypeError:
        return json.dumps(cleaned, indent=2, default=str)


# ─────────────────────────────────────────────
# MAIN UI
# ─────────────────────────────────────────────

def main():
    # ── Header ──────────────────────────────────────
    st.markdown("""
    <div class="miq-header">
        <h1 class="miq-title">MemeIQ</h1>
        <p class="miq-subtitle">Multimodal AI · Meme Analysis · Hate Speech Detection</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Layout ──────────────────────────────────────
    left_col, right_col = st.columns([1, 1.6], gap="large")

    with left_col:
        st.markdown('<div class="section-header">Upload Meme</div>', unsafe_allow_html=True)

        uploaded = st.file_uploader(
            label="Drop a meme image",
            type=["jpg", "jpeg", "png", "gif", "webp"],
            label_visibility="collapsed",
        )

        use_clip = st.checkbox("Enable CLIP multimodal analysis", value=True)

        if uploaded:
            image = Image.open(uploaded).convert("RGB")
            st.image(image, use_column_width=True, caption=uploaded.name)
            analyze_btn = st.button("⚡ Analyze Meme")
        else:
            st.markdown("""
            <div style="text-align:center; padding: 3rem 0; color: #333350;">
                <div style="font-size:3rem;"></div>
                <div style="font-family:'DM Mono',monospace; font-size:0.8rem; margin-top:0.5rem;">
                    Upload a meme to begin
                </div>
            </div>
            """, unsafe_allow_html=True)
            analyze_btn = False

    with right_col:
        if not uploaded:
            st.markdown("""
            <div style="padding: 5rem 2rem; text-align: center; color: #222235;">
                <div style="font-size: 5rem; margin-bottom: 1rem;"></div>
                <div style="font-family:'DM Mono',monospace; font-size:0.85rem; line-height:1.8;">
                    OCR · Sentiment · Hate Detection<br>CLIP Similarity · Zero-Shot Category
                </div>
            </div>
            """, unsafe_allow_html=True)
            return

        if analyze_btn:
            with st.spinner("Analyzing..."):
                try:
                    models = load_models(use_clip=use_clip)
                    from main import analyze_meme
                    result = analyze_meme(image, use_clip=use_clip)
                    st.session_state["result"] = result
                except Exception as e:
                    st.error(f"Analysis failed: {e}")
                    return

        result = st.session_state.get("result")
        if not result:
            st.markdown("""
            <div style="padding:4rem 2rem; text-align:center; color:#333350;">
                <div style="font-family:'DM Mono',monospace; font-size:0.85rem;">
                    Press "⚡ Analyze Meme" to start
                </div>
            </div>
            """, unsafe_allow_html=True)
            return

        # ── OCR Text ──────────────────────────────
        st.markdown('<div class="section-header">Extracted Text</div>', unsafe_allow_html=True)
        ocr = result.get("ocr", {})
        ocr_text = ocr.get("text") or "_No text detected_"
        st.markdown(f'<div class="ocr-text-box">{ocr_text}</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div style="margin-top:0.4rem; font-family:DM Mono,monospace; font-size:0.75rem; color:#444460;">'
            f'OCR confidence: {ocr.get("confidence", 0):.0%}</div>',
            unsafe_allow_html=True
        )

        # ── Sentiment + Hate Speech ───────────────
        st.markdown('<div class="section-header">Analysis</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)

        with c1:
            sent = result.get("sentiment", {})
            s_label = sent.get("label", "neutral")
            s_score = sent.get("score", 0)
            s_color = sentiment_color(s_label)
            emoji = {"positive": "😊", "neutral": "😐", "negative": "😠"}.get(s_label, "🤔")
            warn = " ⚠️" if sent.get("low_confidence") else ""

            st.markdown(f"""
            <div class="miq-card">
                <div class="card-label">Sentiment</div>
                <div class="card-value">{emoji} {s_label.title()}{warn}</div>
                {score_bar(s_score, s_color)}
                <div class="card-sub">{s_score:.0%} confidence</div>
            </div>
            """, unsafe_allow_html=True)

            scores_dict = sent.get("scores", {})
            for lbl, sc in sorted(scores_dict.items(), key=lambda x: -x[1]):
                col_hex = sentiment_color(lbl)
                st.markdown(
                    f'<div style="display:flex; justify-content:space-between; '
                    f'font-family:DM Mono,monospace; font-size:0.75rem; color:#555570; margin:2px 0;">'
                    f'<span>{lbl}</span><span style="color:{col_hex};">{sc:.0%}</span></div>',
                    unsafe_allow_html=True
                )

        with c2:
            hate = result.get("hate_speech", {})
            is_hateful   = hate.get("hateful", False)
            needs_review = hate.get("needs_review", False)
            hate_score   = hate.get("hate_score", 0)
            clip_hate    = hate.get("clip_hate_score")

            if is_hateful:
                card_cls, flag_emoji, flag_txt = "danger", "🚨", "HATEFUL"
                bar_color = "#FF4444"
            elif needs_review:
                card_cls, flag_emoji, flag_txt = "warning", "⚠️", "REVIEW"
                bar_color = "#FF8800"
            else:
                card_cls, flag_emoji, flag_txt = "safe", "✅", "SAFE"
                bar_color = "#00CC66"

            clip_line = (
                f'<div class="card-sub">CLIP hate score: {clip_hate:.0%}</div>'
                if clip_hate is not None else ""
            )

            st.markdown(f"""
            <div class="miq-card {card_cls}">
                <div class="card-label">Hate Speech</div>
                <div class="card-value">{flag_emoji} {flag_txt}</div>
                {score_bar(hate_score, bar_color)}
                <div class="card-sub">combined hate score: {hate_score:.0%}</div>
                {clip_line}
            </div>
            """, unsafe_allow_html=True)

            if needs_review:
                st.markdown(
                    '<div class="badge badge-yellow">Borderline — human review recommended</div>',
                    unsafe_allow_html=True
                )
            elif is_hateful:
                st.markdown(
                    '<div class="badge badge-red">Content flagged</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    '<div class="badge badge-green">No harmful content detected</div>',
                    unsafe_allow_html=True
                )

        # ── CLIP Multimodal ───────────────────────
        clip = result.get("clip", {})
        if clip:
            st.markdown('<div class="section-header">CLIP Multimodal</div>', unsafe_allow_html=True)
            c3, c4 = st.columns(2)

            with c3:
                similarity = clip.get("image_text_similarity", 0)
                sim_color  = "#C8FF00" if similarity > 0.25 else "#444460"
                st.markdown(f"""
                <div class="miq-card">
                    <div class="card-label">Image ↔ Text Similarity</div>
                    <div class="card-value">{similarity:.2f}</div>
                    {score_bar(max(0, similarity), sim_color)}
                    <div class="card-sub">{'Strong alignment' if similarity > 0.25 else 'Weak alignment'}</div>
                </div>
                """, unsafe_allow_html=True)

            with c4:
                img_cls   = clip.get("image_classification", {})
                top       = img_cls.get("top_label", "unknown").replace("a meme about ", "")
                top_score = img_cls.get("top_score", 0)
                st.markdown(f"""
                <div class="miq-card">
                    <div class="card-label">Image Content (CLIP)</div>
                    <div class="card-value" style="font-size:1.3rem;">{top.title()}</div>
                    {score_bar(top_score, "#00FFCC")}
                    <div class="card-sub">{top_score:.0%} confidence</div>
                </div>
                """, unsafe_allow_html=True)

        # ── Category ──────────────────────────────
        cat = result.get("category", {})
        st.markdown('<div class="section-header">Meme Category</div>', unsafe_allow_html=True)
        cat_name  = cat.get("category", "Uncategorized")
        cat_score = cat.get("score", 0)
        cat_warn  = " ⚠️" if cat.get("low_confidence") else ""

        st.markdown(f"""
        <div class="miq-card" style="background: linear-gradient(135deg, #0F0F1A, #0A1420);">
            <div class="card-label">Zero-Shot Category</div>
            <div class="card-value" style="font-size:1.5rem;">🏷️ {cat_name}{cat_warn}</div>
            {score_bar(cat_score, "#FF6BFF")}
            <div class="card-sub">{cat_score:.0%} confidence via DistilBERT-MNLI</div>
        </div>
        """, unsafe_allow_html=True)

        all_cats = cat.get("all_scores", {})
        if all_cats:
            top3 = sorted(all_cats.items(), key=lambda x: -x[1])[:3]
            cols = st.columns(3)
            for i, (lbl, sc) in enumerate(top3):
                with cols[i]:
                    st.markdown(
                        f'<div style="text-align:center; font-family:DM Mono,monospace; '
                        f'font-size:0.75rem; color:#555570;">'
                        f'{lbl.replace("a meme about ","").title()}<br>'
                        f'<span style="color:#FF6BFF; font-size:0.9rem;">{sc:.0%}</span></div>',
                        unsafe_allow_html=True
                    )

        # ── Timing ────────────────────────────────
        elapsed = result.get("elapsed_seconds", "?")
        st.markdown(
            f'<div style="margin-top:1.5rem; font-family:DM Mono,monospace; font-size:0.75rem; '
            f'color:#333350; text-align:right;">Analyzed in {elapsed}s</div>',
            unsafe_allow_html=True
        )

        # ── Raw JSON ──────────────────────────────
        with st.expander("🔍 Raw JSON output"):
            try:
                clean_result = safe_remove_numpy(result)
                st.code(safe_json_dumps(clean_result), language="json")
            except Exception as e:
                st.warning(f"Could not render JSON: {e}")
                st.write(result)


if __name__ == "__main__":
    main()