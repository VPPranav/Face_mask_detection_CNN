import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
import io
import time

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MaskSense AI",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

/* ── Root Variables ── */
:root {
    --bg-primary: #080c14;
    --bg-secondary: #0d1421;
    --bg-card: #111827;
    --bg-card-hover: #151f2e;
    --accent-cyan: #00d4ff;
    --accent-teal: #00b4a0;
    --accent-green: #00e676;
    --accent-amber: #ffab40;
    --accent-red: #ff5252;
    --text-primary: #e8edf5;
    --text-secondary: #8b9ab0;
    --text-muted: #4a5568;
    --border: rgba(0, 212, 255, 0.15);
    --glow-cyan: 0 0 30px rgba(0, 212, 255, 0.25);
    --glow-green: 0 0 30px rgba(0, 230, 118, 0.25);
    --glow-red: 0 0 30px rgba(255, 82, 82, 0.25);
}

/* ── Global Reset ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--bg-primary);
    color: var(--text-primary);
}

.main { background-color: var(--bg-primary); }
.block-container { padding: 2rem 3rem; max-width: 1400px; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0a0f1a 0%, #0d1421 100%);
    border-right: 1px solid var(--border);
}

[data-testid="stSidebar"] .block-container { padding: 2rem 1.5rem; }

/* ── Hero Banner ── */
.hero-banner {
    background: linear-gradient(135deg, #080c14 0%, #0a1628 40%, #060d1a 100%);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 3.5rem 3rem;
    margin-bottom: 2.5rem;
    position: relative;
    overflow: hidden;
    text-align: center;
}

.hero-banner::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -20%;
    width: 60%;
    height: 200%;
    background: radial-gradient(ellipse, rgba(0, 212, 255, 0.06) 0%, transparent 70%);
    pointer-events: none;
}

.hero-banner::after {
    content: '';
    position: absolute;
    bottom: -50%;
    right: -20%;
    width: 60%;
    height: 200%;
    background: radial-gradient(ellipse, rgba(0, 180, 160, 0.06) 0%, transparent 70%);
    pointer-events: none;
}

.hero-badge {
    display: inline-block;
    background: rgba(0, 212, 255, 0.1);
    border: 1px solid rgba(0, 212, 255, 0.3);
    color: var(--accent-cyan);
    padding: 0.35rem 1rem;
    border-radius: 100px;
    font-size: 0.75rem;
    font-weight: 500;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 1.5rem;
}

.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 3.5rem;
    font-weight: 800;
    background: linear-gradient(135deg, #ffffff 0%, #00d4ff 50%, #00b4a0 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.1;
    margin: 0 0 1rem;
    letter-spacing: -1px;
}

.hero-subtitle {
    font-size: 1.1rem;
    color: var(--text-secondary);
    max-width: 600px;
    margin: 0 auto 2rem;
    line-height: 1.7;
    font-weight: 300;
}

.hero-stats {
    display: flex;
    justify-content: center;
    gap: 3rem;
    flex-wrap: wrap;
    margin-top: 2rem;
    padding-top: 2rem;
    border-top: 1px solid var(--border);
}

.stat-block { text-align: center; }

.stat-number {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 800;
    color: var(--accent-cyan);
    display: block;
    line-height: 1;
}

.stat-label {
    font-size: 0.75rem;
    color: var(--text-muted);
    letter-spacing: 1px;
    text-transform: uppercase;
    margin-top: 0.3rem;
}

/* ── Cards ── */
.glass-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 2rem;
    margin-bottom: 1.5rem;
    transition: border-color 0.3s ease;
    position: relative;
    overflow: hidden;
}

.glass-card:hover { border-color: rgba(0, 212, 255, 0.3); }

.section-label {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: var(--accent-cyan);
    margin-bottom: 0.5rem;
}

.section-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.6rem;
    font-weight: 700;
    color: var(--text-primary);
    margin-bottom: 1rem;
    line-height: 1.2;
}

/* ── Result Display ── */
.result-mask {
    background: linear-gradient(135deg, rgba(0,230,118,0.08), rgba(0,180,160,0.08));
    border: 2px solid rgba(0,230,118,0.4);
    border-radius: 20px;
    padding: 2.5rem;
    text-align: center;
    box-shadow: var(--glow-green);
}

.result-no-mask {
    background: linear-gradient(135deg, rgba(255,82,82,0.08), rgba(255,171,64,0.08));
    border: 2px solid rgba(255,82,82,0.4);
    border-radius: 20px;
    padding: 2.5rem;
    text-align: center;
    box-shadow: var(--glow-red);
}

.result-icon { font-size: 4rem; margin-bottom: 0.5rem; }

.result-label {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 800;
    margin: 0.5rem 0;
}

.result-mask .result-label { color: var(--accent-green); }
.result-no-mask .result-label { color: var(--accent-red); }

.confidence-text {
    font-size: 0.85rem;
    color: var(--text-secondary);
    letter-spacing: 1px;
    text-transform: uppercase;
    margin-bottom: 0.25rem;
}

.confidence-value {
    font-family: 'Syne', sans-serif;
    font-size: 3rem;
    font-weight: 800;
}

.result-mask .confidence-value { color: var(--accent-green); }
.result-no-mask .confidence-value { color: var(--accent-red); }

/* ── Probability Bar ── */
.prob-bar-container { margin-top: 1.5rem; }

.prob-label {
    display: flex;
    justify-content: space-between;
    font-size: 0.8rem;
    color: var(--text-secondary);
    margin-bottom: 0.4rem;
}

.prob-bar-track {
    background: rgba(255,255,255,0.06);
    border-radius: 100px;
    height: 8px;
    overflow: hidden;
}

.prob-bar-fill-mask {
    height: 100%;
    border-radius: 100px;
    background: linear-gradient(90deg, #00b4a0, #00e676);
    transition: width 0.8s cubic-bezier(0.4, 0, 0.2, 1);
}

.prob-bar-fill-nomask {
    height: 100%;
    border-radius: 100px;
    background: linear-gradient(90deg, #ff5252, #ffab40);
    transition: width 0.8s cubic-bezier(0.4, 0, 0.2, 1);
}

/* ── Metric Cards ── */
.metric-row {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1rem;
    margin: 1.5rem 0;
}

.metric-card {
    background: rgba(0,212,255,0.04);
    border: 1px solid rgba(0,212,255,0.12);
    border-radius: 12px;
    padding: 1.2rem;
    text-align: center;
}

.metric-val {
    font-family: 'Syne', sans-serif;
    font-size: 1.6rem;
    font-weight: 800;
    color: var(--accent-cyan);
}

.metric-key {
    font-size: 0.7rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 0.25rem;
}

/* ── Architecture Block ── */
.arch-block {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
    margin: 1rem 0;
}

.arch-layer {
    display: flex;
    align-items: center;
    gap: 1rem;
    background: rgba(0,212,255,0.04);
    border-left: 3px solid var(--accent-cyan);
    border-radius: 0 10px 10px 0;
    padding: 0.75rem 1rem;
}

.arch-layer-name {
    font-family: 'Syne', sans-serif;
    font-size: 0.85rem;
    font-weight: 600;
    color: var(--text-primary);
    min-width: 180px;
}

.arch-layer-detail {
    font-size: 0.8rem;
    color: var(--text-secondary);
}

.arch-arrow {
    text-align: center;
    color: var(--accent-teal);
    font-size: 1.2rem;
    margin: -0.25rem 0;
}

/* ── Timeline Pipeline ── */
.pipeline-step {
    display: flex;
    gap: 1rem;
    align-items: flex-start;
    padding: 1rem 0;
    border-bottom: 1px solid var(--border);
}

.pipeline-step:last-child { border-bottom: none; }

.step-num {
    background: linear-gradient(135deg, var(--accent-cyan), var(--accent-teal));
    color: #000;
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 0.8rem;
    width: 28px;
    height: 28px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
    margin-top: 2px;
}

.step-content { flex: 1; }
.step-title {
    font-weight: 600;
    color: var(--text-primary);
    font-size: 0.95rem;
    margin-bottom: 0.2rem;
}

.step-desc {
    font-size: 0.82rem;
    color: var(--text-secondary);
    line-height: 1.5;
}

/* ── Tag Chips ── */
.tag-row { display: flex; flex-wrap: wrap; gap: 0.5rem; margin-top: 0.5rem; }

.tag {
    background: rgba(0,212,255,0.08);
    border: 1px solid rgba(0,212,255,0.2);
    color: var(--accent-cyan);
    padding: 0.25rem 0.75rem;
    border-radius: 100px;
    font-size: 0.75rem;
    font-weight: 500;
}

/* ── Credit Footer ── */
.credit-footer {
    text-align: center;
    padding: 2rem 1rem;
    color: var(--text-muted);
    font-size: 0.8rem;
    letter-spacing: 0.5px;
    border-top: 1px solid var(--border);
    margin-top: 2rem;
}

.credit-footer strong {
    color: var(--accent-cyan);
    font-weight: 600;
}

/* ── Streamlit overrides ── */
.stButton > button {
    background: linear-gradient(135deg, #00d4ff, #00b4a0) !important;
    color: #000 !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.9rem !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.75rem 2rem !important;
    letter-spacing: 0.5px !important;
    transition: all 0.2s !important;
    box-shadow: 0 4px 20px rgba(0, 212, 255, 0.25) !important;
}

.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px rgba(0, 212, 255, 0.4) !important;
}

div[data-testid="stFileUploader"] {
    border: 2px dashed rgba(0,212,255,0.3) !important;
    border-radius: 16px !important;
    background: rgba(0,212,255,0.02) !important;
    padding: 1.5rem !important;
    transition: border-color 0.3s !important;
}

div[data-testid="stFileUploader"]:hover {
    border-color: rgba(0,212,255,0.6) !important;
}

[data-testid="stMarkdownContainer"] h1,
[data-testid="stMarkdownContainer"] h2,
[data-testid="stMarkdownContainer"] h3 {
    font-family: 'Syne', sans-serif;
}

.stProgress > div > div {
    background: linear-gradient(90deg, #00d4ff, #00b4a0) !important;
}

.stSlider > div > div > div { background: var(--accent-cyan) !important; }

div[data-baseweb="tab-list"] {
    background: var(--bg-card) !important;
    border-radius: 12px !important;
    border: 1px solid var(--border) !important;
    padding: 0.25rem !important;
}

div[data-baseweb="tab"] {
    color: var(--text-secondary) !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
}

div[aria-selected="true"] {
    background: rgba(0,212,255,0.12) !important;
    color: var(--accent-cyan) !important;
    border-radius: 8px !important;
}

.stAlert {
    background: rgba(0,212,255,0.06) !important;
    border: 1px solid rgba(0,212,255,0.2) !important;
    border-radius: 12px !important;
    color: var(--text-primary) !important;
}

hr { border-color: var(--border) !important; }

[data-testid="stSidebarNavItems"] a { color: var(--text-secondary) !important; }
</style>
""", unsafe_allow_html=True)


# ─── Model Loading ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    try:
        model = tf.keras.models.load_model("best_mask_model.keras")
        return model
    except Exception as e:
        return None


def preprocess_image(image: Image.Image, target_size=(128, 128)):
    img = np.array(image.convert("RGB"))
    img = cv2.resize(img, target_size)
    img = img / 255.0
    return np.expand_dims(img, axis=0)


def predict(model, img_array):
    prob = float(model.predict(img_array, verbose=0)[0][0])
    label = "With Mask" if prob > 0.5 else "Without Mask"
    confidence = prob if prob > 0.5 else 1 - prob
    return label, confidence, prob


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; margin-bottom:2rem;'>
        <div style='font-size:2.5rem; margin-bottom:0.5rem;'>🎭</div>
        <div style='font-family: Syne, sans-serif; font-size:1.2rem; font-weight:800;
                    background: linear-gradient(135deg,#00d4ff,#00b4a0);
                    -webkit-background-clip:text; -webkit-text-fill-color:transparent;'>
            MaskSense AI
        </div>
        <div style='font-size:0.72rem; color:#4a5568; letter-spacing:2px; text-transform:uppercase; margin-top:0.25rem;'>
            Face Mask Detection
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div class='section-label'>Navigation</div>
    """, unsafe_allow_html=True)

    nav = st.radio(
        "",
        ["🔬 Detection", "📊 About the Project", "🧠 Model Architecture", "📈 Performance"],
        label_visibility="collapsed",
    )

    st.markdown("---")

    st.markdown("""
    <div class='section-label'>Detection Settings</div>
    """, unsafe_allow_html=True)

    threshold = st.slider(
        "Decision Threshold",
        min_value=0.3,
        max_value=0.9,
        value=0.5,
        step=0.05,
        help="Raise threshold to reduce false 'With Mask' predictions. Default: 0.50",
    )

    st.markdown(f"""
    <div style='background:rgba(0,212,255,0.06); border:1px solid rgba(0,212,255,0.15);
                border-radius:10px; padding:0.8rem; font-size:0.78rem; color:#8b9ab0; margin-top:0.5rem;'>
        <b style='color:#00d4ff;'>Threshold: {threshold:.2f}</b><br>
        {'🔒 Conservative — fewer false positives' if threshold > 0.5 else '⚡ Default — balanced detection' if threshold == 0.5 else '🔓 Lenient — more mask detections'}
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("""
    <div style='font-size:0.72rem; color:#4a5568; text-align:center; line-height:1.8;'>
        <div style='margin-bottom:0.5rem;'>Built with TensorFlow & Keras</div>
        <div>Dataset: Kaggle · 7,553 images</div>
        <div style='margin-top:0.5rem; color:#00d4ff; font-weight:600;'>Test Accuracy: 96.82%</div>
    </div>
    """, unsafe_allow_html=True)


# ─── Hero Header ──────────────────────────────────────────────────────────────
st.markdown("""
<div class='hero-banner'>
    <div class='hero-badge'>🎓 Internship Project · Deep Learning</div>
    <div class='hero-title'>MaskSense AI</div>
    <div class='hero-subtitle'>
        A production-grade Convolutional Neural Network for real-time face mask detection —
        achieving 96.82% accuracy across 7,553 real-world images.
    </div>
    <div class='hero-stats'>
        <div class='stat-block'>
            <span class='stat-number'>96.82%</span>
            <span class='stat-label'>Test Accuracy</span>
        </div>
        <div class='stat-block'>
            <span class='stat-number'>0.9965</span>
            <span class='stat-label'>AUC-ROC Score</span>
        </div>
        <div class='stat-block'>
            <span class='stat-number'>6.55M</span>
            <span class='stat-label'>Parameters</span>
        </div>
        <div class='stat-block'>
            <span class='stat-number'>7,553</span>
            <span class='stat-label'>Training Images</span>
        </div>
        <div class='stat-block'>
            <span class='stat-number'>128²</span>
            <span class='stat-label'>Input Resolution</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ─── DETECTION PAGE ────────────────────────────────────────────────────────────
if "Detection" in nav:
    model = load_model()

    if model is None:
        st.markdown("""
        <div style='background:rgba(255,82,82,0.08); border:1px solid rgba(255,82,82,0.3);
                    border-radius:14px; padding:1.5rem; text-align:center; color:#ff5252;'>
            ⚠️ <b>Model not found.</b> Please ensure <code>best_mask_model.keras</code>
            is in the same directory as <code>app.py</code>.
        </div>
        """, unsafe_allow_html=True)
        st.stop()

    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.markdown("""
        <div class='glass-card'>
            <div class='section-label'>Input</div>
            <div class='section-title'>Upload Image</div>
            <p style='color:#8b9ab0; font-size:0.88rem; margin-bottom:1.5rem;'>
                Upload a clear face photo. The model accepts JPG, JPEG, and PNG formats.
                Ensure the face is well-lit and clearly visible for best results.
            </p>
        """, unsafe_allow_html=True)

        uploaded = st.file_uploader(
            "Drop an image here or click to browse",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed",
        )

        st.markdown("</div>", unsafe_allow_html=True)

        if uploaded:
            image = Image.open(uploaded)
            st.image(image, caption="Uploaded Image", use_container_width=True)

            st.markdown("""
            <div style='margin-top:1rem;'>
            """, unsafe_allow_html=True)
            run_btn = st.button("⚡  Run Detection", use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

    with col_right:
        st.markdown("""
        <div class='glass-card' style='min-height:300px;'>
            <div class='section-label'>Output</div>
            <div class='section-title'>Detection Result</div>
        """, unsafe_allow_html=True)

        if not uploaded:
            st.markdown("""
            <div style='text-align:center; padding:3rem 1rem; color:#4a5568;'>
                <div style='font-size:3rem; margin-bottom:1rem;'>📸</div>
                <div style='font-size:0.9rem;'>Upload an image to see the detection result</div>
            </div>
            """, unsafe_allow_html=True)

        elif uploaded and not run_btn:
            st.markdown("""
            <div style='text-align:center; padding:3rem 1rem; color:#8b9ab0;'>
                <div style='font-size:3rem; margin-bottom:1rem;'>🎯</div>
                <div style='font-size:0.9rem;'>Click <b>Run Detection</b> to analyze your image</div>
            </div>
            """, unsafe_allow_html=True)

        if uploaded and run_btn:
            with st.spinner("Analyzing image…"):
                img_array = preprocess_image(image)
                raw_prob = float(model.predict(img_array, verbose=0)[0][0])

            label = "With Mask" if raw_prob > threshold else "Without Mask"
            confidence = raw_prob if raw_prob > threshold else 1 - raw_prob

            if label == "With Mask":
                result_class = "result-mask"
                result_icon = "✅"
                conf_color = "#00e676"
            else:
                result_class = "result-no-mask"
                result_icon = "❌"
                conf_color = "#ff5252"

            mask_pct = raw_prob * 100
            no_mask_pct = (1 - raw_prob) * 100

            st.markdown(f"""
            <div class='{result_class}'>
                <div class='result-icon'>{result_icon}</div>
                <div class='result-label'>{label}</div>
                <div class='confidence-text'>Confidence</div>
                <div class='confidence-value'>{confidence*100:.1f}%</div>

                <div class='prob-bar-container'>
                    <div class='prob-label'>
                        <span>😷 With Mask</span>
                        <span>{mask_pct:.1f}%</span>
                    </div>
                    <div class='prob-bar-track'>
                        <div class='prob-bar-fill-mask' style='width:{mask_pct}%;'></div>
                    </div>
                </div>

                <div class='prob-bar-container' style='margin-top:0.75rem;'>
                    <div class='prob-label'>
                        <span>🚫 Without Mask</span>
                        <span>{no_mask_pct:.1f}%</span>
                    </div>
                    <div class='prob-bar-track'>
                        <div class='prob-bar-fill-nomask' style='width:{no_mask_pct}%;'></div>
                    </div>
                </div>

                <div style='margin-top:1.5rem; font-size:0.75rem; color:rgba(255,255,255,0.4);
                            border-top:1px solid rgba(255,255,255,0.1); padding-top:1rem;'>
                    Raw sigmoid output: {raw_prob:.6f} · Threshold: {threshold:.2f}
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    # ── Tips ──
    st.markdown("""
    <div class='glass-card'>
        <div class='section-label'>Tips for Best Results</div>
        <div style='display:grid; grid-template-columns:repeat(3,1fr); gap:1rem; margin-top:1rem;'>
            <div style='padding:1rem; background:rgba(0,212,255,0.04); border-radius:10px; border:1px solid rgba(0,212,255,0.1);'>
                <div style='font-size:1.5rem; margin-bottom:0.5rem;'>💡</div>
                <div style='font-size:0.85rem; font-weight:600; color:#e8edf5; margin-bottom:0.3rem;'>Good Lighting</div>
                <div style='font-size:0.78rem; color:#8b9ab0;'>Use well-lit photos where the face is clearly visible without harsh shadows.</div>
            </div>
            <div style='padding:1rem; background:rgba(0,212,255,0.04); border-radius:10px; border:1px solid rgba(0,212,255,0.1);'>
                <div style='font-size:1.5rem; margin-bottom:0.5rem;'>🖼️</div>
                <div style='font-size:0.85rem; font-weight:600; color:#e8edf5; margin-bottom:0.3rem;'>Face Centered</div>
                <div style='font-size:0.78rem; color:#8b9ab0;'>Keep the face centered and occupying most of the frame for accurate detection.</div>
            </div>
            <div style='padding:1rem; background:rgba(0,212,255,0.04); border-radius:10px; border:1px solid rgba(0,212,255,0.1);'>
                <div style='font-size:1.5rem; margin-bottom:0.5rem;'>📐</div>
                <div style='font-size:0.85rem; font-weight:600; color:#e8edf5; margin-bottom:0.3rem;'>Front-Facing</div>
                <div style='font-size:0.78rem; color:#8b9ab0;'>Front-facing or slight angle images work best. Extreme side profiles may reduce accuracy.</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ─── ABOUT THE PROJECT PAGE ────────────────────────────────────────────────────
elif "About" in nav:
    st.markdown("""
    <div class='glass-card'>
        <div class='section-label'>Project Overview</div>
        <div class='section-title'>Face Mask Detection using CNN</div>
        <p style='color:#8b9ab0; line-height:1.8; font-size:0.92rem;'>
            This project implements a complete, end-to-end deep learning pipeline for detecting whether
            a person in an image is wearing a face mask. Built from scratch using a custom Convolutional
            Neural Network (CNN) with TensorFlow and Keras, the system addresses a binary image
            classification problem — one of the most practically impactful applications of computer
            vision during and after the COVID-19 pandemic.
        </p>
        <div class='tag-row'>
            <span class='tag'>Binary Classification</span>
            <span class='tag'>Computer Vision</span>
            <span class='tag'>TensorFlow 2.19</span>
            <span class='tag'>Keras 3.13</span>
            <span class='tag'>OpenCV</span>
            <span class='tag'>CNN</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown("""
        <div class='glass-card'>
            <div class='section-label'>Dataset</div>
            <div class='section-title'>Kaggle Face Mask Dataset</div>
            <p style='color:#8b9ab0; font-size:0.88rem; line-height:1.7; margin-bottom:1rem;'>
                Sourced via <code>kagglehub</code> from the publicly available Kaggle dataset
                by <b>omkargurav</b>. The dataset is notably well-balanced, eliminating the need
                for class weighting or oversampling techniques.
            </p>
            <div style='display:grid; grid-template-columns:1fr 1fr 1fr; gap:0.75rem;'>
                <div style='text-align:center; padding:1rem; background:rgba(0,230,118,0.06);
                            border:1px solid rgba(0,230,118,0.2); border-radius:10px;'>
                    <div style='font-family:Syne,sans-serif; font-size:1.4rem; font-weight:800; color:#00e676;'>3,725</div>
                    <div style='font-size:0.72rem; color:#8b9ab0; text-transform:uppercase; letter-spacing:1px;'>With Mask</div>
                </div>
                <div style='text-align:center; padding:1rem; background:rgba(255,82,82,0.06);
                            border:1px solid rgba(255,82,82,0.2); border-radius:10px;'>
                    <div style='font-family:Syne,sans-serif; font-size:1.4rem; font-weight:800; color:#ff5252;'>3,828</div>
                    <div style='font-size:0.72rem; color:#8b9ab0; text-transform:uppercase; letter-spacing:1px;'>Without Mask</div>
                </div>
                <div style='text-align:center; padding:1rem; background:rgba(0,212,255,0.06);
                            border:1px solid rgba(0,212,255,0.2); border-radius:10px;'>
                    <div style='font-family:Syne,sans-serif; font-size:1.4rem; font-weight:800; color:#00d4ff;'>7,553</div>
                    <div style='font-size:0.72rem; color:#8b9ab0; text-transform:uppercase; letter-spacing:1px;'>Total Images</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class='glass-card'>
            <div class='section-label'>Data Preprocessing</div>
            <div class='section-title'>Image Pipeline</div>
            <div class='arch-block'>
                <div class='arch-layer'>
                    <div class='arch-layer-name'>📥 Load with OpenCV</div>
                    <div class='arch-layer-detail'>cv2.imread() — raw image bytes</div>
                </div>
                <div class='arch-layer'>
                    <div class='arch-layer-name'>🎨 BGR → RGB</div>
                    <div class='arch-layer-detail'>cv2.cvtColor() — correct channel order</div>
                </div>
                <div class='arch-layer'>
                    <div class='arch-layer-name'>📐 Resize</div>
                    <div class='arch-layer-detail'>128 × 128 pixels — fixed input shape</div>
                </div>
                <div class='arch-layer'>
                    <div class='arch-layer-name'>📊 Normalize</div>
                    <div class='arch-layer-detail'>÷ 255.0 → values in [0.0, 1.0]</div>
                </div>
                <div class='arch-layer'>
                    <div class='arch-layer-name'>🏷️ Label Encode</div>
                    <div class='arch-layer-detail'>With Mask → 1, Without Mask → 0</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class='glass-card'>
        <div class='section-label'>Full Pipeline</div>
        <div class='section-title'>End-to-End ML Workflow</div>
        <div class='pipeline-step'>
            <div class='step-num'>1</div>
            <div class='step-content'>
                <div class='step-title'>Data Exploration & Visualization</div>
                <div class='step-desc'>Loaded 7,553 images from Kaggle. Visualized sample images from both classes,
                confirmed class balance (3,725 with mask / 3,828 without), and inspected raw pixel distributions.</div>
            </div>
        </div>
        <div class='pipeline-step'>
            <div class='step-num'>2</div>
            <div class='step-content'>
                <div class='step-title'>Preprocessing</div>
                <div class='step-desc'>Applied BGR→RGB conversion, resized all images to 128×128, normalized pixel
                values to [0,1], and encoded labels numerically. Corrupted images were caught and skipped.</div>
            </div>
        </div>
        <div class='pipeline-step'>
            <div class='step-num'>3</div>
            <div class='step-content'>
                <div class='step-title'>Stratified Train / Val / Test Split</div>
                <div class='step-desc'>Used two-stage stratified splitting: 70% train (5,290 images),
                15% validation (1,130 images), 15% test (1,133 images). Stratification preserved class ratios in every split.</div>
            </div>
        </div>
        <div class='pipeline-step'>
            <div class='step-num'>4</div>
            <div class='step-content'>
                <div class='step-title'>Data Augmentation</div>
                <div class='step-desc'>Applied exclusively to training images via Keras ImageDataGenerator:
                ±20° rotation, 20% width/height shifts, shear/zoom (0.2), and horizontal flipping.
                Augmented data was streamed on-the-fly — no disk writes.</div>
            </div>
        </div>
        <div class='pipeline-step'>
            <div class='step-num'>5</div>
            <div class='step-content'>
                <div class='step-title'>Custom CNN Training</div>
                <div class='step-desc'>Trained a 3-block CNN (Adam optimizer, Binary Crossentropy loss,
                batch size 32, max 25 epochs) with three automated callbacks: EarlyStopping, ReduceLROnPlateau,
                and ModelCheckpoint saving the best validation-accuracy weights.</div>
            </div>
        </div>
        <div class='pipeline-step'>
            <div class='step-num'>6</div>
            <div class='step-content'>
                <div class='step-title'>Evaluation & Analysis</div>
                <div class='step-desc'>Comprehensive evaluation on the held-out test set:
                classification report (precision/recall/F1), confusion matrix, ROC curve,
                prediction probability distribution, and misclassification grid of all 36 errors.</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class='glass-card'>
        <div class='section-label'>Key Insights</div>
        <div class='section-title'>What the Research Revealed</div>
        <div style='display:grid; grid-template-columns:1fr 1fr; gap:1rem; margin-top:1rem;'>
            <div style='padding:1.2rem; background:rgba(0,212,255,0.04); border-radius:12px; border-left:3px solid #00d4ff;'>
                <div style='font-size:0.85rem; font-weight:600; color:#e8edf5; margin-bottom:0.4rem;'>No Overfitting</div>
                <div style='font-size:0.8rem; color:#8b9ab0;'>Training accuracy (95.94%) and test accuracy (96.82%) are nearly identical, demonstrating strong generalization.</div>
            </div>
            <div style='padding:1.2rem; background:rgba(0,212,255,0.04); border-radius:12px; border-left:3px solid #00b4a0;'>
                <div style='font-size:0.85rem; font-weight:600; color:#e8edf5; margin-bottom:0.4rem;'>LR Reduction Was Key</div>
                <div style='font-size:0.8rem; color:#8b9ab0;'>ReduceLROnPlateau drove accuracy from ~89% to ~97% after the first reduction at Epoch 10 — the single biggest performance jump.</div>
            </div>
            <div style='padding:1.2rem; background:rgba(0,212,255,0.04); border-radius:12px; border-left:3px solid #00e676;'>
                <div style='font-size:0.85rem; font-weight:600; color:#e8edf5; margin-bottom:0.4rem;'>High-Confidence Predictions</div>
                <div style='font-size:0.8rem; color:#8b9ab0;'>The sigmoid output distribution is strongly bimodal — most predictions cluster near 0.0 or 1.0, not near the 0.5 boundary.</div>
            </div>
            <div style='padding:1.2rem; background:rgba(0,212,255,0.04); border-radius:12px; border-left:3px solid #ffab40;'>
                <div style='font-size:0.85rem; font-weight:600; color:#e8edf5; margin-bottom:0.4rem;'>Zero Class Bias</div>
                <div style='font-size:0.8rem; color:#8b9ab0;'>Both classes achieved identical F1-scores of 0.97, confirming the model has no preference towards either class.</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ─── ARCHITECTURE PAGE ─────────────────────────────────────────────────────────
elif "Architecture" in nav:
    st.markdown("""
    <div class='glass-card'>
        <div class='section-label'>Neural Network</div>
        <div class='section-title'>Custom CNN Architecture</div>
        <p style='color:#8b9ab0; font-size:0.88rem; line-height:1.7; margin-bottom:1.5rem;'>
            A Sequential CNN with 3 convolutional blocks of increasing filter depth (32→64→128),
            followed by two fully connected dense layers. Every convolutional block includes
            BatchNormalization for training stability and Dropout for regularization.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([3, 2], gap="large")

    with col1:
        layers = [
            ("📥 Input", "128 × 128 × 3 RGB image", "#00d4ff"),
            ("🔲 Conv2D (32 filters, 3×3, ReLU)", "Block 1 — edge & texture detection", "#00b4a0"),
            ("📊 BatchNormalization", "Normalize activations, stabilize training", "#00b4a0"),
            ("⬇️ MaxPooling2D (2×2)", "Output: 63 × 63 × 32", "#00b4a0"),
            ("🎲 Dropout (0.25)", "Light regularization in conv block", "#00b4a0"),
            ("🔲 Conv2D (64 filters, 3×3, ReLU)", "Block 2 — mid-level features", "#ffab40"),
            ("📊 BatchNormalization", "Normalize activations", "#ffab40"),
            ("⬇️ MaxPooling2D (2×2)", "Output: 30 × 30 × 64", "#ffab40"),
            ("🎲 Dropout (0.25)", "Light regularization in conv block", "#ffab40"),
            ("🔲 Conv2D (128 filters, 3×3, ReLU)", "Block 3 — complex features", "#ff5252"),
            ("📊 BatchNormalization", "Normalize activations", "#ff5252"),
            ("⬇️ MaxPooling2D (2×2)", "Output: 14 × 14 × 128", "#ff5252"),
            ("🎲 Dropout (0.25)", "Light regularization in conv block", "#ff5252"),
            ("↔️ Flatten", "25,088 units — reshape for dense layers", "#8b9ab0"),
            ("🔵 Dense (256, ReLU)", "First fully connected layer", "#a78bfa"),
            ("📊 BatchNormalization", "Normalize dense activations", "#a78bfa"),
            ("🎲 Dropout (0.50)", "Strong regularization in dense layer", "#a78bfa"),
            ("🔵 Dense (128, ReLU)", "Second fully connected layer", "#c084fc"),
            ("🎲 Dropout (0.50)", "Strong regularization in dense layer", "#c084fc"),
            ("📤 Dense (1, Sigmoid)", "Output: probability ∈ [0, 1]", "#00e676"),
        ]

        for name, detail, color in layers:
            st.markdown(f"""
            <div style='display:flex; align-items:center; gap:1rem; padding:0.6rem 1rem;
                        background:rgba(0,212,255,0.03); border-left:3px solid {color};
                        border-radius:0 10px 10px 0; margin-bottom:0.4rem;'>
                <div style='font-size:0.85rem; font-weight:600; color:#e8edf5; min-width:260px;'>{name}</div>
                <div style='font-size:0.78rem; color:#8b9ab0;'>{detail}</div>
            </div>
            """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class='glass-card'>
            <div class='section-label'>Parameters</div>
            <div class='section-title'>Model Size</div>
            <div style='display:flex; flex-direction:column; gap:0.75rem; margin-top:1rem;'>
                <div style='display:flex; justify-content:space-between; align-items:center;
                            padding:0.75rem; background:rgba(0,212,255,0.06); border-radius:8px;'>
                    <span style='font-size:0.82rem; color:#8b9ab0;'>Total Parameters</span>
                    <span style='font-family:Syne,sans-serif; font-weight:700; color:#00d4ff;'>6.55M</span>
                </div>
                <div style='display:flex; justify-content:space-between; align-items:center;
                            padding:0.75rem; background:rgba(0,212,255,0.04); border-radius:8px;'>
                    <span style='font-size:0.82rem; color:#8b9ab0;'>Trainable</span>
                    <span style='font-family:Syne,sans-serif; font-weight:700; color:#00b4a0;'>6,550,017</span>
                </div>
                <div style='display:flex; justify-content:space-between; align-items:center;
                            padding:0.75rem; background:rgba(0,212,255,0.04); border-radius:8px;'>
                    <span style='font-size:0.82rem; color:#8b9ab0;'>Non-trainable</span>
                    <span style='font-family:Syne,sans-serif; font-weight:700; color:#8b9ab0;'>960</span>
                </div>
                <div style='display:flex; justify-content:space-between; align-items:center;
                            padding:0.75rem; background:rgba(0,212,255,0.04); border-radius:8px;'>
                    <span style='font-size:0.82rem; color:#8b9ab0;'>Model Size</span>
                    <span style='font-family:Syne,sans-serif; font-weight:700; color:#ffab40;'>~25 MB</span>
                </div>
            </div>
        </div>

        <div class='glass-card' style='margin-top:1rem;'>
            <div class='section-label'>Training Config</div>
            <div class='section-title'>Hyperparameters</div>
            <div style='display:flex; flex-direction:column; gap:0.6rem; margin-top:1rem; font-size:0.82rem;'>
                <div style='display:flex; justify-content:space-between;'>
                    <span style='color:#8b9ab0;'>Loss</span>
                    <span style='color:#e8edf5;'>Binary Crossentropy</span>
                </div>
                <div style='display:flex; justify-content:space-between;'>
                    <span style='color:#8b9ab0;'>Optimizer</span>
                    <span style='color:#e8edf5;'>Adam</span>
                </div>
                <div style='display:flex; justify-content:space-between;'>
                    <span style='color:#8b9ab0;'>Initial LR</span>
                    <span style='color:#e8edf5;'>0.001</span>
                </div>
                <div style='display:flex; justify-content:space-between;'>
                    <span style='color:#8b9ab0;'>Batch Size</span>
                    <span style='color:#e8edf5;'>32</span>
                </div>
                <div style='display:flex; justify-content:space-between;'>
                    <span style='color:#8b9ab0;'>Max Epochs</span>
                    <span style='color:#e8edf5;'>25</span>
                </div>
            </div>
        </div>

        <div class='glass-card' style='margin-top:1rem;'>
            <div class='section-label'>Callbacks</div>
            <div class='section-title'>Smart Training</div>
            <div style='display:flex; flex-direction:column; gap:0.75rem; margin-top:1rem;'>
                <div style='padding:0.75rem; background:rgba(0,230,118,0.06); border-radius:8px; border-left:3px solid #00e676;'>
                    <div style='font-size:0.82rem; font-weight:600; color:#e8edf5;'>EarlyStopping</div>
                    <div style='font-size:0.75rem; color:#8b9ab0; margin-top:0.2rem;'>patience=5, monitor=val_loss, restore_best_weights</div>
                </div>
                <div style='padding:0.75rem; background:rgba(255,171,64,0.06); border-radius:8px; border-left:3px solid #ffab40;'>
                    <div style='font-size:0.82rem; font-weight:600; color:#e8edf5;'>ReduceLROnPlateau</div>
                    <div style='font-size:0.75rem; color:#8b9ab0; margin-top:0.2rem;'>factor=0.2, patience=3, min_lr=1e-6</div>
                </div>
                <div style='padding:0.75rem; background:rgba(0,212,255,0.06); border-radius:8px; border-left:3px solid #00d4ff;'>
                    <div style='font-size:0.82rem; font-weight:600; color:#e8edf5;'>ModelCheckpoint</div>
                    <div style='font-size:0.75rem; color:#8b9ab0; margin-top:0.2rem;'>monitor=val_accuracy, save_best_only=True</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ─── PERFORMANCE PAGE ──────────────────────────────────────────────────────────
elif "Performance" in nav:
    st.markdown("""
    <div class='glass-card'>
        <div class='section-label'>Results</div>
        <div class='section-title'>Model Performance Summary</div>
        <p style='color:#8b9ab0; font-size:0.88rem; line-height:1.7;'>
            The model was evaluated on a held-out test set of <b>1,133 images</b> —
            data the model never encountered during training or validation.
        </p>
        <div class='metric-row'>
            <div class='metric-card'>
                <div class='metric-val'>96.82%</div>
                <div class='metric-key'>Test Accuracy</div>
            </div>
            <div class='metric-card'>
                <div class='metric-val'>97.00%</div>
                <div class='metric-key'>Precision</div>
            </div>
            <div class='metric-card'>
                <div class='metric-val'>97.00%</div>
                <div class='metric-key'>Recall</div>
            </div>
            <div class='metric-card'>
                <div class='metric-val'>0.9965</div>
                <div class='metric-key'>AUC-ROC</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown("""
        <div class='glass-card'>
            <div class='section-label'>Confusion Matrix</div>
            <div class='section-title'>Prediction Breakdown</div>
            <div style='display:grid; grid-template-columns:1fr 1fr; gap:0.75rem; margin-top:1.5rem;'>
                <div style='text-align:center; padding:1.5rem; background:rgba(0,230,118,0.08);
                            border:2px solid rgba(0,230,118,0.3); border-radius:14px;'>
                    <div style='font-family:Syne,sans-serif; font-size:2.2rem; font-weight:800; color:#00e676;'>551</div>
                    <div style='font-size:0.75rem; color:#8b9ab0; margin-top:0.4rem; text-transform:uppercase; letter-spacing:1px;'>True Negatives</div>
                    <div style='font-size:0.72rem; color:#4a5568; margin-top:0.2rem;'>No Mask → No Mask ✓</div>
                </div>
                <div style='text-align:center; padding:1.5rem; background:rgba(255,82,82,0.08);
                            border:2px solid rgba(255,82,82,0.3); border-radius:14px;'>
                    <div style='font-family:Syne,sans-serif; font-size:2.2rem; font-weight:800; color:#ff5252;'>23</div>
                    <div style='font-size:0.75rem; color:#8b9ab0; margin-top:0.4rem; text-transform:uppercase; letter-spacing:1px;'>False Positives</div>
                    <div style='font-size:0.72rem; color:#4a5568; margin-top:0.2rem;'>No Mask → Mask ✗</div>
                </div>
                <div style='text-align:center; padding:1.5rem; background:rgba(255,82,82,0.08);
                            border:2px solid rgba(255,82,82,0.3); border-radius:14px;'>
                    <div style='font-family:Syne,sans-serif; font-size:2.2rem; font-weight:800; color:#ff5252;'>13</div>
                    <div style='font-size:0.75rem; color:#8b9ab0; margin-top:0.4rem; text-transform:uppercase; letter-spacing:1px;'>False Negatives</div>
                    <div style='font-size:0.72rem; color:#4a5568; margin-top:0.2rem;'>Mask → No Mask ✗</div>
                </div>
                <div style='text-align:center; padding:1.5rem; background:rgba(0,230,118,0.08);
                            border:2px solid rgba(0,230,118,0.3); border-radius:14px;'>
                    <div style='font-family:Syne,sans-serif; font-size:2.2rem; font-weight:800; color:#00e676;'>546</div>
                    <div style='font-size:0.75rem; color:#8b9ab0; margin-top:0.4rem; text-transform:uppercase; letter-spacing:1px;'>True Positives</div>
                    <div style='font-size:0.72rem; color:#4a5568; margin-top:0.2rem;'>Mask → Mask ✓</div>
                </div>
            </div>
            <div style='margin-top:1rem; padding:0.75rem; background:rgba(0,212,255,0.06);
                        border-radius:8px; text-align:center; font-size:0.82rem; color:#8b9ab0;'>
                Total Misclassifications: <b style='color:#ffab40;'>36 / 1,133</b> (3.18% error rate)
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class='glass-card'>
            <div class='section-label'>Per-Class Metrics</div>
            <div class='section-title'>Class-wise Report</div>
            <div style='margin-top:1.5rem;'>
                <div style='padding:1rem; background:rgba(0,230,118,0.06); border-radius:12px;
                            border:1px solid rgba(0,230,118,0.2); margin-bottom:0.75rem;'>
                    <div style='font-weight:700; color:#00e676; margin-bottom:0.75rem; font-size:0.9rem;'>😷 With Mask</div>
                    <div style='display:grid; grid-template-columns:repeat(3,1fr); gap:0.5rem;'>
                        <div style='text-align:center;'>
                            <div style='font-family:Syne,sans-serif; font-size:1.3rem; font-weight:700; color:#e8edf5;'>0.96</div>
                            <div style='font-size:0.7rem; color:#8b9ab0;'>Precision</div>
                        </div>
                        <div style='text-align:center;'>
                            <div style='font-family:Syne,sans-serif; font-size:1.3rem; font-weight:700; color:#e8edf5;'>0.98</div>
                            <div style='font-size:0.7rem; color:#8b9ab0;'>Recall</div>
                        </div>
                        <div style='text-align:center;'>
                            <div style='font-family:Syne,sans-serif; font-size:1.3rem; font-weight:700; color:#e8edf5;'>0.97</div>
                            <div style='font-size:0.7rem; color:#8b9ab0;'>F1-Score</div>
                        </div>
                    </div>
                </div>
                <div style='padding:1rem; background:rgba(255,82,82,0.06); border-radius:12px;
                            border:1px solid rgba(255,82,82,0.2); margin-bottom:0.75rem;'>
                    <div style='font-weight:700; color:#ff5252; margin-bottom:0.75rem; font-size:0.9rem;'>🚫 Without Mask</div>
                    <div style='display:grid; grid-template-columns:repeat(3,1fr); gap:0.5rem;'>
                        <div style='text-align:center;'>
                            <div style='font-family:Syne,sans-serif; font-size:1.3rem; font-weight:700; color:#e8edf5;'>0.98</div>
                            <div style='font-size:0.7rem; color:#8b9ab0;'>Precision</div>
                        </div>
                        <div style='text-align:center;'>
                            <div style='font-family:Syne,sans-serif; font-size:1.3rem; font-weight:700; color:#e8edf5;'>0.96</div>
                            <div style='font-size:0.7rem; color:#8b9ab0;'>Recall</div>
                        </div>
                        <div style='text-align:center;'>
                            <div style='font-family:Syne,sans-serif; font-size:1.3rem; font-weight:700; color:#e8edf5;'>0.97</div>
                            <div style='font-size:0.7rem; color:#8b9ab0;'>F1-Score</div>
                        </div>
                    </div>
                </div>
                <div style='padding:1rem; background:rgba(0,212,255,0.06); border-radius:12px;
                            border:1px solid rgba(0,212,255,0.2);'>
                    <div style='font-weight:700; color:#00d4ff; margin-bottom:0.75rem; font-size:0.9rem;'>⚖️ Weighted Average</div>
                    <div style='display:grid; grid-template-columns:repeat(3,1fr); gap:0.5rem;'>
                        <div style='text-align:center;'>
                            <div style='font-family:Syne,sans-serif; font-size:1.3rem; font-weight:700; color:#00d4ff;'>0.97</div>
                            <div style='font-size:0.7rem; color:#8b9ab0;'>Precision</div>
                        </div>
                        <div style='text-align:center;'>
                            <div style='font-family:Syne,sans-serif; font-size:1.3rem; font-weight:700; color:#00d4ff;'>0.97</div>
                            <div style='font-size:0.7rem; color:#8b9ab0;'>Recall</div>
                        </div>
                        <div style='text-align:center;'>
                            <div style='font-family:Syne,sans-serif; font-size:1.3rem; font-weight:700; color:#00d4ff;'>0.97</div>
                            <div style='font-size:0.7rem; color:#8b9ab0;'>F1-Score</div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class='glass-card'>
        <div class='section-label'>Learning Rate Schedule</div>
        <div class='section-title'>Auto-adjusted by ReduceLROnPlateau</div>
        <div style='display:grid; grid-template-columns:repeat(3,1fr); gap:1rem; margin-top:1.5rem;'>
            <div style='padding:1rem; background:rgba(0,212,255,0.06); border-radius:10px; text-align:center;
                        border:1px solid rgba(0,212,255,0.2);'>
                <div style='font-size:0.72rem; color:#8b9ab0; text-transform:uppercase; letter-spacing:1px; margin-bottom:0.5rem;'>Epoch 1–10</div>
                <div style='font-family:Syne,sans-serif; font-size:1.3rem; font-weight:800; color:#00d4ff;'>0.001000</div>
                <div style='font-size:0.72rem; color:#4a5568; margin-top:0.3rem;'>Initial learning rate</div>
            </div>
            <div style='padding:1rem; background:rgba(255,171,64,0.06); border-radius:10px; text-align:center;
                        border:1px solid rgba(255,171,64,0.2);'>
                <div style='font-size:0.72rem; color:#8b9ab0; text-transform:uppercase; letter-spacing:1px; margin-bottom:0.5rem;'>Epoch 11–20</div>
                <div style='font-family:Syne,sans-serif; font-size:1.3rem; font-weight:800; color:#ffab40;'>0.000200</div>
                <div style='font-size:0.72rem; color:#4a5568; margin-top:0.3rem;'>↓ Reduced by factor 0.2</div>
            </div>
            <div style='padding:1rem; background:rgba(0,230,118,0.06); border-radius:10px; text-align:center;
                        border:1px solid rgba(0,230,118,0.2);'>
                <div style='font-size:0.72rem; color:#8b9ab0; text-transform:uppercase; letter-spacing:1px; margin-bottom:0.5rem;'>Epoch 21–25</div>
                <div style='font-family:Syne,sans-serif; font-size:1.3rem; font-weight:800; color:#00e676;'>0.000040</div>
                <div style='font-size:0.72rem; color:#4a5568; margin-top:0.3rem;'>↓ Final fine-tuning LR</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ─── Credit Footer ─────────────────────────────────────────────────────────────
st.markdown("""
<div class='credit-footer'>
    <div style='font-family: Syne, sans-serif; font-size:1rem; font-weight:700;
                background:linear-gradient(135deg,#00d4ff,#00b4a0);
                -webkit-background-clip:text; -webkit-text-fill-color:transparent; margin-bottom:0.4rem;'>
        MaskSense AI · Face Mask Detection
    </div>
    Developed by <strong>Pranav V P</strong> as an Internship Project &nbsp;·&nbsp;
    Custom CNN · TensorFlow 2.19 · Keras 3.13 &nbsp;·&nbsp;
    Test Accuracy: 96.82% · AUC-ROC: 0.9965
</div>
""", unsafe_allow_html=True)