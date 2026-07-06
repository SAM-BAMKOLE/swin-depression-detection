import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import os
import plotly.graph_objects as go
from PIL import Image
import timm

st.set_page_config(
    page_title="Depression Detection · Swin Transformer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.block-container { max-width: 1080px; padding-top: 0 !important; }

/* ── theme-aware tokens ── */
:root {
    --bg:       #ffffff;
    --bg2:      #f8fafc;
    --border:   #e2e8f0;
    --text:     #0f172a;
    --muted:    #64748b;
    --accent:   #2563eb;
    --card-bg:  #f1f5f9;
    --mono:     'JetBrains Mono', monospace;
}
@media (prefers-color-scheme: dark) {
    :root {
        --bg:      #0b0e17;
        --bg2:     #111827;
        --border:  #1e293b;
        --text:    #f1f5f9;
        --muted:   #64748b;
        --accent:  #3b82f6;
        --card-bg: #161d2e;
    }
}
[data-theme="dark"] {
    --bg:      #0b0e17;
    --bg2:     #111827;
    --border:  #1e293b;
    --text:    #f1f5f9;
    --muted:   #64748b;
    --accent:  #3b82f6;
    --card-bg: #161d2e;
}
[data-theme="light"] {
    --bg:      #ffffff;
    --bg2:     #f8fafc;
    --border:  #e2e8f0;
    --text:    #0f172a;
    --muted:   #64748b;
    --accent:  #2563eb;
    --card-bg: #f1f5f9;
}

.stApp { background: var(--bg) !important; color: var(--text) !important; }
.stApp > div { background: var(--bg) !important; }

/* ── hero ── */
.hero {
    background: var(--bg2);
    border-bottom: 1px solid var(--border);
    padding: 4rem 2rem 3rem;
    text-align: center;
    margin-bottom: 3rem;
}
.hero-eyebrow {
    font-size: 0.72rem; font-weight: 600; letter-spacing: 0.18em;
    text-transform: uppercase; color: var(--accent); margin-bottom: 1rem;
}
.hero h1 {
    font-size: 2.6rem; font-weight: 700; letter-spacing: -0.03em;
    color: var(--text); margin: 0 0 1rem; line-height: 1.2;
}
.hero h1 span { color: var(--accent); }
.hero-sub {
    font-size: 1rem; color: var(--muted);
    max-width: 680px; margin: 0 auto 2rem;
    line-height: 1.7; font-weight: 300;
    white-space: normal; word-wrap: break-word;
}
.hero-pills { display: flex; gap: 0.5rem; justify-content: center; flex-wrap: wrap; }
.pill {
    background: var(--card-bg); border: 1px solid var(--border);
    border-radius: 999px; padding: 0.3rem 0.9rem;
    font-size: 0.73rem; color: var(--muted); font-weight: 500;
}

/* ── section ── */
.section-head {
    font-size: 1.35rem; font-weight: 600; color: var(--text);
    letter-spacing: -0.02em; margin: 0 0 0.4rem;
}
.section-sub {
    font-size: 0.88rem; color: var(--muted);
    margin: 0 0 1.5rem; font-weight: 400; line-height: 1.7;
}
hr.divider {
    border: none; border-top: 1px solid var(--border); margin: 3rem 0;
}

/* ── stat cards ── */
.stat-grid { display: flex; gap: 0.8rem; flex-wrap: wrap; margin-bottom: 1.5rem; }
.stat-card {
    flex: 1; min-width: 120px; background: var(--card-bg);
    border: 1px solid var(--border); border-radius: 10px;
    padding: 1.1rem; text-align: center;
}
.stat-val {
    font-family: var(--mono); font-size: 1.5rem;
    font-weight: 500; color: var(--accent);
}
.stat-label {
    font-size: 0.68rem; text-transform: uppercase;
    letter-spacing: 0.1em; color: var(--muted); margin-top: 0.3rem;
}

/* ── pipeline ── */
.pipeline { display: flex; gap: 0; margin: 1.5rem 0 2rem; }
.step {
    flex: 1; background: var(--card-bg); border: 1px solid var(--border);
    padding: 1rem 0.9rem; position: relative;
}
.step:first-child { border-radius: 8px 0 0 8px; }
.step:last-child  { border-radius: 0 8px 8px 0; }
.step-num { font-size: 0.62rem; font-weight: 600; letter-spacing: 0.1em;
    text-transform: uppercase; color: var(--accent); margin-bottom: 0.35rem; }
.step-title { font-size: 0.82rem; font-weight: 600; color: var(--text); margin-bottom: 0.2rem; }
.step-desc { font-size: 0.72rem; color: var(--muted); line-height: 1.5; }
.step-arrow { position: absolute; right: -9px; top: 50%; transform: translateY(-50%);
    color: var(--border); font-size: 1.1rem; z-index: 10; }

/* ── result cards ── */
.result-card { border-radius: 12px; padding: 1.8rem; text-align: center; margin-bottom: 0.8rem; }
.result-depressed { background: #fef2f2; border: 1px solid #fecaca; }
.result-normal    { background: #f0fdf4; border: 1px solid #bbf7d0; }
@media (prefers-color-scheme: dark) {
    .result-depressed { background: #1f0a0a; border-color: #7f1d1d; }
    .result-normal    { background: #071a0f; border-color: #14532d; }
}
[data-theme="dark"] .result-depressed { background: #1f0a0a !important; border-color: #7f1d1d !important; }
[data-theme="dark"] .result-normal    { background: #071a0f !important; border-color: #14532d !important; }
.result-label { font-size: 0.68rem; font-weight: 600; letter-spacing: 0.14em;
    text-transform: uppercase; color: var(--muted); margin-bottom: 0.4rem; }
.result-verdict { font-size: 1.7rem; font-weight: 700; letter-spacing: -0.02em; margin: 0.2rem 0 0.5rem; }
.result-prob { font-family: var(--mono); font-size: 0.88rem; color: var(--muted); }

/* ── upload ── */
.upload-label { font-size: 0.72rem; font-weight: 600; text-transform: uppercase;
    letter-spacing: 0.1em; color: var(--muted); margin-bottom: 0.5rem; }
.stButton > button {
    width: 100%; background: var(--accent); color: white; border: none;
    border-radius: 8px; padding: 0.7rem 1rem; font-size: 0.9rem;
    font-weight: 500; font-family: 'Inter', sans-serif; cursor: pointer;
}
.stButton > button:hover { opacity: 0.9; }

/* ── metric row ── */
.metric-row { display: flex; gap: 0.6rem; margin: 1rem 0; }
.metric-box {
    flex: 1; background: var(--card-bg); border: 1px solid var(--border);
    border-radius: 8px; padding: 0.8rem; text-align: center;
}
.metric-val { font-family: var(--mono); font-size: 1.1rem; color: var(--text); }
.metric-name { font-size: 0.65rem; text-transform: uppercase;
    letter-spacing: 0.08em; color: var(--muted); margin-top: 0.2rem; }

/* ── banners ── */
.info-banner {
    background: var(--card-bg); border: 1px solid var(--border);
    border-left: 3px solid var(--accent); border-radius: 6px;
    padding: 0.85rem 1.1rem; font-size: 0.83rem;
    color: var(--muted); margin-bottom: 1.5rem; line-height: 1.65;
}
.warn-banner {
    background: var(--card-bg); border: 1px solid var(--border);
    border-left: 3px solid #f59e0b; border-radius: 6px;
    padding: 0.85rem 1.1rem; font-size: 0.83rem;
    color: var(--muted); margin-bottom: 1.5rem; line-height: 1.65;
}

/* ── dataset info rows ── */
.info-row {
    display: flex; justify-content: space-between; align-items: center;
    background: var(--card-bg); border: 1px solid var(--border);
    border-radius: 8px; padding: 0.75rem 1rem; margin-bottom: 0.5rem;
}
.info-key { color: var(--muted); font-size: 0.82rem; }
.info-val { font-family: var(--mono); color: var(--accent); font-size: 0.82rem; }

/* ── footer ── */
.footer {
    text-align: center; padding: 2.5rem 0 1.5rem;
    color: var(--muted); font-size: 0.75rem;
    border-top: 1px solid var(--border); margin-top: 4rem; line-height: 2;
}
</style>
""", unsafe_allow_html=True)

# ── constants ─────────────────────────────────────────────────────────────────
MAX_LEN = 512; SWIN_IMG_WIDTH = 32; SWIN_WINDOW_SIZE = 4
AUDIO_DIM = 128; VISUAL_DIM = 171
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))

# ── model definitions ─────────────────────────────────────────────────────────
class SourceAdapter(nn.Module):
    def __init__(self, in_dim, out_dim=SWIN_IMG_WIDTH):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim); self.norm = nn.LayerNorm(out_dim)
    def forward(self, x): return self.norm(self.proj(x))

class InputAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.audio_adapter  = SourceAdapter(AUDIO_DIM)
        self.visual_adapter = SourceAdapter(VISUAL_DIM)
    def forward(self, a, v): return self.audio_adapter(a), self.visual_adapter(v)

class SwinTower(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(
            "swin_tiny_patch4_window7_224", pretrained=False, in_chans=1,
            img_size=(MAX_LEN, SWIN_IMG_WIDTH), window_size=SWIN_WINDOW_SIZE, num_classes=0)
        self.out_dim = self.backbone.num_features
    def forward(self, x): return self.backbone.forward_features(x.unsqueeze(1))

class CrossAttentionFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.a2v = nn.MultiheadAttention(dim, 8, dropout=0.1, batch_first=True)
        self.v2a = nn.MultiheadAttention(dim, 8, dropout=0.1, batch_first=True)
        self.na = nn.LayerNorm(dim); self.nv = nn.LayerNorm(dim)
    def forward(self, a, v):
        af, _ = self.a2v(a, v, v); vf, _ = self.v2a(v, a, a)
        return self.na(a + af), self.nv(v + vf)

class SwinDepressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.adapter = InputAdapter()
        self.audio_swin = SwinTower(); self.visual_swin = SwinTower()
        dim = self.audio_swin.out_dim
        self.fusion = CrossAttentionFusion(dim)
        self.classifier = nn.Sequential(
            nn.Linear(dim * 2, dim), nn.ReLU(), nn.Dropout(0.3), nn.Linear(dim, 1))
    def forward(self, b):
        a, v = self.adapter(b["audio"], b["visual"])
        am, vm = self.audio_swin(a), self.visual_swin(v)
        B = am.shape[0]
        at = am.reshape(B, -1, am.shape[-1]); vt = vm.reshape(B, -1, vm.shape[-1])
        af, vf = self.fusion(at, vt)
        return self.classifier(torch.cat([af.mean(1), vf.mean(1)], -1)).squeeze(-1)

class InputAdapterBaseline(nn.Module):
    def __init__(self):
        super().__init__()
        self.ap = nn.Sequential(nn.Linear(128, 128), nn.LayerNorm(128))
        self.vp = nn.Sequential(nn.Linear(171, 171), nn.LayerNorm(171))
    def forward(self, a, v): return self.ap(a), self.vp(v)

class TemporalBlock(nn.Module):
    def __init__(self, ic, oc, ks=3, d=1, dr=0.2):
        super().__init__()
        p = (ks - 1) * d
        self.conv = nn.Conv1d(ic, oc, ks, padding=p, dilation=d); self.chomp = p
        self.relu = nn.ReLU(); self.drop = nn.Dropout(dr)
        self.ds = nn.Conv1d(ic, oc, 1) if ic != oc else None
        self.norm = nn.BatchNorm1d(oc)
    def forward(self, x):
        o = self.conv(x)
        if self.chomp: o = o[:, :, :-self.chomp]
        o = self.relu(self.norm(o)); o = self.drop(o)
        return o + (x if self.ds is None else self.ds(x))

class TCN(nn.Module):
    def __init__(self, d, ch=(128, 128, 128)):
        super().__init__()
        ls, p = [], d
        for i, c in enumerate(ch): ls.append(TemporalBlock(p, c, d=2 ** i)); p = c
        self.net = nn.Sequential(*ls)
    def forward(self, x): return self.net(x.transpose(1, 2)).transpose(1, 2)

class TCNBaseline(nn.Module):
    def __init__(self):
        super().__init__()
        self.adapter = InputAdapterBaseline()
        self.at = TCN(128); self.vt = TCN(171)
        self.cls = nn.Sequential(nn.Linear(256, 128), nn.ReLU(),
                                   nn.Dropout(0.3), nn.Linear(128, 1))
    def pool(self, x, m):
        m = m.unsqueeze(-1).float()
        return (x * m).sum(1) / m.sum(1).clamp(min=1)
    def forward(self, b):
        a, v = self.adapter(b["audio"], b["visual"])
        return self.cls(torch.cat([
            self.pool(self.at(a), b["audio_mask"]),
            self.pool(self.vt(v), b["visual_mask"])
        ], -1)).squeeze(-1)

# ── model loading ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_models():
    swin = SwinDepressionModel()
    swin.load_state_dict(torch.load(
        os.path.join(MODEL_DIR, "swin_seed123.pt"),
        map_location="cpu", weights_only=False))
    swin.eval()
    base = TCNBaseline()
    base.load_state_dict(torch.load(
        os.path.join(MODEL_DIR, "baseline_seed2024.pt"),
        map_location="cpu", weights_only=False))
    base.eval()
    return swin, base

# ── inference ─────────────────────────────────────────────────────────────────
def preprocess(audio_arr, visual_arr):
    if audio_arr.shape[0] > MAX_LEN: audio_arr = audio_arr[:MAX_LEN]
    if visual_arr.shape[0] > MAX_LEN:
        idx = np.linspace(0, visual_arr.shape[0] - 1, MAX_LEN).round().astype(int)
        visual_arr = visual_arr[idx]
    a_len, v_len = audio_arr.shape[0], visual_arr.shape[0]
    ap = np.zeros((MAX_LEN, AUDIO_DIM), dtype=np.float32)
    vp = np.zeros((MAX_LEN, VISUAL_DIM), dtype=np.float32)
    ap[:a_len] = audio_arr; vp[:v_len] = visual_arr
    am = torch.zeros(1, MAX_LEN, dtype=torch.bool); am[0, :a_len] = True
    vm = torch.zeros(1, MAX_LEN, dtype=torch.bool); vm[0, :v_len] = True
    return {"audio": torch.tensor(ap).unsqueeze(0), "visual": torch.tensor(vp).unsqueeze(0),
            "audio_mask": am, "visual_mask": vm}

def predict(model, batch):
    with torch.no_grad():
        return torch.sigmoid(model(batch)).item()

def gauge(prob, title):
    color = "#ef4444" if prob >= 0.5 else "#22c55e"
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=round(prob * 100, 1),
        number={"suffix": "%", "font": {"size": 28, "family": "JetBrains Mono"}},
        title={"text": title, "font": {"size": 11, "family": "Inter"}},
        gauge={"axis": {"range": [0, 100]},
               "bar": {"color": color, "thickness": 0.25},
               "steps": [{"range": [0, 50], "color": "rgba(34,197,94,0.08)"},
                          {"range": [50, 100], "color": "rgba(239,68,68,0.08)"}],
               "threshold": {"line": {"color": "gray", "width": 2},
                              "thickness": 0.8, "value": 50}}))
    fig.update_layout(height=200, margin=dict(t=45, b=5, l=20, r=20),
                       paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    return fig

def show_img(filename, caption=None):
    path = os.path.join(MODEL_DIR, filename)
    if os.path.exists(path):
        st.image(Image.open(path), caption=caption, use_column_width=True)

# ═════════════════════════════════════════════════════════════════════════════
# HERO
# ═════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero">
    <div class="hero-eyebrow">Final Year Project · Computer Science</div>
    <h1>Depression Detection<br><span>using Swin Transformer</span></h1>
    <p class="hero-sub">
        A multimodal deep learning system that analyses audio and visual behavioural
        features extracted from video recordings to detect signs of depression —
        outperforming a strong TCN baseline across four evaluation metrics.
    </p>
    <div class="hero-pills">
        <span class="pill">LMVD Dataset</span>
        <span class="pill">1,804 samples</span>
        <span class="pill">Audio + Visual fusion</span>
        <span class="pill">Swin Transformer</span>
        <span class="pill">Cross-attention fusion</span>
        <span class="pill">Multi-seed evaluation</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 1 — ABOUT
# ═════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-head">About this project</div>', unsafe_allow_html=True)
st.markdown("""<div class="section-sub">
    Depression is one of the most prevalent mental health conditions worldwide, yet screening
    relies almost entirely on subjective self-report instruments. This project explores whether
    a Swin Transformer applied to multimodal behavioural features can automate and improve
    that screening — learning jointly from acoustic and facial cues captured in short video
    recordings collected from social media platforms.
</div>""", unsafe_allow_html=True)

st.markdown("""
<div class="stat-grid">
    <div class="stat-card"><div class="stat-val">1,804</div><div class="stat-label">LMVD samples</div></div>
    <div class="stat-card"><div class="stat-val">50/50</div><div class="stat-label">class balance</div></div>
    <div class="stat-card"><div class="stat-val">0.846</div><div class="stat-label">Swin AUC</div></div>
    <div class="stat-card"><div class="stat-val">+2.4pp</div><div class="stat-label">F1 over baseline</div></div>
    <div class="stat-card"><div class="stat-val">+4.6pp</div><div class="stat-label">Accuracy over baseline</div></div>
    <div class="stat-card"><div class="stat-val">3</div><div class="stat-label">evaluation seeds</div></div>
</div>
""", unsafe_allow_html=True)

show_img("eda_class_balance.png", caption="Class and split distribution across the LMVD dataset")

st.markdown('<hr class="divider">', unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 2 — DATASET
# ═════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-head">Dataset · LMVD</div>', unsafe_allow_html=True)
st.markdown("""<div class="section-sub">
    The Large-scale Multimodal Video Depression (LMVD) dataset contains 1,823 vlog recordings
    collected from Bilibili, TikTok, Sina Weibo, and YouTube — each labelled by clinical
    annotation. After filtering 19 near-empty audio files, 1,804 samples remain with a
    near-perfect 50/50 class balance, split 70/15/15 into train, validation, and test sets
    via stratified sampling.
</div>""", unsafe_allow_html=True)

st.markdown("""
<div class="info-row"><span class="info-key">Audio features</span><span class="info-val">VGGish · 128-dim per second</span></div>
<div class="info-row"><span class="info-key">Visual features</span><span class="info-val">OpenFace (AU + landmarks + gaze + head pose) · 171-dim</span></div>
<div class="info-row"><span class="info-key">Audio sequence length</span><span class="info-val">variable · min 10, mean 567, max 11,044 frames</span></div>
<div class="info-row"><span class="info-key">Visual sequence length</span><span class="info-val">fixed at 915 frames per sample (TCN-pooled)</span></div>
<div class="info-row"><span class="info-key">Train / Val / Test</span><span class="info-val">1,262 / 271 / 271 samples</span></div>
<div class="info-row"><span class="info-key">Sequence truncation</span><span class="info-val">512 frames · affects 36% of samples</span></div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
show_img("eda_audio_lengths.png", caption="Audio sequence length distribution (VGGish embeddings) — dashed line marks the 512-frame truncation point")

st.markdown('<hr class="divider">', unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 3 — ARCHITECTURE
# ═════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-head">Model architecture</div>', unsafe_allow_html=True)
st.markdown("""<div class="section-sub">
    Each modality is projected into a 32-dim pseudo-image grid (512×32), treated as a
    single-channel spatial input, and passed through an independent pretrained Swin-Tiny
    backbone. The resulting token sequences are fused via bidirectional cross-attention
    before a shared classification head produces a depression-risk score.
</div>""", unsafe_allow_html=True)

st.markdown("""
<div class="pipeline">
    <div class="step">
        <div class="step-num">01</div>
        <div class="step-title">Input adapter</div>
        <div class="step-desc">Projects audio (128→32) and visual (171→32) via Linear + LayerNorm into a shared pseudo-image width</div>
        <div class="step-arrow">›</div>
    </div>
    <div class="step">
        <div class="step-num">02</div>
        <div class="step-title">Pseudo-image</div>
        <div class="step-desc">Reshaped to (512, 32) and treated as a single-channel 2D grid — enabling Swin's windowed patch attention</div>
        <div class="step-arrow">›</div>
    </div>
    <div class="step">
        <div class="step-num">03</div>
        <div class="step-title">Swin-Tiny towers</div>
        <div class="step-desc">Two independent pretrained Swin-Tiny backbones (one per modality), window_size=4, differential learning rate</div>
        <div class="step-arrow">›</div>
    </div>
    <div class="step">
        <div class="step-num">04</div>
        <div class="step-title">Cross-attention fusion</div>
        <div class="step-desc">Bidirectional multi-head attention lets audio tokens attend to visual context and vice versa</div>
        <div class="step-arrow">›</div>
    </div>
    <div class="step">
        <div class="step-num">05</div>
        <div class="step-title">Classifier</div>
        <div class="step-desc">Mean-pooled fused tokens → Linear → ReLU → Dropout → Linear → depression-risk logit</div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown('<hr class="divider">', unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 4 — RESULTS
# ═════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-head">Results</div>', unsafe_allow_html=True)
st.markdown("""<div class="section-sub">
    Both models were evaluated across 3 independent random seeds (42, 123, 2024).
    Results are reported as mean ± std on the held-out test set of 271 samples.
    AUC is the primary metric: it is threshold-independent, most stable across seeds,
    and directly comparable to published depression-detection literature.
</div>""", unsafe_allow_html=True)

show_img("results_comparison.png", caption="Mean ± std across 3 seeds — Swin Transformer vs TCN Baseline on all five metrics")

c1, c2 = st.columns(2)
with c1:
    show_img("roc_curves.png", caption="ROC curves — best seed per model")
with c2:
    show_img("confusion_matrices.png", caption="Confusion matrices — best seed per model")

show_img("training_curves_swin.png", caption="Loss and F1 per epoch during Swin training (best seed)")

st.markdown('<hr class="divider">', unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 5 — ABLATION
# ═════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-head">Ablation study</div>', unsafe_allow_html=True)
st.markdown("""<div class="section-sub">
    To isolate each modality's contribution, audio-only and visual-only Swin variants
    were trained alongside the full fused model. The results confirm that visual features
    carry the primary discriminative signal, audio alone performs near-randomly on this
    dataset, and cross-attention fusion over both streams yields the best performance.
</div>""", unsafe_allow_html=True)

show_img("ablation_chart.png", caption="Audio-only vs Visual-only vs Fused Swin vs TCN Baseline — F1, Accuracy, AUC")

st.markdown("""
<div class="info-banner">
    <strong>Key finding:</strong> Visual-only Swin (AUC 0.784) already approaches the full TCN
    baseline (AUC 0.817), and fusion pushes decisively to 0.846. Audio-only Swin is near-random
    (AUC 0.587), confirming that VGGish embeddings alone are insufficient without the
    complementary facial behavioural signal from OpenFace.
</div>
""", unsafe_allow_html=True)

st.markdown('<hr class="divider">', unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 6 — LIVE DEMO
# ═════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-head">Live inference demo</div>', unsafe_allow_html=True)
st.markdown("""<div class="section-sub">
    Upload a pair of pre-extracted feature files (.npy) for one participant.
    Both the proposed Swin model and the TCN baseline will run simultaneously
    so you can compare their outputs directly.
</div>""", unsafe_allow_html=True)

st.markdown("""
<div class="info-banner">
    <strong>Expected file formats —</strong>
    audio: shape (T, 128) float32, VGGish embeddings (one 128-dim vector per second) ·
    visual: shape (T, 171) float32, OpenFace AU / landmark / gaze / head-pose features
</div>
""", unsafe_allow_html=True)

uc1, uc2 = st.columns(2)
with uc1:
    st.markdown('<div class="upload-label">Audio features (.npy · 128-dim)</div>', unsafe_allow_html=True)
    audio_file = st.file_uploader("audio", type=["npy"], label_visibility="collapsed", key="au")
with uc2:
    st.markdown('<div class="upload-label">Visual features (.npy · 171-dim)</div>', unsafe_allow_html=True)
    visual_file = st.file_uploader("visual", type=["npy"], label_visibility="collapsed", key="vi")

run_btn = st.button("Run inference", disabled=(audio_file is None or visual_file is None))

if audio_file and visual_file:
    audio_arr  = np.load(audio_file).astype(np.float32)
    visual_arr = np.load(visual_file).astype(np.float32)
    ok = True
    if audio_arr.ndim != 2 or audio_arr.shape[1] != AUDIO_DIM:
        st.error(f"Audio file must be shape (T, 128). Got {audio_arr.shape}.")
        ok = False
    if visual_arr.ndim != 2 or visual_arr.shape[1] != VISUAL_DIM:
        st.error(f"Visual file must be shape (T, 171). Got {visual_arr.shape}.")
        ok = False

    if ok and not run_btn:
        st.markdown(f"""
        <div class="metric-row">
            <div class="metric-box"><div class="metric-val">{audio_arr.shape[0]}</div>
                <div class="metric-name">audio frames</div></div>
            <div class="metric-box"><div class="metric-val">{audio_arr.shape[1]}</div>
                <div class="metric-name">audio dim</div></div>
            <div class="metric-box"><div class="metric-val">{visual_arr.shape[0]}</div>
                <div class="metric-name">visual frames</div></div>
            <div class="metric-box"><div class="metric-val">{visual_arr.shape[1]}</div>
                <div class="metric-name">visual dim</div></div>
        </div>
        """, unsafe_allow_html=True)

    if ok and run_btn:
        with st.spinner("Loading models and running inference..."):
            swin_m, base_m = load_models()
            batch     = preprocess(audio_arr, visual_arr)
            swin_prob = predict(swin_m, batch)
            base_prob = predict(base_m, batch)

        rc1, rc2 = st.columns(2)
        for col, prob, name in [
            (rc1, swin_prob, "Swin Transformer"),
            (rc2, base_prob, "TCN Baseline"),
        ]:
            with col:
                depressed = prob >= 0.5
                cls     = "result-depressed" if depressed else "result-normal"
                verdict = "Depressed" if depressed else "Non-depressed"
                color   = "#dc2626" if depressed else "#16a34a"
                st.markdown(f"""
                <div class="result-card {cls}">
                    <div class="result-label">{name}</div>
                    <div class="result-verdict" style="color:{color}">{verdict}</div>
                    <div class="result-prob">P(depressed) = {prob:.4f}</div>
                </div>
                """, unsafe_allow_html=True)
                st.plotly_chart(
                    gauge(prob, f"{name} · Depression probability"),
                    use_container_width=True)

        st.markdown("#### Validated test-set performance (mean ± std, 3 seeds, n=271)")
        st.dataframe({
            "Metric":    ["F1",           "Precision",    "Recall",       "Accuracy",     "AUC"],
            "Baseline":  ["0.756 ±0.012", "0.687 ±0.042", "0.844 ±0.038", "0.727 ±0.028", "0.817 ±0.027"],
            "Swin":      ["0.780 ±0.021", "0.751 ±0.004", "0.812 ±0.045", "0.772 ±0.014", "0.846 ±0.014"],
            "Δ (mean)":  ["+0.024 ✓",     "+0.064 ✓",     "−0.032",       "+0.046 ✓",     "+0.030 ✓"],
        }, use_container_width=True, hide_index=True)

        st.markdown("""
        <div class="warn-banner">
            ⚠️ This system is a research prototype and is <strong>not a clinical diagnostic tool</strong>.
            Predictions should not be used to inform clinical decisions.
        </div>
        """, unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# FOOTER
# ═════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="footer">
    Depression Detection using Swin Transformer · LMVD Dataset<br>
    Final Year Project · Computer Science<br><br>
    For research and demonstration purposes only — not a clinical diagnostic tool.
</div>
""", unsafe_allow_html=True)