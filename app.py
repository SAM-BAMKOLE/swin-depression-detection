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
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: #0f1117; color: #e8eaf0; }
.block-container { max-width: 860px; }

.hero {
    text-align: center; padding: 3rem 0 2rem;
    border-bottom: 1px solid #1e2130; margin-bottom: 2.5rem;
}
.hero-eyebrow {
    font-size: 0.7rem; font-weight: 600; letter-spacing: 0.18em;
    text-transform: uppercase; color: #3b82f6; margin-bottom: 0.8rem;
}
.hero h1 {
    font-size: 2.2rem; font-weight: 700; letter-spacing: -0.03em;
    color: #f1f5f9; margin: 0 0 0.8rem; line-height: 1.2;
}
.hero h1 span { color: #3b82f6; }
.hero-sub {
    font-size: 0.95rem; color: #6b7280; line-height: 1.7;
    font-weight: 300; margin-bottom: 1.5rem;
}
.hero-pills { display: flex; gap: 0.5rem; justify-content: center; flex-wrap: wrap; }
.pill {
    background: #161b27; border: 1px solid #1e2130; border-radius: 999px;
    padding: 0.25rem 0.8rem; font-size: 0.72rem; color: #94a3b8; font-weight: 500;
}

.section-head {
    font-size: 1.2rem; font-weight: 600; color: #f1f5f9;
    letter-spacing: -0.02em; margin: 0 0 0.35rem;
}
.section-sub {
    font-size: 0.86rem; color: #6b7280; margin: 0 0 1.2rem; line-height: 1.7;
}
hr.div { border: none; border-top: 1px solid #1e2130; margin: 2.5rem 0; }

.stat-grid { display: flex; gap: 0.7rem; flex-wrap: wrap; margin-bottom: 1.5rem; }
.stat-card {
    flex: 1; min-width: 100px; background: #161b27;
    border: 1px solid #1e2130; border-radius: 10px; padding: 1rem; text-align: center;
}
.stat-val { font-family: 'JetBrains Mono', monospace; font-size: 1.4rem; font-weight: 500; color: #3b82f6; }
.stat-label { font-size: 0.65rem; text-transform: uppercase; letter-spacing: 0.1em; color: #475569; margin-top: 0.25rem; }

.pipeline { display: flex; gap: 0; margin: 1.2rem 0 1.8rem; }
.step {
    flex: 1; background: #161b27; border: 1px solid #1e2130;
    padding: 0.9rem 0.8rem; position: relative;
}
.step:first-child { border-radius: 8px 0 0 8px; }
.step:last-child  { border-radius: 0 8px 8px 0; }
.step-num { font-size: 0.6rem; font-weight: 600; letter-spacing: 0.1em; text-transform: uppercase; color: #3b82f6; margin-bottom: 0.3rem; }
.step-title { font-size: 0.8rem; font-weight: 600; color: #e2e8f0; margin-bottom: 0.2rem; }
.step-desc { font-size: 0.7rem; color: #6b7280; line-height: 1.45; }
.step-arrow { position: absolute; right: -9px; top: 50%; transform: translateY(-50%); color: #334155; font-size: 1rem; z-index: 10; }

.info-row {
    display: flex; justify-content: space-between; align-items: center;
    background: #161b27; border: 1px solid #1e2130; border-radius: 8px;
    padding: 0.65rem 1rem; margin-bottom: 0.45rem;
}
.info-key { color: #6b7280; font-size: 0.8rem; }
.info-val { font-family: 'JetBrains Mono', monospace; color: #3b82f6; font-size: 0.8rem; }

.upload-label {
    font-size: 0.78rem; font-weight: 500; text-transform: uppercase;
    letter-spacing: 0.08em; color: #6b7280; margin-bottom: 0.5rem;
}
.result-card { border-radius: 12px; padding: 2rem; text-align: center; margin: 1.5rem 0; }
.result-depressed { background: linear-gradient(135deg, #1f0f0f 0%, #2d1111 100%); border: 1px solid #7f1d1d; }
.result-normal    { background: linear-gradient(135deg, #0a1f14 0%, #0f2d1a 100%); border: 1px solid #14532d; }
.result-label { font-size: 0.75rem; font-weight: 500; letter-spacing: 0.12em; text-transform: uppercase; margin-bottom: 0.5rem; }
.result-verdict { font-size: 2rem; font-weight: 600; margin: 0.2rem 0 0.6rem; letter-spacing: -0.02em; }
.result-prob { font-family: 'JetBrains Mono', monospace; font-size: 1rem; color: #9ca3af; }

.metric-row { display: flex; gap: 0.75rem; margin-top: 1rem; }
.metric-box {
    flex: 1; background: #161b27; border: 1px solid #1e2130;
    border-radius: 8px; padding: 0.9rem; text-align: center;
}
.metric-val2 { font-family: 'JetBrains Mono', monospace; font-size: 1.25rem; font-weight: 500; color: #e8eaf0; }
.metric-name { font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.08em; color: #6b7280; margin-top: 0.2rem; }

.info-banner {
    background: #161b27; border: 1px solid #1e2130;
    border-left: 3px solid #3b82f6; border-radius: 6px;
    padding: 0.8rem 1rem; font-size: 0.82rem; color: #9ca3af; margin-bottom: 1.5rem; line-height: 1.6;
}
.warn-banner {
    background: #161b27; border: 1px solid #1e2130;
    border-left: 3px solid #f59e0b; border-radius: 6px;
    padding: 0.8rem 1rem; font-size: 0.82rem; color: #9ca3af; margin-bottom: 1.5rem; line-height: 1.6;
}
.footer {
    text-align: center; padding: 2rem 0 1rem; color: #374151;
    font-size: 0.75rem; border-top: 1px solid #1e2130; margin-top: 3rem; line-height: 2;
}

div[data-testid="stFileUploader"] { background: transparent; }
section[data-testid="stFileUploadDropzone"] {
    background: #0f1117 !important; border: 1.5px dashed #2d3748 !important; border-radius: 8px !important;
}
.stButton > button {
    width: 100%; background: #2563eb; color: white; border: none;
    border-radius: 8px; padding: 0.65rem 1rem; font-size: 0.9rem;
    font-weight: 500; font-family: 'Inter', sans-serif; cursor: pointer;
}
.stButton > button:hover { background: #1d4ed8; }
</style>
""", unsafe_allow_html=True)

# ── constants ─────────────────────────────────────────────────────────────────
MAX_LEN = 512
SWIN_IMG_WIDTH = 32
SWIN_WINDOW_SIZE = 4
AUDIO_DIM = 128
VISUAL_DIM = 171
MODEL_DIR = os.path.dirname(__file__)

# ── model definitions (EXACTLY as trained) ────────────────────────────────────
class SourceAdapter(nn.Module):
    def __init__(self, in_dim, out_dim=SWIN_IMG_WIDTH):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
    def forward(self, x):
        return self.norm(self.proj(x))

class InputAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.audio_adapter = SourceAdapter(AUDIO_DIM)
        self.visual_adapter = SourceAdapter(VISUAL_DIM)
    def forward(self, audio, visual):
        return self.audio_adapter(audio), self.visual_adapter(visual)

class SwinTower(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(
            "swin_tiny_patch4_window7_224", pretrained=False, in_chans=1,
            img_size=(MAX_LEN, SWIN_IMG_WIDTH), window_size=SWIN_WINDOW_SIZE, num_classes=0
        )
        self.out_dim = self.backbone.num_features
    def forward(self, x):
        return self.backbone.forward_features(x.unsqueeze(1))

class CrossAttentionFusion(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.a2v_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.v2a_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm_a = nn.LayerNorm(dim)
        self.norm_v = nn.LayerNorm(dim)
    def forward(self, audio_tokens, visual_tokens):
        a_fused, _ = self.a2v_attn(audio_tokens, visual_tokens, visual_tokens)
        v_fused, _ = self.v2a_attn(visual_tokens, audio_tokens, audio_tokens)
        return self.norm_a(audio_tokens + a_fused), self.norm_v(visual_tokens + v_fused)

class SwinDepressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.adapter = InputAdapter()
        self.audio_swin = SwinTower()
        self.visual_swin = SwinTower()
        dim = self.audio_swin.out_dim
        self.fusion = CrossAttentionFusion(dim)
        self.classifier = nn.Sequential(
            nn.Linear(dim * 2, dim), nn.ReLU(), nn.Dropout(0.3), nn.Linear(dim, 1)
        )
    def forward(self, batch):
        audio, visual = self.adapter(batch["audio"], batch["visual"])
        audio_map, visual_map = self.audio_swin(audio), self.visual_swin(visual)
        B = audio_map.shape[0]
        audio_tokens = audio_map.reshape(B, -1, audio_map.shape[-1])
        visual_tokens = visual_map.reshape(B, -1, visual_map.shape[-1])
        audio_fused, visual_fused = self.fusion(audio_tokens, visual_tokens)
        fused = torch.cat([audio_fused.mean(dim=1), visual_fused.mean(dim=1)], dim=-1)
        return self.classifier(fused).squeeze(-1)

class InputAdapterBaseline(nn.Module):
    def __init__(self):
        super().__init__()
        self.audio_proj = nn.Sequential(nn.Linear(128, 128), nn.LayerNorm(128))
        self.visual_proj = nn.Sequential(nn.Linear(171, 171), nn.LayerNorm(171))
    def forward(self, audio, visual):
        return self.audio_proj(audio), self.visual_proj(visual)

class TemporalBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, dilation=1, dropout=0.2):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation)
        self.chomp = padding
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None
        self.norm = nn.BatchNorm1d(out_ch)
    def forward(self, x):
        out = self.conv(x)
        if self.chomp > 0: out = out[:, :, :-self.chomp]
        out = self.relu(self.norm(out))
        out = self.dropout(out)
        return out + (x if self.downsample is None else self.downsample(x))

class TCN(nn.Module):
    def __init__(self, in_dim, channels=(128, 128, 128)):
        super().__init__()
        layers, prev = [], in_dim
        for i, ch in enumerate(channels):
            layers.append(TemporalBlock(prev, ch, dilation=2 ** i))
            prev = ch
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x.transpose(1, 2)).transpose(1, 2)

class TCNBaseline(nn.Module):
    def __init__(self):
        super().__init__()
        self.adapter = InputAdapterBaseline()
        self.audio_tcn = TCN(128)
        self.visual_tcn = TCN(171)
        self.classifier = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 1)
        )
    def masked_mean_pool(self, x, mask):
        mask = mask.unsqueeze(-1).float()
        return (x * mask).sum(1) / mask.sum(1).clamp(min=1)
    def forward(self, batch):
        audio, visual = self.adapter(batch["audio"], batch["visual"])
        audio_pooled = self.masked_mean_pool(self.audio_tcn(audio), batch["audio_mask"])
        visual_pooled = self.masked_mean_pool(self.visual_tcn(visual), batch["visual_mask"])
        return self.classifier(torch.cat([audio_pooled, visual_pooled], -1)).squeeze(-1)

# ── model loading (original — unchanged) ─────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_models():
    swin = SwinDepressionModel()
    swin.load_state_dict(torch.load(
        os.path.join(MODEL_DIR, "swin_seed123.pt"), map_location="cpu"
    ))
    swin.eval()
    baseline = TCNBaseline()
    baseline.load_state_dict(torch.load(
        os.path.join(MODEL_DIR, "baseline_seed2024.pt"), map_location="cpu"
    ))
    baseline.eval()
    return swin, baseline

# ── inference (original — unchanged) ─────────────────────────────────────────
def preprocess(audio_arr, visual_arr):
    if audio_arr.shape[0] > MAX_LEN:
        audio_arr = audio_arr[:MAX_LEN]
    if visual_arr.shape[0] > MAX_LEN:
        idxs = np.linspace(0, visual_arr.shape[0] - 1, MAX_LEN).round().astype(int)
        visual_arr = visual_arr[idxs]
    a_len = audio_arr.shape[0]
    v_len = visual_arr.shape[0]
    audio_padded  = np.zeros((MAX_LEN, AUDIO_DIM),  dtype=np.float32)
    visual_padded = np.zeros((MAX_LEN, VISUAL_DIM), dtype=np.float32)
    audio_padded[:a_len]  = audio_arr
    visual_padded[:v_len] = visual_arr
    audio_mask  = torch.zeros(1, MAX_LEN, dtype=torch.bool)
    visual_mask = torch.zeros(1, MAX_LEN, dtype=torch.bool)
    audio_mask[0, :a_len]  = True
    visual_mask[0, :v_len] = True
    return {
        "audio":       torch.tensor(audio_padded).unsqueeze(0),
        "visual":      torch.tensor(visual_padded).unsqueeze(0),
        "audio_mask":  audio_mask,
        "visual_mask": visual_mask,
    }

def predict(model, batch):
    with torch.no_grad():
        logit = model(batch)
        prob  = torch.sigmoid(logit).item()
    return prob

def make_gauge(prob, title):
    color = "#ef4444" if prob >= 0.5 else "#22c55e"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(prob * 100, 1),
        number={"suffix": "%", "font": {"size": 28, "color": "#e8eaf0", "family": "JetBrains Mono"}},
        title={"text": title, "font": {"size": 12, "color": "#6b7280", "family": "Inter"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#374151",
                     "tickfont": {"color": "#6b7280", "size": 10}},
            "bar": {"color": color, "thickness": 0.25},
            "bgcolor": "#161b27", "bordercolor": "#1e2130",
            "steps": [{"range": [0, 50],   "color": "#0a1f14"},
                       {"range": [50, 100], "color": "#1f0f0f"}],
            "threshold": {"line": {"color": "#ffffff", "width": 2},
                           "thickness": 0.8, "value": 50},
        }
    ))
    fig.update_layout(height=200, margin=dict(t=40, b=10, l=20, r=20),
                       paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                       font_color="#e8eaf0")
    return fig

def show_img(filename, caption=None):
    path = os.path.join(MODEL_DIR, filename)
    if os.path.exists(path):
        st.image(Image.open(path), caption=caption, use_column_width=True)

# ════════════════════════════════════════════════════════════════════
# HERO
# ════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero">
    <div class="hero-eyebrow">Final Year Project · Computer Science</div>
    <h1>Depression Detection<br><span>using Swin Transformer</span></h1>
    <p class="hero-sub">
        A multimodal deep learning system that analyses audio and visual
        behavioural features extracted from video recordings to screen for
        signs of depression — outperforming a strong TCN baseline on four
        out of five evaluation metrics.
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

# ════════════════════════════════════════════════════════════════════
# SECTION 1 — ABOUT
# ════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-head">About this project</div>', unsafe_allow_html=True)
st.markdown("""<div class="section-sub">
    Depression is one of the most prevalent mental health conditions worldwide, yet
    clinical screening still relies almost entirely on subjective self-report
    instruments. This project explores whether a Swin Transformer, applied to
    multimodal behavioural features, can automate and improve depression screening —
    learning jointly from acoustic and facial cues captured in short video recordings
    collected from social media platforms.
</div>""", unsafe_allow_html=True)

st.markdown("""
<div class="stat-grid">
    <div class="stat-card"><div class="stat-val">1,804</div><div class="stat-label">samples</div></div>
    <div class="stat-card"><div class="stat-val">50/50</div><div class="stat-label">class balance</div></div>
    <div class="stat-card"><div class="stat-val">0.846</div><div class="stat-label">Swin AUC</div></div>
    <div class="stat-card"><div class="stat-val">+2.4pp</div><div class="stat-label">F1 gain</div></div>
    <div class="stat-card"><div class="stat-val">+4.6pp</div><div class="stat-label">accuracy gain</div></div>
    <div class="stat-card"><div class="stat-val">3</div><div class="stat-label">eval seeds</div></div>
</div>
""", unsafe_allow_html=True)

show_img("eda_class_balance.png", caption="Class balance and split distribution across the LMVD dataset")

st.markdown('<hr class="div">', unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════
# SECTION 2 — DATASET
# ════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-head">Dataset · LMVD</div>', unsafe_allow_html=True)
st.markdown("""<div class="section-sub">
    The Large-scale Multimodal Video Depression (LMVD) dataset contains 1,823 vlog
    recordings collected from Bilibili, TikTok, Sina Weibo, and YouTube, each labelled
    by clinical annotation. After filtering 19 near-empty audio files, 1,804 samples
    remain with a near-perfect 50/50 class balance, split 70/15/15 via stratified sampling.
</div>""", unsafe_allow_html=True)

st.markdown("""
<div class="info-row"><span class="info-key">Audio features</span><span class="info-val">VGGish · 128-dim per second</span></div>
<div class="info-row"><span class="info-key">Visual features</span><span class="info-val">OpenFace · AU + landmarks + gaze + head pose · 171-dim</span></div>
<div class="info-row"><span class="info-key">Audio sequence length</span><span class="info-val">variable · mean 567 frames · max 11,044</span></div>
<div class="info-row"><span class="info-key">Visual sequence length</span><span class="info-val">fixed at 915 frames (TCN-pooled)</span></div>
<div class="info-row"><span class="info-key">Train / Val / Test split</span><span class="info-val">1,262 / 271 / 271 samples</span></div>
<div class="info-row"><span class="info-key">Sequence truncation</span><span class="info-val">512 frames · affects 36% of samples</span></div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
show_img("eda_audio_lengths.png", caption="Audio sequence length distribution — dashed line marks the 512-frame truncation point")

st.markdown('<hr class="div">', unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════
# SECTION 3 — ARCHITECTURE
# ════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-head">Model architecture</div>', unsafe_allow_html=True)
st.markdown("""<div class="section-sub">
    Each modality is projected into a 32-dim pseudo-image grid (512×32),
    treated as a single-channel spatial input, and processed by an independent
    pretrained Swin-Tiny backbone. The resulting token sequences are fused via
    bidirectional cross-attention before a shared classification head.
</div>""", unsafe_allow_html=True)

st.markdown("""
<div class="pipeline">
    <div class="step">
        <div class="step-num">01</div>
        <div class="step-title">Input adapter</div>
        <div class="step-desc">Linear + LayerNorm projects audio (128→32) and visual (171→32)</div>
        <div class="step-arrow">›</div>
    </div>
    <div class="step">
        <div class="step-num">02</div>
        <div class="step-title">Pseudo-image</div>
        <div class="step-desc">Reshaped to (512, 32) single-channel 2D grid for Swin's patch attention</div>
        <div class="step-arrow">›</div>
    </div>
    <div class="step">
        <div class="step-num">03</div>
        <div class="step-title">Swin-Tiny towers</div>
        <div class="step-desc">Two pretrained backbones (one per modality), window_size=4, differential LR</div>
        <div class="step-arrow">›</div>
    </div>
    <div class="step">
        <div class="step-num">04</div>
        <div class="step-title">Cross-attention</div>
        <div class="step-desc">Audio tokens attend to visual context and vice versa before mean pooling</div>
        <div class="step-arrow">›</div>
    </div>
    <div class="step">
        <div class="step-num">05</div>
        <div class="step-title">Classifier</div>
        <div class="step-desc">Linear → ReLU → Dropout → Linear producing a depression-risk logit</div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown('<hr class="div">', unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════
# SECTION 4 — RESULTS
# ════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-head">Results</div>', unsafe_allow_html=True)
st.markdown("""<div class="section-sub">
    Both models were evaluated across 3 independent random seeds (42, 123, 2024)
    on the same held-out test set of 271 samples. AUC is the primary reported metric
    as it is threshold-independent and consistent across seeds.
</div>""", unsafe_allow_html=True)

show_img("results_comparison.png", caption="Mean ± std across 3 seeds — Swin Transformer vs TCN Baseline")

c1, c2 = st.columns(2)
with c1:
    show_img("roc_curves.png", caption="ROC curves — best seed per model")
with c2:
    show_img("confusion_matrices.png", caption="Confusion matrices — best seed per model")

show_img("training_curves_swin.png", caption="Loss and F1 per epoch during Swin Transformer training (best seed)")

st.markdown('<hr class="div">', unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════
# SECTION 5 — ABLATION
# ════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-head">Ablation study</div>', unsafe_allow_html=True)
st.markdown("""<div class="section-sub">
    Audio-only and visual-only Swin variants were trained to isolate each modality's
    contribution. Visual features carry the primary signal (AUC 0.784), audio alone
    is near-random (AUC 0.587), and cross-attention fusion decisively outperforms
    both unimodal variants (AUC 0.846).
</div>""", unsafe_allow_html=True)

show_img("ablation_chart.png", caption="Audio-only vs Visual-only vs Fused Swin vs TCN Baseline")

st.markdown("""
<div class="info-banner">
    <strong>Key finding:</strong> Fusion is genuinely necessary — it is not one modality
    carrying the result. Visual-only Swin already approaches the full TCN baseline
    on AUC (0.784 vs 0.817), and adding audio via cross-attention pushes decisively
    to 0.846, confirming complementary inter-modal signal.
</div>
""", unsafe_allow_html=True)

st.markdown('<hr class="div">', unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════
# SECTION 6 — LIVE DEMO (original inference, untouched)
# ════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-head">Live inference demo</div>', unsafe_allow_html=True)
st.markdown("""<div class="section-sub">
    Upload a pair of pre-extracted feature files (.npy) for one participant.
    Both the proposed Swin model and the TCN baseline will run simultaneously.
</div>""", unsafe_allow_html=True)

st.markdown("""
<div class="info-banner">
    Upload a pair of pre-extracted feature files (.npy) for one participant —
    VGGish audio embeddings (128-dim) and TCN-processed visual features (171-dim).
    Both the proposed Swin model and the TCN baseline will run inference simultaneously.
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    st.markdown('<div class="upload-label">Audio features (.npy)</div>', unsafe_allow_html=True)
    audio_file = st.file_uploader("audio", type=["npy"], label_visibility="collapsed",
                                   key="audio_upload")
with col2:
    st.markdown('<div class="upload-label">Visual features (.npy)</div>', unsafe_allow_html=True)
    visual_file = st.file_uploader("visual", type=["npy"], label_visibility="collapsed",
                                    key="visual_upload")

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
            <div class="metric-box"><div class="metric-val2">{audio_arr.shape[0]}</div><div class="metric-name">audio frames</div></div>
            <div class="metric-box"><div class="metric-val2">{audio_arr.shape[1]}</div><div class="metric-name">audio dim</div></div>
            <div class="metric-box"><div class="metric-val2">{visual_arr.shape[0]}</div><div class="metric-name">visual frames</div></div>
            <div class="metric-box"><div class="metric-val2">{visual_arr.shape[1]}</div><div class="metric-name">visual dim</div></div>
        </div>
        """, unsafe_allow_html=True)

    if ok and run_btn:
        with st.spinner("Loading models and running inference..."):
            swin_model, baseline_model = load_models()
            batch     = preprocess(audio_arr, visual_arr)
            swin_prob = predict(swin_model, batch)
            base_prob = predict(baseline_model, batch)

        swin_depressed = swin_prob >= 0.5
        base_depressed = base_prob >= 0.5

        st.markdown("### Results")

        tab1, tab2 = st.tabs(["Swin Transformer (proposed)", "TCN Baseline"])

        with tab1:
            card_cls = "result-depressed" if swin_depressed else "result-normal"
            verdict  = "Depressed" if swin_depressed else "Non-depressed"
            color    = "#f87171" if swin_depressed else "#4ade80"
            st.markdown(f"""
            <div class="result-card {card_cls}">
                <div class="result-label">Swin Transformer · Prediction</div>
                <div class="result-verdict" style="color:{color}">{verdict}</div>
                <div class="result-prob">P(depressed) = {swin_prob:.4f}</div>
            </div>
            """, unsafe_allow_html=True)
            st.plotly_chart(make_gauge(swin_prob, "Depression probability"), use_container_width=True)

        with tab2:
            card_cls = "result-depressed" if base_depressed else "result-normal"
            verdict  = "Depressed" if base_depressed else "Non-depressed"
            color    = "#f87171" if base_depressed else "#4ade80"
            st.markdown(f"""
            <div class="result-card {card_cls}">
                <div class="result-label">TCN Baseline · Prediction</div>
                <div class="result-verdict" style="color:{color}">{verdict}</div>
                <div class="result-prob">P(depressed) = {base_prob:.4f}</div>
            </div>
            """, unsafe_allow_html=True)
            st.plotly_chart(make_gauge(base_prob, "Depression probability"), use_container_width=True)

        st.markdown("#### Model comparison (test set, mean ± std, 3 seeds)")
        st.dataframe({
            "Metric":   ["F1",     "Precision", "Recall",  "Accuracy", "AUC"],
            "Baseline": ["0.756 ±0.012", "0.687 ±0.042", "0.844 ±0.038", "0.727 ±0.028", "0.817 ±0.027"],
            "Swin":     ["0.780 ±0.021", "0.751 ±0.004", "0.812 ±0.045", "0.772 ±0.014", "0.846 ±0.014"],
            "Δ":        ["+0.024 ✓", "+0.064 ✓", "−0.032", "+0.046 ✓", "+0.030 ✓"],
        }, use_container_width=True, hide_index=True)

        st.markdown("""
        <div class="warn-banner">
            ⚠️ This system is a research prototype and is <strong>not a clinical diagnostic tool</strong>.
            Predictions should not be used to inform any clinical decisions.
        </div>
        """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════
# FOOTER
# ════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="footer">
    Depression Detection using Swin Transformer · LMVD Dataset<br>
    Final Year Project · Computer Science<br>
    For research and demonstration purposes only — not a clinical diagnostic tool.
</div>
""", unsafe_allow_html=True)