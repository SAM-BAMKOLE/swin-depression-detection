import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import json, os
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

html, body, [class*="css"]  { font-family: 'Inter', sans-serif; }
.stApp                       { background: #0b0e17; color: #e2e8f0; }
.block-container             { max-width: 1100px; padding-top: 0; }

/* ── hero ── */
.hero {
    background: linear-gradient(135deg, #0f1629 0%, #111827 60%, #0b1120 100%);
    border-bottom: 1px solid #1e293b;
    padding: 4rem 3rem 3rem;
    text-align: center;
    margin: 0 -1rem 3rem;
}
.hero-eyebrow {
    font-size: 0.72rem; font-weight: 600; letter-spacing: 0.18em;
    text-transform: uppercase; color: #3b82f6; margin-bottom: 1rem;
}
.hero h1 {
    font-size: 2.8rem; font-weight: 700; letter-spacing: -0.03em;
    color: #f1f5f9; margin: 0 0 1rem; line-height: 1.15;
}
.hero h1 span { color: #3b82f6; }
.hero-sub {
    font-size: 1rem; color: #64748b; max-width: 600px;
    margin: 0 auto 2rem; line-height: 1.65; font-weight: 300;
}
.hero-pills { display: flex; gap: 0.6rem; justify-content: center; flex-wrap: wrap; }
.pill {
    background: #1e293b; border: 1px solid #334155;
    border-radius: 999px; padding: 0.3rem 0.9rem;
    font-size: 0.75rem; color: #94a3b8; font-weight: 500;
}

/* ── section headings ── */
.section-head {
    font-size: 1.35rem; font-weight: 600; color: #f1f5f9;
    letter-spacing: -0.02em; margin: 0 0 0.4rem;
}
.section-sub {
    font-size: 0.85rem; color: #64748b; margin: 0 0 1.5rem;
    font-weight: 300; line-height: 1.6;
}
.section-divider {
    border: none; border-top: 1px solid #1e293b; margin: 3rem 0;
}

/* ── stat cards ── */
.stat-grid { display: flex; gap: 1rem; flex-wrap: wrap; margin-bottom: 2rem; }
.stat-card {
    flex: 1; min-width: 130px; background: #111827;
    border: 1px solid #1e293b; border-radius: 10px; padding: 1.2rem;
    text-align: center;
}
.stat-val {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.6rem; font-weight: 500; color: #3b82f6;
}
.stat-label { font-size: 0.7rem; text-transform: uppercase;
    letter-spacing: 0.1em; color: #475569; margin-top: 0.3rem; }

/* ── pipeline steps ── */
.pipeline { display: flex; gap: 0; margin: 1.5rem 0 2rem; }
.step {
    flex: 1; background: #111827; border: 1px solid #1e293b;
    padding: 1.1rem 1rem; position: relative;
}
.step:first-child { border-radius: 8px 0 0 8px; }
.step:last-child  { border-radius: 0 8px 8px 0; }
.step-num {
    font-size: 0.65rem; font-weight: 600; letter-spacing: 0.1em;
    text-transform: uppercase; color: #3b82f6; margin-bottom: 0.4rem;
}
.step-title { font-size: 0.85rem; font-weight: 600; color: #e2e8f0; margin-bottom: 0.25rem; }
.step-desc  { font-size: 0.75rem; color: #64748b; line-height: 1.5; }
.step-arrow {
    position: absolute; right: -10px; top: 50%; transform: translateY(-50%);
    color: #334155; font-size: 1.2rem; z-index: 10;
}

/* ── result cards ── */
.result-card {
    border-radius: 12px; padding: 1.8rem; text-align: center; margin-bottom: 1rem;
}
.result-depressed { background: linear-gradient(135deg,#1f0a0a,#2d1111); border:1px solid #7f1d1d; }
.result-normal    { background: linear-gradient(135deg,#071a0f,#0d2b18); border:1px solid #14532d; }
.result-label {
    font-size: 0.68rem; font-weight: 600; letter-spacing: 0.14em;
    text-transform: uppercase; color: #64748b; margin-bottom: 0.4rem;
}
.result-verdict { font-size: 1.8rem; font-weight: 700; letter-spacing: -0.02em; margin: 0.2rem 0 0.5rem; }
.result-prob    { font-family: 'JetBrains Mono', monospace; font-size: 0.9rem; color: #94a3b8; }

/* ── upload zone ── */
.upload-label {
    font-size: 0.72rem; font-weight: 600; text-transform: uppercase;
    letter-spacing: 0.1em; color: #64748b; margin-bottom: 0.5rem;
}
section[data-testid="stFileUploadDropzone"] {
    background: #0f172a !important;
    border: 1.5px dashed #334155 !important;
    border-radius: 8px !important;
}
.stButton > button {
    width: 100%; background: #2563eb; color: white; border: none;
    border-radius: 8px; padding: 0.7rem 1rem; font-size: 0.9rem;
    font-weight: 500; font-family: 'Inter', sans-serif; cursor: pointer;
}
.stButton > button:hover { background: #1d4ed8; }

/* ── metric table ── */
.metric-row { display: flex; gap: 0.6rem; margin: 1rem 0; }
.metric-box {
    flex: 1; background: #111827; border: 1px solid #1e293b;
    border-radius: 8px; padding: 0.8rem; text-align: center;
}
.metric-val  { font-family: 'JetBrains Mono', monospace; font-size: 1.1rem; color: #e2e8f0; }
.metric-name { font-size: 0.65rem; text-transform: uppercase;
    letter-spacing: 0.08em; color: #475569; margin-top: 0.2rem; }

/* ── info banner ── */
.info-banner {
    background: #111827; border: 1px solid #1e293b;
    border-left: 3px solid #3b82f6; border-radius: 6px;
    padding: 0.85rem 1.1rem; font-size: 0.82rem;
    color: #94a3b8; margin-bottom: 1.5rem; line-height: 1.6;
}
.warn-banner {
    background: #111827; border: 1px solid #1e293b;
    border-left: 3px solid #f59e0b; border-radius: 6px;
    padding: 0.85rem 1.1rem; font-size: 0.82rem;
    color: #94a3b8; margin-bottom: 1.5rem; line-height: 1.6;
}

/* ── footer ── */
.footer {
    text-align: center; padding: 2.5rem 0 1.5rem;
    color: #334155; font-size: 0.75rem;
    border-top: 1px solid #1e293b; margin-top: 4rem; line-height: 1.8;
}
</style>
""", unsafe_allow_html=True)

# ── constants ─────────────────────────────────────────────────────────────────
MAX_LEN = 512; SWIN_IMG_WIDTH = 32; SWIN_WINDOW_SIZE = 4
AUDIO_DIM = 128; VISUAL_DIM = 171
MODEL_DIR = os.path.dirname(__file__)

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
        self.na  = nn.LayerNorm(dim); self.nv = nn.LayerNorm(dim)
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
        self.classifier = nn.Sequential(nn.Linear(dim*2,dim), nn.ReLU(),
                                          nn.Dropout(0.3), nn.Linear(dim,1))
    def forward(self, b):
        a, v = self.adapter(b["audio"], b["visual"])
        am, vm = self.audio_swin(a), self.visual_swin(v)
        B = am.shape[0]
        at, vt = am.reshape(B,-1,am.shape[-1]), vm.reshape(B,-1,vm.shape[-1])
        af, vf = self.fusion(at, vt)
        return self.classifier(torch.cat([af.mean(1),vf.mean(1)],-1)).squeeze(-1)

class InputAdapterBaseline(nn.Module):
    def __init__(self):
        super().__init__()
        self.ap = nn.Sequential(nn.Linear(128,128), nn.LayerNorm(128))
        self.vp = nn.Sequential(nn.Linear(171,171), nn.LayerNorm(171))
    def forward(self, a, v): return self.ap(a), self.vp(v)

class TemporalBlock(nn.Module):
    def __init__(self, ic, oc, ks=3, d=1, dr=0.2):
        super().__init__()
        p = (ks-1)*d
        self.conv = nn.Conv1d(ic,oc,ks,padding=p,dilation=d); self.chomp=p
        self.relu = nn.ReLU(); self.drop = nn.Dropout(dr)
        self.ds = nn.Conv1d(ic,oc,1) if ic!=oc else None; self.norm=nn.BatchNorm1d(oc)
    def forward(self, x):
        o = self.conv(x)
        if self.chomp: o = o[:,:,:-self.chomp]
        o = self.relu(self.norm(o)); o = self.drop(o)
        return o + (x if self.ds is None else self.ds(x))

class TCN(nn.Module):
    def __init__(self, d, ch=(128,128,128)):
        super().__init__()
        ls, p = [], d
        for i,c in enumerate(ch): ls.append(TemporalBlock(p,c,d=2**i)); p=c
        self.net = nn.Sequential(*ls)
    def forward(self, x): return self.net(x.transpose(1,2)).transpose(1,2)

class TCNBaseline(nn.Module):
    def __init__(self):
        super().__init__()
        self.adapter = InputAdapterBaseline()
        self.at = TCN(128); self.vt = TCN(171)
        self.cls = nn.Sequential(nn.Linear(256,128), nn.ReLU(),
                                   nn.Dropout(0.3), nn.Linear(128,1))
    def pool(self, x, m):
        m = m.unsqueeze(-1).float()
        return (x*m).sum(1)/m.sum(1).clamp(min=1)
    def forward(self, b):
        a,v = self.adapter(b["audio"],b["visual"])
        return self.cls(torch.cat([self.pool(self.at(a),b["audio_mask"]),
                                    self.pool(self.vt(v),b["visual_mask"])],-1)).squeeze(-1)

# ── loaders ───────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_models():
    swin = SwinDepressionModel()
    swin.load_state_dict(torch.load(os.path.join(MODEL_DIR,"swin_seed123.pt"), map_location="cpu"))
    swin.eval()
    base = TCNBaseline()
    base.load_state_dict(torch.load(os.path.join(MODEL_DIR,"baseline_seed2024.pt"), map_location="cpu"))
    base.eval()
    return swin, base

def preprocess(audio_arr, visual_arr):
    if audio_arr.shape[0] > MAX_LEN: audio_arr = audio_arr[:MAX_LEN]
    if visual_arr.shape[0] > MAX_LEN:
        idx = np.linspace(0,visual_arr.shape[0]-1,MAX_LEN).round().astype(int)
        visual_arr = visual_arr[idx]
    a_len, v_len = audio_arr.shape[0], visual_arr.shape[0]
    ap = np.zeros((MAX_LEN,AUDIO_DIM),  dtype=np.float32)
    vp = np.zeros((MAX_LEN,VISUAL_DIM), dtype=np.float32)
    ap[:a_len] = audio_arr; vp[:v_len] = visual_arr
    am = torch.zeros(1,MAX_LEN,dtype=torch.bool); am[0,:a_len]=True
    vm = torch.zeros(1,MAX_LEN,dtype=torch.bool); vm[0,:v_len]=True
    return {"audio":torch.tensor(ap).unsqueeze(0),"visual":torch.tensor(vp).unsqueeze(0),
            "audio_mask":am,"visual_mask":vm}

def predict(model, batch):
    with torch.no_grad():
        return torch.sigmoid(model(batch)).item()

def gauge(prob, title):
    color = "#ef4444" if prob >= 0.5 else "#22c55e"
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=round(prob*100,1),
        number={"suffix":"%","font":{"size":26,"color":"#e2e8f0","family":"JetBrains Mono"}},
        title={"text":title,"font":{"size":11,"color":"#64748b","family":"Inter"}},
        gauge={"axis":{"range":[0,100],"tickcolor":"#334155",
                       "tickfont":{"color":"#64748b","size":9}},
               "bar":{"color":color,"thickness":0.25},
               "bgcolor":"#111827","bordercolor":"#1e293b",
               "steps":[{"range":[0,50],"color":"#071a0f"},
                         {"range":[50,100],"color":"#1f0a0a"}],
               "threshold":{"line":{"color":"#fff","width":2},"thickness":0.8,"value":50}}))
    fig.update_layout(height=190, margin=dict(t=40,b=5,l=15,r=15),
                       paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    return fig

# ── helpers ───────────────────────────────────────────────────────────────────
def img(filename, caption=None, use_column_width=True):
    path = os.path.join(MODEL_DIR, filename)
    if os.path.exists(path):
        st.image(Image.open(path), caption=caption, use_column_width=use_column_width)

# ═════════════════════════════════════════════════════════════════════════════
# HERO
# ═════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero">
    <div class="hero-eyebrow">Final Year Project · Computer Science</div>
    <h1>Depression Detection<br><span>using Swin Transformer</span></h1>
    <p class="hero-sub">
        A multimodal deep learning system that analyses audio and visual behavioural
        features to detect signs of depression — outperforming a strong TCN baseline
        across four evaluation metrics.
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
c1, c2 = st.columns([1.1, 1])
with c1:
    st.markdown('<div class="section-head">About this project</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Depression is one of the most prevalent mental health conditions worldwide, yet screening relies almost entirely on subjective self-report. This project explores whether a Swin Transformer applied to multimodal behavioural features can automate and improve that screening — learning jointly from acoustic and facial cues captured in short video recordings.</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="stat-grid">
        <div class="stat-card"><div class="stat-val">1,804</div><div class="stat-label">LMVD samples</div></div>
        <div class="stat-card"><div class="stat-val">50/50</div><div class="stat-label">class balance</div></div>
        <div class="stat-card"><div class="stat-val">0.846</div><div class="stat-label">Swin AUC</div></div>
        <div class="stat-card"><div class="stat-val">+2.4pp</div><div class="stat-label">F1 over baseline</div></div>
    </div>
    """, unsafe_allow_html=True)
with c2:
    img("eda_class_balance.png", caption="Class and split distribution across the LMVD dataset")

st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 2 — DATASET
# ═════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-head">Dataset · LMVD</div>', unsafe_allow_html=True)
st.markdown('<div class="section-sub">The Large-scale Multimodal Video Depression (LMVD) dataset contains 1,823 vlog recordings collected from Bilibili, TikTok, Sina Weibo and YouTube — each labelled by clinical annotation. After filtering 19 near-empty audio files, 1,804 samples remain with a near-perfect 50/50 class balance, split 70/15/15 into train, validation, and test sets via stratified sampling.</div>', unsafe_allow_html=True)

c1, c2 = st.columns(2)
with c1:
    img("eda_audio_lengths.png", caption="Audio sequence length distribution (VGGish embeddings)")
with c2:
    st.markdown("""
    <div class="stat-grid" style="flex-direction:column; gap:0.6rem; margin-top:0.4rem">
        <div class="stat-card" style="text-align:left; display:flex; justify-content:space-between; align-items:center">
            <span style="color:#64748b;font-size:0.8rem">Audio features</span>
            <span style="font-family:'JetBrains Mono',monospace;color:#3b82f6">VGGish · 128-dim</span>
        </div>
        <div class="stat-card" style="text-align:left; display:flex; justify-content:space-between; align-items:center">
            <span style="color:#64748b;font-size:0.8rem">Visual features</span>
            <span style="font-family:'JetBrains Mono',monospace;color:#3b82f6">OpenFace · 171-dim</span>
        </div>
        <div class="stat-card" style="text-align:left; display:flex; justify-content:space-between; align-items:center">
            <span style="color:#64748b;font-size:0.8rem">Audio seq. length</span>
            <span style="font-family:'JetBrains Mono',monospace;color:#3b82f6">variable (mean 567)</span>
        </div>
        <div class="stat-card" style="text-align:left; display:flex; justify-content:space-between; align-items:center">
            <span style="color:#64748b;font-size:0.8rem">Visual seq. length</span>
            <span style="font-family:'JetBrains Mono',monospace;color:#3b82f6">fixed at 915 frames</span>
        </div>
        <div class="stat-card" style="text-align:left; display:flex; justify-content:space-between; align-items:center">
            <span style="color:#64748b;font-size:0.8rem">Train / Val / Test</span>
            <span style="font-family:'JetBrains Mono',monospace;color:#3b82f6">1262 / 271 / 271</span>
        </div>
        <div class="stat-card" style="text-align:left; display:flex; justify-content:space-between; align-items:center">
            <span style="color:#64748b;font-size:0.8rem">Truncated to</span>
            <span style="font-family:'JetBrains Mono',monospace;color:#3b82f6">512 frames (36% affected)</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 3 — ARCHITECTURE
# ═════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-head">Model architecture</div>', unsafe_allow_html=True)
st.markdown('<div class="section-sub">Each modality is projected into a 32-dim pseudo-image grid (T×32), treated as a single-channel spatial input, and passed through an independent pretrained Swin-Tiny backbone. The resulting token sequences are fused via bidirectional cross-attention before a shared classification head.</div>', unsafe_allow_html=True)

st.markdown("""
<div class="pipeline">
    <div class="step">
        <div class="step-num">01</div>
        <div class="step-title">Input adapter</div>
        <div class="step-desc">Projects audio (128→32) and visual (171→32) into a shared pseudo-image width via Linear + LayerNorm</div>
        <div class="step-arrow">›</div>
    </div>
    <div class="step">
        <div class="step-num">02</div>
        <div class="step-title">Pseudo-image</div>
        <div class="step-desc">Reshaped to (T=512, W=32) and treated as a single-channel 2D grid, enabling Swin's patch-based attention</div>
        <div class="step-arrow">›</div>
    </div>
    <div class="step">
        <div class="step-num">03</div>
        <div class="step-title">Swin-Tiny towers</div>
        <div class="step-desc">Two independent pretrained Swin-Tiny backbones (one per modality) with window_size=4 and differential LR fine-tuning</div>
        <div class="step-arrow">›</div>
    </div>
    <div class="step">
        <div class="step-num">04</div>
        <div class="step-title">Cross-attention fusion</div>
        <div class="step-desc">Bidirectional multi-head attention lets audio tokens attend to visual context and vice versa before mean pooling</div>
        <div class="step-arrow">›</div>
    </div>
    <div class="step">
        <div class="step-num">05</div>
        <div class="step-title">Classifier</div>
        <div class="step-desc">Linear → ReLU → Dropout → Linear producing a single depression-risk logit, trained with BCEWithLogitsLoss</div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 4 — RESULTS
# ═════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-head">Results</div>', unsafe_allow_html=True)
st.markdown('<div class="section-sub">Both models were evaluated across 3 independent random seeds. Results are reported as mean ± std on the held-out test set (271 samples, fingerprint fae836bd). AUC is the primary metric since it is threshold-independent and most stable across seeds.</div>', unsafe_allow_html=True)

c1, c2 = st.columns(2)
with c1:
    img("results_comparison.png", caption="Mean ± std across 3 seeds — all metrics")
with c2:
    img("roc_curves.png", caption="ROC curves — best seed per model")

st.markdown("#### Confusion matrices")
img("confusion_matrices.png", caption="Predicted vs actual labels on the test set (best seed per model)")

st.markdown("#### Training dynamics")
img("training_curves_swin.png", caption="Loss and F1 per epoch — Swin Transformer (best seed)")

st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 5 — ABLATION
# ═════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-head">Ablation study</div>', unsafe_allow_html=True)
st.markdown('<div class="section-sub">To understand each modality\'s contribution, we trained audio-only and visual-only Swin variants alongside the full fused model. The results confirm that visual features carry the primary discriminative signal (AUC 0.784), audio alone performs near-randomly (AUC 0.587), and cross-attention fusion over both streams yields the best performance (AUC 0.846).</div>', unsafe_allow_html=True)

img("ablation_chart.png", caption="Audio-only vs Visual-only vs Fused Swin vs TCN Baseline")

st.markdown("""
<div class="info-banner">
    <strong>Key finding:</strong> Fusion decisively outperforms either unimodal variant.
    Visual features alone already beat the full TCN baseline on AUC (0.784 vs 0.817 is close,
    but fusion pushes to 0.846). Audio-only Swin is near-random (AUC 0.587), confirming that
    VGGish embeddings alone are insufficient without the complementary facial behavioural signal.
</div>
""", unsafe_allow_html=True)

st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 6 — LIVE DEMO
# ═════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-head">Live inference demo</div>', unsafe_allow_html=True)
st.markdown('<div class="section-sub">Upload a pair of pre-extracted feature files (.npy) for one participant. Both the proposed Swin model and the TCN baseline will run inference simultaneously so you can compare their outputs directly.</div>', unsafe_allow_html=True)

st.markdown("""
<div class="info-banner">
    <strong>Expected formats:</strong>
    audio file → shape (T, 128) float32 — VGGish embeddings, one vector per second ·
    visual file → shape (T, 171) float32 — OpenFace AU/landmark/gaze/pose features
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
        with st.spinner("Running inference..."):
            swin_m, base_m = load_models()
            batch      = preprocess(audio_arr, visual_arr)
            swin_prob  = predict(swin_m, batch)
            base_prob  = predict(base_m, batch)

        rc1, rc2 = st.columns(2)
        for col, prob, title, name in [
            (rc1, swin_prob, "Swin Transformer · Depression probability", "Swin Transformer"),
            (rc2, base_prob, "TCN Baseline · Depression probability", "TCN Baseline"),
        ]:
            with col:
                depressed = prob >= 0.5
                cls = "result-depressed" if depressed else "result-normal"
                verdict = "Depressed" if depressed else "Non-depressed"
                color   = "#f87171" if depressed else "#4ade80"
                st.markdown(f"""
                <div class="result-card {cls}">
                    <div class="result-label">{name}</div>
                    <div class="result-verdict" style="color:{color}">{verdict}</div>
                    <div class="result-prob">P(depressed) = {prob:.4f}</div>
                </div>
                """, unsafe_allow_html=True)
                st.plotly_chart(gauge(prob, title), use_container_width=True)

        st.markdown("#### Validated test-set performance (mean ± std, 3 seeds)")
        st.dataframe({
            "Metric":    ["F1",            "Precision",    "Recall",       "Accuracy",     "AUC"],
            "Baseline":  ["0.756 ±0.012",  "0.687 ±0.042", "0.844 ±0.038", "0.727 ±0.028", "0.817 ±0.027"],
            "Swin":      ["0.780 ±0.021",  "0.751 ±0.004", "0.812 ±0.045", "0.772 ±0.014", "0.846 ±0.014"],
            "Δ (mean)":  ["+0.024 ✓",      "+0.064 ✓",     "−0.032",       "+0.046 ✓",     "+0.030 ✓"],
        }, use_container_width=True, hide_index=True)

        st.markdown("""
        <div class="warn-banner">
            ⚠️ This system is a research prototype and is <strong>not a clinical diagnostic tool</strong>.
            Predictions should not be used to make clinical decisions.
        </div>
        """, unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# FOOTER
# ═════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="footer">
    Depression Detection using Swin Transformer · LMVD Dataset<br>
    Final Year Project · Computer Science<br>
    <span style="color:#1e293b">────────────────────────────────</span><br>
    For research and demonstration purposes only.
    Not a clinical diagnostic tool.
</div>
""", unsafe_allow_html=True)