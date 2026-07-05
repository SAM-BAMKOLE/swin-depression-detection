import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import json
import os
import plotly.graph_objects as go
import timm

# ── page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Depression Detection · LMVD",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp { background: #0f1117; color: #e8eaf0; }

.hero {
    text-align: center;
    padding: 2.5rem 0 1.5rem;
    border-bottom: 1px solid #1e2130;
    margin-bottom: 2rem;
}
.hero h1 {
    font-size: 1.8rem;
    font-weight: 600;
    letter-spacing: -0.02em;
    color: #ffffff;
    margin: 0 0 0.4rem;
}
.hero p {
    color: #6b7280;
    font-size: 0.9rem;
    font-weight: 300;
    margin: 0;
}

.upload-card {
    background: #161b27;
    border: 1px solid #1e2130;
    border-radius: 10px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
}
.upload-label {
    font-size: 0.78rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #6b7280;
    margin-bottom: 0.5rem;
}

.result-card {
    border-radius: 12px;
    padding: 2rem;
    text-align: center;
    margin: 1.5rem 0;
}
.result-depressed {
    background: linear-gradient(135deg, #1f0f0f 0%, #2d1111 100%);
    border: 1px solid #7f1d1d;
}
.result-normal {
    background: linear-gradient(135deg, #0a1f14 0%, #0f2d1a 100%);
    border: 1px solid #14532d;
}
.result-label {
    font-size: 0.75rem;
    font-weight: 500;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}
.result-verdict {
    font-size: 2rem;
    font-weight: 600;
    margin: 0.2rem 0 0.6rem;
    letter-spacing: -0.02em;
}
.result-prob {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1rem;
    color: #9ca3af;
}

.metric-row {
    display: flex;
    gap: 0.75rem;
    margin-top: 1rem;
}
.metric-box {
    flex: 1;
    background: #161b27;
    border: 1px solid #1e2130;
    border-radius: 8px;
    padding: 0.9rem;
    text-align: center;
}
.metric-val {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.25rem;
    font-weight: 500;
    color: #e8eaf0;
}
.metric-name {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #6b7280;
    margin-top: 0.2rem;
}

.info-banner {
    background: #161b27;
    border: 1px solid #1e2130;
    border-left: 3px solid #3b82f6;
    border-radius: 6px;
    padding: 0.8rem 1rem;
    font-size: 0.82rem;
    color: #9ca3af;
    margin-bottom: 1.5rem;
}
.footer {
    text-align: center;
    padding: 2rem 0 1rem;
    color: #374151;
    font-size: 0.75rem;
    border-top: 1px solid #1e2130;
    margin-top: 3rem;
}

/* override streamlit widget styling */
div[data-testid="stFileUploader"] {
    background: transparent;
}
section[data-testid="stFileUploadDropzone"] {
    background: #0f1117 !important;
    border: 1.5px dashed #2d3748 !important;
    border-radius: 8px !important;
}
.stButton > button {
    width: 100%;
    background: #2563eb;
    color: white;
    border: none;
    border-radius: 8px;
    padding: 0.65rem 1rem;
    font-size: 0.9rem;
    font-weight: 500;
    font-family: 'Inter', sans-serif;
    cursor: pointer;
    transition: background 0.2s;
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


# ── model definitions (must match training exactly) ───────────────────────────
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


# ── model loading ─────────────────────────────────────────────────────────────
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


# ── inference ─────────────────────────────────────────────────────────────────
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
        "audio":        torch.tensor(audio_padded).unsqueeze(0),
        "visual":       torch.tensor(visual_padded).unsqueeze(0),
        "audio_mask":   audio_mask,
        "visual_mask":  visual_mask,
    }

def predict(model, batch):
    with torch.no_grad():
        logit = model(batch)
        prob = torch.sigmoid(logit).item()
    return prob


# ── gauge chart ───────────────────────────────────────────────────────────────
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
            "bgcolor": "#161b27",
            "bordercolor": "#1e2130",
            "steps": [
                {"range": [0, 50], "color": "#0a1f14"},
                {"range": [50, 100], "color": "#1f0f0f"},
            ],
            "threshold": {
                "line": {"color": "#ffffff", "width": 2},
                "thickness": 0.8,
                "value": 50,
            },
        }
    ))
    fig.update_layout(
        height=200, margin=dict(t=40, b=10, l=20, r=20),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#e8eaf0",
    )
    return fig


# ── UI ────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>🧠 Depression Detection</h1>
    <p>Swin Transformer · LMVD Dataset · Final Year Project</p>
</div>
""", unsafe_allow_html=True)

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
    audio_arr = np.load(audio_file).astype(np.float32)
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
            <div class="metric-box">
                <div class="metric-val">{audio_arr.shape[0]}</div>
                <div class="metric-name">audio frames</div>
            </div>
            <div class="metric-box">
                <div class="metric-val">{audio_arr.shape[1]}</div>
                <div class="metric-name">audio dim</div>
            </div>
            <div class="metric-box">
                <div class="metric-val">{visual_arr.shape[0]}</div>
                <div class="metric-name">visual frames</div>
            </div>
            <div class="metric-box">
                <div class="metric-val">{visual_arr.shape[1]}</div>
                <div class="metric-name">visual dim</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    if ok and run_btn:
        with st.spinner("Loading models and running inference..."):
            swin_model, baseline_model = load_models()
            batch = preprocess(audio_arr, visual_arr)
            swin_prob = predict(swin_model, batch)
            base_prob = predict(baseline_model, batch)

        swin_depressed = swin_prob >= 0.5
        base_depressed = base_prob >= 0.5

        st.markdown("### Results")

        tab1, tab2 = st.tabs(["Swin Transformer (proposed)", "TCN Baseline"])

        with tab1:
            card_cls = "result-depressed" if swin_depressed else "result-normal"
            verdict = "Depressed" if swin_depressed else "Non-depressed"
            verdict_color = "#f87171" if swin_depressed else "#4ade80"
            st.markdown(f"""
            <div class="result-card {card_cls}">
                <div class="result-label">Swin Transformer · Prediction</div>
                <div class="result-verdict" style="color:{verdict_color}">{verdict}</div>
                <div class="result-prob">P(depressed) = {swin_prob:.4f}</div>
            </div>
            """, unsafe_allow_html=True)
            st.plotly_chart(make_gauge(swin_prob, "Depression probability"), use_container_width=True)

        with tab2:
            card_cls = "result-depressed" if base_depressed else "result-normal"
            verdict = "Depressed" if base_depressed else "Non-depressed"
            verdict_color = "#f87171" if base_depressed else "#4ade80"
            st.markdown(f"""
            <div class="result-card {card_cls}">
                <div class="result-label">TCN Baseline · Prediction</div>
                <div class="result-verdict" style="color:{verdict_color}">{verdict}</div>
                <div class="result-prob">P(depressed) = {base_prob:.4f}</div>
            </div>
            """, unsafe_allow_html=True)
            st.plotly_chart(make_gauge(base_prob, "Depression probability"), use_container_width=True)

        st.markdown("#### Model comparison (test set, 3 seeds)")
        metrics_data = {
            "Metric":    ["F1",     "Precision", "Recall",  "Accuracy", "AUC"],
            "Baseline":  ["0.756",  "0.687",     "0.844",   "0.727",    "0.817"],
            "Swin":      ["0.780",  "0.751",     "0.812",   "0.772",    "0.846"],
            "Δ":         ["+0.024", "+0.064",    "−0.032",  "+0.046",   "+0.030"],
        }
        st.dataframe(metrics_data, use_container_width=True, hide_index=True)

st.markdown("""
<div class="footer">
    Depression Detection using Swin Transformer · LMVD Dataset · Final Year Project<br>
    For research and demonstration purposes only. Not a clinical diagnostic tool.
</div>
""", unsafe_allow_html=True)
