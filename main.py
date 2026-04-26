import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.datasets import make_moons, make_circles
from sklearn.preprocessing import StandardScaler

# ─── Force CPU ───────────────────────────────────────────────────────────────────
device = torch.device("cpu")
torch.set_num_threads(4)

# ─── Page Config ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Neural Network Playground",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=Sora:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Sora', sans-serif;
        background-color: #08080f;
    }

    .main-title {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 2.2rem;
        font-weight: 600;
        background: linear-gradient(135deg, #a78bfa 0%, #60a5fa 50%, #34d399 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        letter-spacing: -1px;
        margin-bottom: 0.2rem;
    }

    .sub-title {
        text-align: center;
        color: #6b7280;
        font-size: 0.85rem;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin-bottom: 2rem;
        font-weight: 300;
    }

    .metric-card {
        background: linear-gradient(135deg, #0f0f1f 0%, #12122a 100%);
        border: 1px solid #a78bfa33;
        border-radius: 14px;
        padding: 1rem 1.2rem;
        margin: 0.4rem 0;
    }

    .metric-label {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.65rem;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        margin-bottom: 4px;
    }

    .metric-value {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 1.6rem;
        font-weight: 600;
        color: #a78bfa;
    }

    .metric-sub {
        font-size: 0.75rem;
        color: #9ca3af;
        margin-top: 2px;
    }

    .section-header {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.72rem;
        color: #a78bfa;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        border-bottom: 1px solid #a78bfa33;
        padding-bottom: 6px;
        margin: 1.4rem 0 0.9rem 0;
    }

    .stButton>button {
        background: linear-gradient(135deg, #7c3aed, #2563eb);
        color: #ffffff;
        font-family: 'IBM Plex Mono', monospace;
        font-weight: 600;
        font-size: 0.82rem;
        border: none;
        border-radius: 10px;
        padding: 0.65rem 1.2rem;
        width: 100%;
        letter-spacing: 0.04em;
        transition: all 0.2s;
    }

    .stButton>button:hover {
        background: linear-gradient(135deg, #6d28d9, #1d4ed8);
        transform: translateY(-1px);
        box-shadow: 0 4px 20px rgba(124,58,237,0.4);
    }

    .info-box {
        background: #0f0f1f;
        border: 1px solid #a78bfa33;
        border-radius: 10px;
        padding: 0.9rem 1rem;
        font-size: 0.8rem;
        color: #9ca3af;
        margin: 0.5rem 0;
    }

    .chip {
        display: inline-block;
        background: #a78bfa22;
        color: #a78bfa;
        border: 1px solid #a78bfa44;
        border-radius: 20px;
        padding: 2px 10px;
        font-size: 0.7rem;
        font-family: 'IBM Plex Mono', monospace;
        margin: 2px;
    }

    .chip-green {
        display: inline-block;
        background: #34d39922;
        color: #34d399;
        border: 1px solid #34d39944;
        border-radius: 20px;
        padding: 2px 10px;
        font-size: 0.7rem;
        font-family: 'IBM Plex Mono', monospace;
        margin: 2px;
    }

    .stTabs [data-baseweb="tab"] {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.75rem;
        color: #6b7280;
    }

    .stTabs [aria-selected="true"] {
        color: #a78bfa !important;
    }

    .stSelectbox label, .stSlider label, .stMultiselect label, .stRadio label, .stToggle label {
        font-size: 0.78rem !important;
        color: #9ca3af !important;
        font-family: 'IBM Plex Mono', monospace !important;
        letter-spacing: 0.05em !important;
    }

    .layer-row {
        background: #0f0f1f;
        border: 1px solid #a78bfa22;
        border-radius: 8px;
        padding: 8px 12px;
        margin: 4px 0;
        font-size: 0.78rem;
        color: #d1d5db;
        font-family: 'IBM Plex Mono', monospace;
    }

    .status-trained {
        display: inline-block;
        background: #34d39920;
        color: #34d399;
        border: 1px solid #34d39955;
        border-radius: 6px;
        padding: 3px 10px;
        font-size: 0.72rem;
        font-family: 'IBM Plex Mono', monospace;
    }

    .status-untrained {
        display: inline-block;
        background: #f4724820;
        color: #f47248;
        border: 1px solid #f4724855;
        border-radius: 6px;
        padding: 3px 10px;
        font-size: 0.72rem;
        font-family: 'IBM Plex Mono', monospace;
    }

    div[data-testid="stSidebar"] {
        background: #0a0a18;
        border-right: 1px solid #a78bfa22;
    }
</style>
""", unsafe_allow_html=True)

# ─── Title ───────────────────────────────────────────────────────────────────────
st.markdown('<h1 class="main-title">🧠 Neural Network Playground</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Architecture Design · Live Training · Decision Boundaries · Weight Analysis</p>', unsafe_allow_html=True)

# ─── Session State Init ───────────────────────────────────────────────────────────
if "model" not in st.session_state:
    st.session_state.model = None
if "losses" not in st.session_state:
    st.session_state.losses = []
if "accuracies" not in st.session_state:
    st.session_state.accuracies = []
if "trained" not in st.session_state:
    st.session_state.trained = False
if "X" not in st.session_state:
    st.session_state.X = None
if "y" not in st.session_state:
    st.session_state.y = None

# ─── Dataset Generator ───────────────────────────────────────────────────────────
def generate_dataset(dataset_type, n_samples, noise):
    np.random.seed(42)
    if dataset_type == "🌙 Two Moons":
        X, y = make_moons(n_samples=n_samples, noise=noise, random_state=42)
    elif dataset_type == "⭕ Circles":
        X, y = make_circles(n_samples=n_samples, noise=noise, factor=0.45, random_state=42)
    elif dataset_type == "⊕ XOR":
        X = np.random.randn(n_samples, 2) * 1.5
        y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(int)
        X += np.random.randn(*X.shape) * noise * 0.8
    elif dataset_type == "🌀 Spiral":
        N = n_samples // 2
        theta = np.linspace(0, 4 * np.pi, N)
        r = np.linspace(0.2, 1.0, N)
        X0 = np.c_[r * np.cos(theta), r * np.sin(theta)]
        X1 = np.c_[r * np.cos(theta + np.pi), r * np.sin(theta + np.pi)]
        X0 += np.random.randn(N, 2) * noise * 0.25
        X1 += np.random.randn(N, 2) * noise * 0.25
        X = np.vstack([X0, X1])
        y = np.hstack([np.zeros(N), np.ones(N)]).astype(int)
    elif dataset_type == "🔵 Gaussian Blobs":
        X = np.vstack([
            np.random.randn(n_samples // 2, 2) + [1.8, 1.8],
            np.random.randn(n_samples // 2, 2) + [-1.8, -1.8]
        ])
        X += np.random.randn(*X.shape) * noise * 0.5
        y = np.hstack([np.zeros(n_samples // 2), np.ones(n_samples // 2)]).astype(int)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X.astype(np.float32), y.astype(np.int64)

# ─── Model Builder ────────────────────────────────────────────────────────────────
ACTIVATIONS = {
    "ReLU": nn.ReLU,
    "Tanh": nn.Tanh,
    "Sigmoid": nn.Sigmoid,
    "LeakyReLU": lambda: nn.LeakyReLU(0.1),
    "ELU": nn.ELU,
    "GELU": nn.GELU,
}

def build_model(hidden_sizes, activations):
    layers = []
    sizes = [2] + hidden_sizes + [2]
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            act_fn = ACTIVATIONS.get(activations[i], nn.ReLU)
            layers.append(act_fn())
    return nn.Sequential(*layers)

# ─── Training ────────────────────────────────────────────────────────────────────
def train_model(model, X, y, optimizer_name, lr, batch_size, epochs, progress_bar, status_text):
    X_t = torch.FloatTensor(X)
    y_t = torch.LongTensor(y)
    n = len(X)

    if optimizer_name == "Adam":
        opt = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    elif optimizer_name == "SGD + Momentum":
        opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    elif optimizer_name == "RMSProp":
        opt = optim.RMSprop(model.parameters(), lr=lr, weight_decay=1e-4)
    elif optimizer_name == "AdaGrad":
        opt = optim.Adagrad(model.parameters(), lr=lr, weight_decay=1e-4)

    criterion = nn.CrossEntropyLoss()
    losses, accuracies = [], []
    update_every = max(1, epochs // 30)

    for epoch in range(epochs):
        model.train()
        idx = torch.randperm(n)
        epoch_loss = 0.0

        for start in range(0, n, batch_size):
            b = idx[start:start + batch_size]
            Xb, yb = X_t[b], y_t[b]
            opt.zero_grad()
            out = model(Xb)
            loss = criterion(out, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            opt.step()
            epoch_loss += loss.item() * len(b)

        epoch_loss /= n
        model.eval()
        with torch.no_grad():
            preds = model(X_t).argmax(dim=1)
            acc = (preds == y_t).float().mean().item() * 100

        losses.append(epoch_loss)
        accuracies.append(acc)

        if epoch % update_every == 0 or epoch == epochs - 1:
            progress_bar.progress((epoch + 1) / epochs)
            status_text.markdown(
                f'<div style="font-family:IBM Plex Mono;font-size:0.78rem;color:#a78bfa;">'
                f'Epoch {epoch+1}/{epochs} &nbsp;·&nbsp; '
                f'<span style="color:#f472b6">Loss: {epoch_loss:.4f}</span> &nbsp;·&nbsp; '
                f'<span style="color:#34d399">Acc: {acc:.1f}%</span></div>',
                unsafe_allow_html=True
            )

    return losses, accuracies

# ─── Plots ───────────────────────────────────────────────────────────────────────
def plot_dataset(X, y, title="Dataset"):
    colors = np.where(y == 0, "#f472b6", "#34d399")
    fig = go.Figure()
    for cls, color, name in [(0, "#f472b6", "Class 0"), (1, "#34d399", "Class 1")]:
        mask = y == cls
        fig.add_trace(go.Scatter(
            x=X[mask, 0], y=X[mask, 1], mode="markers", name=name,
            marker=dict(color=color, size=6, opacity=0.85,
                        line=dict(width=0.8, color="white"))
        ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(8,8,20,0.9)",
        font=dict(color="white", family="Sora"),
        height=310,
        margin=dict(l=10, r=10, t=35, b=10),
        legend=dict(font=dict(size=10), bgcolor="rgba(0,0,0,0)"),
        title=dict(text=title, font=dict(color="#a78bfa", size=12, family="IBM Plex Mono")),
        xaxis=dict(showgrid=True, gridcolor="#1f1f3a", zeroline=False),
        yaxis=dict(showgrid=True, gridcolor="#1f1f3a", zeroline=False),
    )
    return fig


def plot_decision_boundary(model, X, y):
    h = 0.04
    pad = 0.6
    x1_min, x1_max = X[:, 0].min() - pad, X[:, 0].max() + pad
    x2_min, x2_max = X[:, 1].min() - pad, X[:, 1].max() + pad

    xx, yy = np.meshgrid(np.arange(x1_min, x1_max, h),
                          np.arange(x2_min, x2_max, h))
    grid = torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()])

    model.eval()
    with torch.no_grad():
        Z = torch.softmax(model(grid), dim=1)[:, 1].numpy()
    Z = Z.reshape(xx.shape)

    fig = go.Figure()
    fig.add_trace(go.Contour(
        x=np.arange(x1_min, x1_max, h),
        y=np.arange(x2_min, x2_max, h),
        z=Z,
        colorscale=[
            [0.0,  "#be185d"],
            [0.45, "#581c87"],
            [0.5,  "#1e1b4b"],
            [0.55, "#164e63"],
            [1.0,  "#065f46"]
        ],
        showscale=True,
        colorbar=dict(tickfont=dict(color="white", size=9), len=0.8),
        contours=dict(showlines=True, coloring="fill",
                      start=0, end=1, size=0.1,
                      showlabels=False),
        opacity=0.75,
        name="P(Class 1)"
    ))
    # Decision boundary line at 0.5
    fig.add_trace(go.Contour(
        x=np.arange(x1_min, x1_max, h),
        y=np.arange(x2_min, x2_max, h),
        z=Z,
        colorscale=[[0, "white"], [1, "white"]],
        showscale=False,
        contours=dict(showlines=True, coloring="lines",
                      start=0.5, end=0.5, size=0.1),
        line=dict(width=2, color="white"),
        opacity=0.9,
        name="Boundary"
    ))
    # Data points
    for cls, color, name in [(0, "#f472b6", "Class 0"), (1, "#34d399", "Class 1")]:
        mask = y == cls
        fig.add_trace(go.Scatter(
            x=X[mask, 0], y=X[mask, 1], mode="markers", name=name,
            marker=dict(color=color, size=6, opacity=0.9,
                        line=dict(width=1, color="white"))
        ))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(8,8,20,0.9)",
        font=dict(color="white", family="Sora"),
        height=400,
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(font=dict(size=10), bgcolor="rgba(0,0,0,0)"),
        title=dict(text="Decision Boundary  (white line = 50% confidence)",
                   font=dict(color="#a78bfa", size=12, family="IBM Plex Mono")),
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(showgrid=False, zeroline=False),
    )
    return fig


def plot_training_curves(losses, accuracies):
    epochs_x = list(range(1, len(losses) + 1))
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Training Loss", "Training Accuracy (%)"),
        horizontal_spacing=0.12
    )
    fig.add_trace(go.Scatter(
        x=epochs_x, y=losses,
        line=dict(color="#f472b6", width=2.5, shape="spline"),
        fill="tozeroy", fillcolor="rgba(244,114,182,0.08)",
        name="Loss"
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=epochs_x, y=accuracies,
        line=dict(color="#34d399", width=2.5, shape="spline"),
        fill="tozeroy", fillcolor="rgba(52,211,153,0.08)",
        name="Accuracy"
    ), row=1, col=2)
    # Final value annotations
    if losses:
        fig.add_annotation(x=epochs_x[-1], y=losses[-1],
            text=f" {losses[-1]:.4f}", showarrow=False,
            font=dict(color="#f472b6", size=10), xanchor="left")
        fig.add_annotation(x=epochs_x[-1], y=accuracies[-1],
            text=f" {accuracies[-1]:.1f}%", showarrow=False,
            font=dict(color="#34d399", size=10), xanchor="left", row=1, col=2)

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(8,8,20,0.9)",
        font=dict(color="white", family="Sora"),
        height=280,
        showlegend=False,
        margin=dict(l=10, r=50, t=45, b=20),
    )
    for col in [1, 2]:
        fig.update_xaxes(showgrid=True, gridcolor="#1f1f3a", title_text="Epoch",
                          title_font=dict(size=10), row=1, col=col)
        fig.update_yaxes(showgrid=True, gridcolor="#1f1f3a", row=1, col=col)
    fig.update_annotations(font=dict(color="#a78bfa", size=11))
    return fig


def plot_weight_histograms(model):
    fig = go.Figure()
    palette = ["#a78bfa", "#34d399", "#f472b6", "#60a5fa", "#fbbf24", "#f87171"]
    layer_idx = 0
    for name, param in model.named_parameters():
        if "weight" in name:
            w = param.detach().numpy().flatten()
            fig.add_trace(go.Histogram(
                x=w, name=f"Layer {layer_idx + 1}",
                nbinsx=40, opacity=0.75,
                marker_color=palette[layer_idx % len(palette)]
            ))
            layer_idx += 1
    fig.update_layout(
        barmode="overlay",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(8,8,20,0.9)",
        font=dict(color="white", family="Sora"),
        height=260,
        margin=dict(l=10, r=10, t=40, b=20),
        legend=dict(font=dict(size=10), bgcolor="rgba(0,0,0,0)"),
        title=dict(text="Weight Distributions per Layer",
                   font=dict(color="#a78bfa", size=12, family="IBM Plex Mono")),
        xaxis=dict(showgrid=True, gridcolor="#1f1f3a", title_text="Weight Value"),
        yaxis=dict(showgrid=True, gridcolor="#1f1f3a", title_text="Count"),
    )
    return fig


def plot_architecture_diagram(hidden_sizes, activations):
    sizes = [2] + hidden_sizes + [2]
    n_layers = len(sizes)
    fig = go.Figure()

    xs = np.linspace(0.08, 0.92, n_layers)
    palette = ["#60a5fa"] + ["#a78bfa"] * (n_layers - 2) + ["#34d399"]
    MAX_SHOW = 7

    # Draw edges first (background)
    for l in range(n_layers - 1):
        n1 = min(sizes[l], MAX_SHOW)
        n2 = min(sizes[l + 1], MAX_SHOW)
        y1s = np.linspace(0.15, 0.85, n1)
        y2s = np.linspace(0.15, 0.85, n2)
        for y1 in y1s:
            for y2 in y2s:
                fig.add_shape(type="line",
                    x0=xs[l], y0=y1, x1=xs[l + 1], y1=y2,
                    line=dict(color="rgba(167,139,250,0.12)", width=0.8))

    # Draw nodes
    for l in range(n_layers):
        n_show = min(sizes[l], MAX_SHOW)
        ys = np.linspace(0.15, 0.85, n_show)
        for y in ys:
            fig.add_shape(type="circle",
                x0=xs[l] - 0.022, y0=y - 0.038,
                x1=xs[l] + 0.022, y1=y + 0.038,
                fillcolor=palette[l],
                line=dict(color="#ffffff", width=1.2),
                opacity=0.85)
        if sizes[l] > MAX_SHOW:
            fig.add_annotation(x=xs[l], y=0.06,
                text=f"···{sizes[l]}···", showarrow=False,
                font=dict(color="#9ca3af", size=8))

        # Layer label
        if l == 0:
            label = "Input<br>(2)"
        elif l == n_layers - 1:
            label = "Output<br>(2)"
        else:
            act = activations[l - 1] if l - 1 < len(activations) else "?"
            label = f"H{l}<br>{sizes[l]}n<br><span style='font-size:8px;color:#9ca3af'>{act}</span>"
        fig.add_annotation(x=xs[l], y=-0.02, text=label, showarrow=False,
            font=dict(color="#d1d5db", size=9, family="IBM Plex Mono"),
            align="center")

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=240,
        margin=dict(l=10, r=10, t=20, b=55),
        xaxis=dict(showgrid=False, showticklabels=False, range=[-0.02, 1.02], zeroline=False),
        yaxis=dict(showgrid=False, showticklabels=False, range=[-0.12, 1.05], zeroline=False),
    )
    return fig


def plot_activation_landscape(activation_name):
    x = np.linspace(-4, 4, 300)
    x_t = torch.FloatTensor(x)
    fn_map = {
        "ReLU": nn.ReLU(),
        "Tanh": nn.Tanh(),
        "Sigmoid": nn.Sigmoid(),
        "LeakyReLU": nn.LeakyReLU(0.1),
        "ELU": nn.ELU(),
        "GELU": nn.GELU(),
    }
    y_vals = {}
    for name, fn in fn_map.items():
        with torch.no_grad():
            y_vals[name] = fn(x_t).numpy()

    fig = go.Figure()
    palette = {"ReLU": "#a78bfa", "Tanh": "#34d399", "Sigmoid": "#f472b6",
               "LeakyReLU": "#60a5fa", "ELU": "#fbbf24", "GELU": "#f87171"}
    for name, y in y_vals.items():
        fig.add_trace(go.Scatter(
            x=x, y=y, name=name,
            line=dict(color=palette[name], width=2.5 if name == activation_name else 1.2,
                      dash="solid" if name == activation_name else "dot"),
            opacity=1.0 if name == activation_name else 0.35
        ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(8,8,20,0.9)",
        font=dict(color="white", family="Sora"),
        height=240,
        margin=dict(l=10, r=10, t=35, b=20),
        legend=dict(font=dict(size=9), bgcolor="rgba(0,0,0,0)"),
        title=dict(text=f"Activation Functions  (highlighted: {activation_name})",
                   font=dict(color="#a78bfa", size=11, family="IBM Plex Mono")),
        xaxis=dict(showgrid=True, gridcolor="#1f1f3a", zeroline=True,
                   zerolinecolor="#374151", title_text="Input"),
        yaxis=dict(showgrid=True, gridcolor="#1f1f3a", zeroline=True,
                   zerolinecolor="#374151", title_text="Output"),
    )
    return fig

# ─── Sidebar ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="section-header">📊 Dataset</div>', unsafe_allow_html=True)
    dataset_choice = st.selectbox("Type", [
        "🌙 Two Moons", "⭕ Circles", "⊕ XOR", "🌀 Spiral", "🔵 Gaussian Blobs"
    ], index=0)
    n_samples = st.slider("Samples", 100, 600, 300, 50)
    noise = st.slider("Noise Level", 0.0, 0.5, 0.15, 0.01)

    st.markdown('<div class="section-header">🏗️ Architecture</div>', unsafe_allow_html=True)
    n_hidden = st.slider("Hidden Layers", 1, 5, 2)

    hidden_sizes = []
    hidden_acts  = []
    for i in range(n_hidden):
        c1, c2 = st.columns([1, 1])
        with c1:
            n = st.selectbox(f"L{i+1} neurons", [4, 8, 16, 32, 64, 128], index=2, key=f"n{i}")
        with c2:
            a = st.selectbox(f"L{i+1} act.", list(ACTIVATIONS.keys()), index=0, key=f"a{i}")
        hidden_sizes.append(n)
        hidden_acts.append(a)

    # Show architecture summary
    arch_str = "2 → " + " → ".join(str(n) for n in hidden_sizes) + " → 2"
    total_params = sum(
        (([2] + hidden_sizes)[i] + 1) * ([hidden_sizes + [2]][0][i])
        for i in range(n_hidden + 1)
    )
    st.markdown(f"""
    <div class="info-box">
        <span class="chip">Arch: {arch_str}</span><br>
        <span class="chip">Params: ~{total_params:,}</span>
        <span class="chip">CPU Only</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">⚙️ Training</div>', unsafe_allow_html=True)
    optimizer_name = st.selectbox("Optimizer", ["Adam", "SGD + Momentum", "RMSProp", "AdaGrad"])
    lr = st.select_slider("Learning Rate",
        options=[0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3],
        value=0.01, format_func=lambda x: f"{x:.4f}")
    batch_size = st.select_slider("Batch Size", options=[16, 32, 64, 128, 256], value=32)
    epochs = st.slider("Epochs", 20, 500, 150, 10)

    st.markdown("---")
    btn_train = st.button("▶  Train Network")
    btn_reset = st.button("↺  Reset")

    st.markdown("""
    <div class="info-box" style="margin-top:1rem;">
        <b style="color:#a78bfa;">Quick Guide</b><br>
        1. Pick a dataset + noise<br>
        2. Design your architecture<br>
        3. Choose optimizer & LR<br>
        4. Hit <b>Train Network</b><br><br>
        <b style="color:#a78bfa;">Tips</b><br>
        • Spiral/XOR needs deeper nets<br>
        • High noise → regularize (L2 built-in)<br>
        • Adam usually converges fastest
    </div>
    """, unsafe_allow_html=True)

# ─── Reset ───────────────────────────────────────────────────────────────────────
if btn_reset:
    st.session_state.model    = None
    st.session_state.losses   = []
    st.session_state.accuracies = []
    st.session_state.trained  = False
    st.session_state.X = None
    st.session_state.y = None
    st.rerun()

# ─── Generate Dataset ─────────────────────────────────────────────────────────────
X, y = generate_dataset(dataset_choice, n_samples, noise)
st.session_state.X = X
st.session_state.y = y

# ─── Train ───────────────────────────────────────────────────────────────────────
if btn_train:
    with st.spinner("Building network…"):
        model = build_model(hidden_sizes, hidden_acts)

    progress_bar  = st.progress(0)
    status_text   = st.empty()

    losses, accuracies = train_model(
        model, X, y,
        optimizer_name, lr, batch_size, epochs,
        progress_bar, status_text
    )

    progress_bar.empty()
    status_text.empty()

    st.session_state.model      = model
    st.session_state.losses     = losses
    st.session_state.accuracies = accuracies
    st.session_state.trained    = True
    st.rerun()

# ─── Layout ──────────────────────────────────────────────────────────────────────
col_left, col_right = st.columns([1, 1.6], gap="large")

with col_left:
    # Status chip
    if st.session_state.trained:
        st.markdown('<span class="status-trained">● Model Trained</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-untrained">○ Not Trained</span>', unsafe_allow_html=True)

    st.markdown('<div class="section-header">Dataset Preview</div>', unsafe_allow_html=True)
    st.plotly_chart(plot_dataset(X, y, title=dataset_choice), width='stretch')

    st.markdown('<div class="section-header">Network Architecture</div>', unsafe_allow_html=True)
    st.plotly_chart(plot_architecture_diagram(hidden_sizes, hidden_acts), width='stretch')

    st.markdown('<div class="section-header">Activation Functions</div>', unsafe_allow_html=True)
    st.plotly_chart(plot_activation_landscape(hidden_acts[0] if hidden_acts else "ReLU"),
                    width='stretch')

with col_right:
    if st.session_state.trained and st.session_state.model is not None:
        model = st.session_state.model
        losses = st.session_state.losses
        accs = st.session_state.accuracies

        # ── Metric row
        final_loss  = losses[-1]
        final_acc   = accs[-1]
        best_acc    = max(accs)
        total_params_real = sum(p.numel() for p in model.parameters())

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-label">Final Loss</div>
                <div class="metric-value" style="font-size:1.3rem;color:#f472b6">{final_loss:.4f}</div>
            </div>""", unsafe_allow_html=True)
        with m2:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-label">Accuracy</div>
                <div class="metric-value" style="font-size:1.3rem">{final_acc:.1f}%</div>
            </div>""", unsafe_allow_html=True)
        with m3:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-label">Best Acc</div>
                <div class="metric-value" style="font-size:1.3rem;color:#34d399">{best_acc:.1f}%</div>
            </div>""", unsafe_allow_html=True)
        with m4:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-label">Parameters</div>
                <div class="metric-value" style="font-size:1.3rem;color:#60a5fa">{total_params_real:,}</div>
            </div>""", unsafe_allow_html=True)

        tab1, tab2, tab3 = st.tabs(["🗺️ Decision Boundary", "📈 Training Curves", "⚖️ Weights"])

        with tab1:
            with st.spinner("Rendering boundary…"):
                st.plotly_chart(plot_decision_boundary(model, X, y), width='stretch')
            st.markdown("""
            <div class="info-box">
                🟣 <b style="color:#be185d">Pink region</b> → model predicts Class 0 &nbsp;|&nbsp;
                🟢 <b style="color:#065f46">Green region</b> → model predicts Class 1 &nbsp;|&nbsp;
                ⬜ <b>White line</b> → 50% confidence boundary
            </div>
            """, unsafe_allow_html=True)

        with tab2:
            st.plotly_chart(plot_training_curves(losses, accs), width='stretch')
            # Convergence analysis
            if len(losses) > 10:
                early_loss = np.mean(losses[:10])
                late_loss  = np.mean(losses[-10:])
                improvement = (early_loss - late_loss) / early_loss * 100
                st.markdown(f"""
                <div class="info-box">
                    📉 Loss improved by <b style="color:#34d399">{improvement:.1f}%</b> 
                    from first 10 to last 10 epochs &nbsp;·&nbsp;
                    Final accuracy: <b style="color:#a78bfa">{final_acc:.1f}%</b>
                </div>
                """, unsafe_allow_html=True)

        with tab3:
            st.plotly_chart(plot_weight_histograms(model), width='stretch')
            # Weight stats
            all_w = np.concatenate([p.detach().numpy().flatten()
                                    for n, p in model.named_parameters() if "weight" in n])
            st.markdown(f"""
            <div class="info-box">
                Weight stats — 
                Mean: <b style="color:#a78bfa">{all_w.mean():.4f}</b> &nbsp;·&nbsp;
                Std: <b style="color:#f472b6">{all_w.std():.4f}</b> &nbsp;·&nbsp;
                Min: <b style="color:#60a5fa">{all_w.min():.4f}</b> &nbsp;·&nbsp;
                Max: <b style="color:#34d399">{all_w.max():.4f}</b>
            </div>
            """, unsafe_allow_html=True)

    else:
        st.markdown('<div class="section-header">Output</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="info-box" style="padding:2rem;text-align:center;">
            <div style="font-size:2.5rem;margin-bottom:0.8rem;">🧠</div>
            <div style="font-family:'IBM Plex Mono';color:#6b7280;font-size:0.85rem;">
                Configure your network in the sidebar<br>and press <b style="color:#a78bfa">▶ Train Network</b>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('<div class="section-header">What you will see</div>', unsafe_allow_html=True)
        for item in [
            ("🗺️ Decision Boundary", "How the model separates the two classes in 2D space"),
            ("📈 Loss & Accuracy Curves", "How fast and smoothly the network converges"),
            ("⚖️ Weight Distributions", "What values the learned weights take — check for collapse or explosion"),
        ]:
            st.markdown(f"""
            <div class="layer-row">
                {item[0]} — <span style="color:#9ca3af">{item[1]}</span>
            </div>""", unsafe_allow_html=True)

# ─── Learn Section ────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("## 📚 Concepts")
t1, t2, t3, t4, t5 = st.tabs([
    "🧠 Neural Nets", "📉 Optimizers", "🔗 Activations", "⚙️ Hyperparameters", "🔬 Overfitting"
])

with t1:
    st.markdown("""
    ### How Neural Networks Learn
    A neural network is a function approximator made of **layers of neurons**.  
    Each neuron computes a **weighted sum** of its inputs, adds a bias, then applies a **non-linear activation**.

    ```
    output = activation( W·x + b )
    ```

    Training works by:
    1. **Forward pass** — compute prediction
    2. **Loss** — measure how wrong the prediction is (CrossEntropy for classification)
    3. **Backward pass (Backprop)** — compute gradients of loss w.r.t. every weight
    4. **Update** — nudge weights in the direction that reduces loss
    """)

with t2:
    st.markdown("""
    ### Optimizers Compared

    | Optimizer | Best For | Pitfalls |
    |-----------|----------|----------|
    | **SGD + Momentum** | Large datasets, fine-tuned LR | Slow start, sensitive to LR |
    | **Adam** | Most tasks — adaptive per-param LR | Can overfit on small datasets |
    | **RMSProp** | Non-stationary problems, RNNs | Hyperparameter sensitive |
    | **AdaGrad** | Sparse features | LR shrinks to zero over time |

    **Rule of thumb:** Start with Adam at lr=0.001. If it overfits, try SGD.
    """)

with t3:
    st.markdown("""
    ### Activation Functions

    | Function | Formula | Use When |
    |----------|---------|----------|
    | **ReLU** | max(0, x) | Default — fast, simple, avoids vanishing gradient |
    | **Tanh** | (eˣ-e⁻ˣ)/(eˣ+e⁻ˣ) | Centered output, good for shallow nets |
    | **Sigmoid** | 1/(1+e⁻ˣ) | Output layer for binary classification |
    | **LeakyReLU** | max(0.1x, x) | Prevents dying ReLU problem |
    | **GELU** | x·Φ(x) | Transformers — smooth, probabilistic gating |

    **Dead ReLU:** Neurons that always output 0 because weights push input < 0. Use LeakyReLU or ELU to fix.
    """)

with t4:
    st.markdown("""
    ### Key Hyperparameters

    **Learning Rate** — the single most important hyperparameter.  
    - Too high → loss explodes or oscillates  
    - Too low → training is painfully slow  
    - Sweet spot: ~0.001–0.01 for Adam

    **Batch Size:**
    - Small batches → noisy gradients, can escape local minima, slower per epoch
    - Large batches → stable gradients, faster per epoch, might overfit

    **Network Width vs. Depth:**
    - *Width* (neurons per layer) → captures complex patterns within a level of abstraction  
    - *Depth* (more layers) → learns hierarchical, compositional features
    """)

with t5:
    st.markdown("""
    ### Overfitting & Underfitting

    | Symptom | Cause | Fix |
    |---------|-------|-----|
    | High train acc, low test acc | **Overfitting** | More data, dropout, L2 reg, simpler model |
    | Low train acc | **Underfitting** | More layers/neurons, more epochs, lower LR |
    | Loss explodes | LR too high or bad init | Lower LR, clip gradients |
    | Loss plateaus early | LR too low, vanishing gradient | Higher LR, better activation |

    This playground uses **L2 regularization (weight_decay=1e-4)** in all optimizers.  
    Watch the weight histogram — healthy weights should be roughly **Gaussian centered near 0**.
    """)

with st.expander("ℹ️ About This Lab"):
    st.markdown("""
    ### Neural Network Playground — Deep Learning Virtual Lab

    **Objective:** Build, train, and understand neural networks visually — no GPU needed.

    **Features:**
    - 5 classic 2D classification datasets (Moons, Circles, XOR, Spiral, Gaussian)
    - Dynamic architecture builder: 1–5 hidden layers, 6 activation choices
    - 4 optimizers with configurable LR, batch size, epochs
    - Live training progress with loss & accuracy curves
    - Decision boundary visualization (probability heatmap + boundary line)
    - Layer-wise weight distribution histograms
    - Activation function comparison chart
    - Built-in L2 regularization + gradient clipping

    **Stack:** PyTorch (CPU) · Streamlit · Plotly · scikit-learn
    """)