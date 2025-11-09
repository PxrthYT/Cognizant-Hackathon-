import os, glob, json, io, time
from typing import List, Tuple, Dict, Any
import numpy as np
from PIL import Image
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision import models
from shutil import copyfile
import csv, time, os
import streamlit as st
import pandas as pd
import plotly.express as px

# --- NEW IMPORTS ---
import requests
import google.generativeai as genai
from dotenv import load_dotenv
import re  # for JSON repair
# ---------------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
INDEX_PREFIX = "soil_memory"  # soil_memory_X.npy + soil_memory_meta.json

# --- Gemini API Configuration ---
load_dotenv()
api_key = os.environ.get("GOOGLE_API_KEY")

if not api_key:
    st.error("API key not found. Please create a .env file with GOOGLE_API_KEY.")
else:
    genai.configure(api_key=api_key)

# --- Gemini Model Caching ---
@st.cache_resource(show_spinner="Loading Gemini AI model...")
def get_gemini_model():
    """
    Returns a GenerativeModel your key can actually use.
    Prefers 2.5/`latest` models; falls back gracefully.
    """
    available = []
    try:
        for m in genai.list_models():
            if "generateContent" in getattr(m, "supported_generation_methods", []):
                available.append(m.name)  # e.g., "models/gemini-2.5-flash"
    except Exception:
        pass

    preferred = [
        "models/gemini-2.5-flash",
        "models/gemini-2.5-pro",
        "models/gemini-flash-latest",
        "models/gemini-pro-latest",
        "models/gemini-2.0-flash",
        "models/gemini-2.0-flash-001",
        "models/gemini-1.5-flash",
        "models/gemini-1.5-pro",
        "models/gemini-pro",
    ]
    chosen = next((m for m in preferred if m in available), None)

    candidates = [chosen] if chosen else (available or preferred)
    for c in candidates:
        try:
            mdl = genai.GenerativeModel(c)
            # Store model name in session state for sidebar display
            if 'gemini_model_name' not in st.session_state:
                st.session_state.gemini_model_name = c
            return mdl
        except Exception:
            continue

    raise RuntimeError(f"No compatible Gemini model found. Visible: {available or '(none)'}")

# --- JSON Repair/Parser ---
def parse_llm_json_output(raw_output: str):
    """
    Attempts to coerce almost-JSON from LLMs into valid JSON:
    - strips code fences
    - normalizes quotes
    - trims to the outermost { ... }
    - removes trailing commas
    - balances braces/brackets if off by a little
    """
    try:
        s = (raw_output or "").strip()

        # strip ``` and ```json fences
        s = re.sub(r"```(?:json)?", "", s, flags=re.IGNORECASE).replace("```", "").strip()

        # normalize smart quotes -> straight quotes
        s = (s.replace("“", '"').replace("”", '"')
               .replace("’", "'").replace("‘", "'"))

        # keep only the outermost JSON object
        start, end = s.find("{"), s.rfind("}")
        if start == -1 or end == -1:
            raise ValueError("No JSON object braces found.")
        s = s[start:end+1]

        # remove trailing commas before } or ]
        s = re.sub(r",\s*(?=[}\]])", "", s)

        # balance braces/brackets if off by a small count
        diff_brace = s.count("{") - s.count("}")
        if diff_brace > 0:
            s += "}" * diff_brace
        diff_brack = s.count("[") - s.count("]")
        if diff_brack > 0:
            s += "]" * diff_brack

        return json.loads(s)
    except Exception as e:
        st.error(f"Error decoding AI JSON response: {e}")
        st.text_area("Raw AI Output", raw_output, height=240)
        return None
# ---------------------------------


# ---------- Model / Featurizer ----------
@st.cache_resource(show_spinner="Loading classification model...")
def get_featurizer():
    m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    backbone = nn.Sequential(*list(m.children())[:-1]).to(DEVICE).eval()
    return backbone

PREPROCESS = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

def embed_pil(img: Image.Image, backbone) -> np.ndarray:
    img = img.convert("RGB")
    x = PREPROCESS(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        f = backbone(x).squeeze(-1).squeeze(-1)
        f = nn.functional.normalize(f, dim=-1)
    return f.cpu().numpy().astype(np.float32)

def embed_path(path: str, backbone) -> np.ndarray:
    return embed_pil(Image.open(path), backbone)

# ---------- Index build/load ----------
def list_classes(data_dir: str) -> List[str]:
    return sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])

def safe_relpath(p: str) -> str:
    return os.path.relpath(p, start=os.getcwd())

def build_index(data_dir: str, prefix: str = INDEX_PREFIX) -> Tuple[np.ndarray, dict]:
    backbone = get_featurizer()
    classes = list_classes(data_dir)
    labels, paths, vecs = [], [], []
    for c in classes:
        for fp in glob.glob(os.path.join(data_dir, c, "*")):
            if os.path.isdir(fp):
                continue
            ext = os.path.splitext(fp)[1].lower()
            if ext not in [".jpg", ".jpeg", ".png", ".webp"]:
                continue
            try:
                v = embed_path(fp, backbone)
                labels.append(c); paths.append(safe_relpath(fp)); vecs.append(v)
            except Exception:
                pass
    if len(vecs) == 0:
        raise RuntimeError("No images found to build the index.")
    X = np.vstack([v.reshape(1, -1) for v in vecs]).astype(np.float32)
    X = normalize(X)
    np.save(prefix + "_X.npy", X)
    meta = {"labels": labels, "paths": paths, "classes": classes, "data_dir": data_dir}
    with open(prefix + "_meta.json", "w") as f:
        json.dump(meta, f)
    return X, meta

def load_index(prefix: str = INDEX_PREFIX) -> Tuple[np.ndarray, dict]:
    X = np.load(prefix + "_X.npy")
    with open(prefix + "_meta.json") as f:
        meta = json.load(f)
    return X, meta

def ensure_index(data_dir: str):
    if not (os.path.exists(INDEX_PREFIX + "_X.npy") and os.path.exists(INDEX_PREFIX + "_meta.json")):
        return build_index(data_dir)
    return load_index()

def log_prediction(logfile, src, shown_label, top_label, sim, low_conf):
    os.makedirs(os.path.dirname(logfile), exist_ok=True) if os.path.dirname(logfile) else None
    header = ["ts","source","shown_label","top_label","similarity","low_confidence"]
    ts = int(time.time() * 1000)
    write_header = not os.path.exists(logfile)
    with open(logfile, "a", newline="") as f:
        w = csv.writer(f)
        if write_header: w.writerow(header)
        w.writerow([ts, src, shown_label, top_label, f"{sim:.4f}", int(low_conf)])

# ---------- kNN Predict ----------
def knn_predict(img: Image.Image, X: np.ndarray, labels: List[str], k: int = 3):
    nbrs = NearestNeighbors(n_neighbors=min(k, len(labels)), metric="cosine").fit(X)
    q = embed_pil(img, get_featurizer()).reshape(1, -1)
    dists, idxs = nbrs.kneighbors(q, return_distance=True)
    sims = 1 - dists[0]
    return idxs[0], sims

# ---------- Utils ----------
def save_uploaded_to_class(upload, class_name: str, data_dir: str) -> str:
    os.makedirs(os.path.join(data_dir, class_name), exist_ok=True)
    ts = int(time.time() * 1000)
    ext = os.path.splitext(upload.name)[1] or ".jpg"
    fp = os.path.join(data_dir, class_name, f"user_{ts}{ext}")
    with open(fp, "wb") as f:
        f.write(upload.getbuffer())
    return fp

# --- Weather Function (Open-Meteo Climate) ---
@st.cache_data(ttl=3600)
def get_climate_data(lat, lon):
    try:
        url = "https://climate-api.open-meteo.com/v1/climate"
        params = {
            "latitude": float(lat),
            "longitude": float(lon),
            "start_date": "2010-01-01",
            "end_date": "2020-12-31",
            "daily": "temperature_2m_mean,precipitation_sum",
        }
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Weather API Error: {e}")
        return None
    except ValueError:
        st.error("Invalid Latitude/Longitude. Please enter numbers.")
        return None

# --- UI Rendering Functions ---
def render_recommendation_cards(parsed_json: Dict[str, Any]):
    """Render AI recommendations in beautiful card format"""
    
    # Custom CSS for beautiful cards
    st.markdown("""
    <style>
    .recommendation-card {
        background: white;
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        border: 1px solid #e0e0e0;
    }
    /* Text in recommendation cards should be black (white background) */
    .recommendation-card, .recommendation-card * {
        color: #1a1a1a !important;
    }
    .card-header {
        font-size: 24px;
        font-weight: 700;
        color: #1a1a1a !important;
        margin-bottom: 12px;
    }
    .card-description {
        color: #1a1a1a !important;
        font-size: 14px;
        margin-bottom: 16px;
        line-height: 1.5;
    }
    .metric-badge {
        display: inline-block;
        background: #e8f5e9;
        color: #2e7d32;
        padding: 6px 12px;
        border-radius: 20px;
        font-size: 13px;
        font-weight: 600;
        margin-right: 8px;
        margin-bottom: 8px;
    }
    /* Metric badges in white cards should have black text if no specific color */
    .recommendation-card .metric-badge:not([style*="color"]) {
        color: #1a1a1a !important;
        background: #f0f0f0 !important;
    }
    .score-badge {
        display: inline-block;
        background: #4caf50;
        color: white;
        padding: 8px 16px;
        border-radius: 8px;
        font-size: 18px;
        font-weight: 700;
    }
    .cost-item {
        display: flex;
        justify-content: space-between;
        padding: 8px 0;
        border-bottom: 1px solid #f0f0f0;
    }
    .cost-total {
        font-weight: 700;
        font-size: 18px;
        color: #1a1a1a !important;
        margin-top: 8px;
    }
    .timeline-button {
        display: inline-block;
        background: #4caf50;
        color: white;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
        margin: 4px;
    }
    .timeline-button-inactive {
        display: inline-block;
        background: #f5f5f5;
        color: #999;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
        margin: 4px;
    }
    .top-plan-section {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 12px;
        padding: 24px;
        color: white;
        margin-top: 24px;
    }
    .top-plan-title {
        font-size: 20px;
        font-weight: 700;
        margin-bottom: 16px;
    }
    /* Analysis text: white on dark background, black on white background */
    /* Default: white text for dark backgrounds */
    .analysis-content {
        color: #ffffff !important;
    }
    /* Headings on dark background - white */
    .analysis-content h3, .analysis-content h2, .analysis-content h1 {
        color: #ffffff !important;
    }
    /* All markdown text on dark background - white */
    .analysis-content .stMarkdown,
    .analysis-content .stMarkdown p,
    .analysis-content .stMarkdown h1,
    .analysis-content .stMarkdown h2,
    .analysis-content .stMarkdown h3 {
        color: #ffffff !important;
    }
    /* But override for white background elements - make everything black */
    .analysis-content .recommendation-card,
    .analysis-content .recommendation-card *,
    .analysis-content .recommendation-card h1,
    .analysis-content .recommendation-card h2,
    .analysis-content .recommendation-card h3,
    .analysis-content .recommendation-card h4,
    .analysis-content .recommendation-card h5,
    .analysis-content .recommendation-card h6,
    .analysis-content .recommendation-card p,
    .analysis-content .recommendation-card div,
    .analysis-content .recommendation-card span,
    .analysis-content .recommendation-card strong,
    .analysis-content .recommendation-card label {
        color: #1a1a1a !important;
    }
    /* Cost items black (on white background) */
    .cost-item, .cost-item * {
        color: #1a1a1a !important;
    }
    .cost-item strong {
        color: #1a1a1a !important;
    }
    /* Metric badges in cards - black text */
    .recommendation-card .metric-badge {
        color: #1a1a1a !important;
    }
    /* Markdown text in white cards - black */
    .analysis-content .recommendation-card .stMarkdown,
    .analysis-content .recommendation-card .stMarkdown *,
    .analysis-content .recommendation-card .stMarkdown p,
    .analysis-content .recommendation-card .stMarkdown h1,
    .analysis-content .recommendation-card .stMarkdown h2,
    .analysis-content .recommendation-card .stMarkdown h3 {
        color: #1a1a1a !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Add wrapper div - text color handled by CSS based on background
    st.markdown('<div class="analysis-content">', unsafe_allow_html=True)
    
    recommendations = parsed_json.get("ai_recommendations", [])
    top_plan = parsed_json.get("top_plan", {})
    
    # Render recommendations
    if recommendations:
        st.markdown("### 🌾 Crop Recommendations")
        st.markdown("<br>", unsafe_allow_html=True)
        
        for idx, rec in enumerate(recommendations, 1):
            crop_name = rec.get("crop", "Unknown Crop")
            reason = rec.get("reason", "")
            score = rec.get("score", 85)
            yield_value = rec.get("yield", "N/A")
            water_req = rec.get("water_requirement", "Medium")
            npk = rec.get("npk", "N/A")
            seed_rate = rec.get("seed_rate", "N/A")
            costs = rec.get("costs", {})
            timeline = rec.get("timeline", {})
            
            # Card container with white background
            st.markdown('<div class="recommendation-card">', unsafe_allow_html=True)
            with st.container():
                col_left, col_right = st.columns([2, 1])
                
                with col_left:
                    st.markdown(f'<div class="card-header">#{idx} {crop_name}</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="card-description">{reason}</div>', unsafe_allow_html=True)
                    
                    # Metrics row
                    metric_col1, metric_col2 = st.columns(2)
                    with metric_col1:
                        water_color = "#4caf50" if "Medium" in water_req else "#ff9800" if "High" in water_req else "#2196f3"
                        st.markdown(f'<span class="metric-badge" style="background: {water_color}20; color: {water_color};">💧 {water_req} Water</span>', unsafe_allow_html=True)
                        st.markdown(f'<span class="metric-badge">📊 {yield_value}</span>', unsafe_allow_html=True)
                
                with col_right:
                    st.markdown(f'<div class="score-badge">{score}/100</div>', unsafe_allow_html=True)
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown(f'<div style="color: #1a1a1a !important;">**NPK:** {npk}</div>', unsafe_allow_html=True)
                    st.markdown(f'<div style="color: #1a1a1a !important;">**Seed:** {seed_rate}</div>', unsafe_allow_html=True)
                    
                    # Cost breakdown
                    st.markdown('<div style="color: #1a1a1a !important;">**Cost Breakdown:**</div>', unsafe_allow_html=True)
                    seed_cost = costs.get("seed", 0)
                    fert_cost = costs.get("fertilizer", 0)
                    other_cost = costs.get("other", 0)
                    total_cost = costs.get("total", seed_cost + fert_cost + other_cost)
                    
                    st.markdown(f'<div class="cost-item">Seed: <strong>${seed_cost}</strong></div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="cost-item">Fertilizer: <strong>${fert_cost}</strong></div>', unsafe_allow_html=True)
                    if other_cost > 0:
                        st.markdown(f'<div class="cost-item">Other: <strong>${other_cost}</strong></div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="cost-total">Total: ${total_cost}</div>', unsafe_allow_html=True)
                
                # Timeline buttons
                st.markdown("<br>", unsafe_allow_html=True)
                planting = timeline.get("planting_window", "")
                harvest = timeline.get("harvest_window", "")
                
                if planting:
                    # Handle formats like "Apr-May", "Apr, May", or "Apr"
                    planting_months = re.split(r'[,\-]', planting) if re.search(r'[,\-]', planting) else [planting]
                    for month in planting_months:
                        month_clean = month.strip()
                        if month_clean:
                            st.markdown(f'<span class="timeline-button-inactive">Plant {month_clean}</span>', unsafe_allow_html=True)
                
                if harvest:
                    # Handle formats like "Jun, Aug, Sep" or "Jun-Aug"
                    harvest_months = re.split(r'[,\-]', harvest) if re.search(r'[,\-]', harvest) else [harvest]
                    for month in harvest_months:
                        month_clean = month.strip()
                        if month_clean:
                            st.markdown(f'<span class="timeline-button">Harvest {month_clean}</span>', unsafe_allow_html=True)
            
            # Close recommendation card
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown("---")
            
    # Top Recommended Plan Section
    if top_plan:
        crop_name = top_plan.get("crop_name", "Unknown")
        seed_rate = top_plan.get("seed_rate_per_acre", "N/A")
        npk = top_plan.get("fertilizer_npk_per_acre", "N/A")
        costs = top_plan.get("est_cost_cad_per_acre", {})
        schedule = top_plan.get("schedule", {})
        insights = top_plan.get("ai_insights", "")
        
        st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    border-radius: 12px; padding: 24px; margin-top: 24px; margin-bottom: 24px;">
            <h2 style="color: white; margin: 0 0 20px 0;">⭐ Top Recommended Plan</h2>
            <h3 style="color: white; margin: 0;">🌱 """ + crop_name + """</h3>
        </div>
        """, unsafe_allow_html=True)
        
        plan_col1, plan_col2 = st.columns(2)
        
        with plan_col1:
            st.markdown("### 📋 Inputs per Acre")
            st.markdown(f"**Seed Rate:** {seed_rate}")
            st.markdown(f"**Fertilizer (NPK):** {npk}")
            st.markdown("---")
            st.markdown("**Cost Breakdown:**")
            st.markdown(f"- Seed: **${costs.get('seed', 0)}**")
            st.markdown(f"- Fertilizer: **${costs.get('fertilizer', 0)}**")
            if costs.get('other', 0) > 0:
                st.markdown(f"- Other: **${costs.get('other', 0)}**")
            st.markdown(f"### Total: **${costs.get('total', 0)}**")
        
        with plan_col2:
            st.markdown("### 📅 Timeline")
            st.markdown(f"**Planting Window:** {schedule.get('planting_window', 'N/A')}")
            st.markdown(f"**Harvest Window:** {schedule.get('harvest_window', 'N/A')}")
            if insights:
                st.markdown("---")
                st.markdown("### 💡 AI Insights")
                st.info(insights)
    
    # AI Soil Validation
    if parsed_json.get("ai_soil_validation"):
        st.markdown("---")
        st.markdown("### 🔍 AI Soil Validation")
        st.info(parsed_json["ai_soil_validation"])
    
    # Close wrapper div
    st.markdown('</div>', unsafe_allow_html=True)

def create_climate_charts(climate_data: Dict[str, Any]):
    """Create visual charts for climate data"""
    if not climate_data or "daily" not in climate_data:
        return
    
    daily = climate_data["daily"]
    dates = pd.to_datetime(daily.get("time", []))
    temps = daily.get("temperature_2m_mean", [])
    precip = daily.get("precipitation_sum", [])
    
    # Aggregate by month
    df = pd.DataFrame({
        "date": dates,
        "temperature": temps,
        "precipitation": precip
    })
    df["month"] = df["date"].dt.month
    monthly = df.groupby("month").agg({
        "temperature": "mean",
        "precipitation": "sum"
    }).reset_index()
    monthly["month_name"] = monthly["month"].map({
        1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
        7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"
    })
    
    # Create charts
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        fig_temp = px.line(monthly, x="month_name", y="temperature", 
                          title="Average Temperature by Month",
                          labels={"temperature": "Temperature (°C)", "month_name": "Month"})
        fig_temp.update_traces(line_color="#667eea", line_width=3)
        fig_temp.update_layout(
            plot_bgcolor="white", 
            paper_bgcolor="white",
            font=dict(color="black", size=12),
            title_font=dict(color="black", size=14),
            xaxis=dict(title_font=dict(color="black"), tickfont=dict(color="black")),
            yaxis=dict(title_font=dict(color="black"), tickfont=dict(color="black"))
        )
        st.plotly_chart(fig_temp, use_container_width=True)
    
    with chart_col2:
        fig_precip = px.bar(monthly, x="month_name", y="precipitation",
                           title="Total Precipitation by Month",
                           labels={"precipitation": "Precipitation (mm)", "month_name": "Month"})
        fig_precip.update_traces(marker_color="#4caf50")
        fig_precip.update_layout(
            plot_bgcolor="white", 
            paper_bgcolor="white",
            font=dict(color="black", size=12),
            title_font=dict(color="black", size=14),
            xaxis=dict(title_font=dict(color="black"), tickfont=dict(color="black")),
            yaxis=dict(title_font=dict(color="black"), tickfont=dict(color="black"))
        )
        st.plotly_chart(fig_precip, use_container_width=True)

# ================== UI ==================
st.set_page_config(page_title="Sustainable Farming", page_icon="🌱", layout="wide", initial_sidebar_state="collapsed")

# Custom CSS for overall styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2.5rem 2rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
    }
    .main-header h1 {
        color: #4caf50 !important;
        margin: 0;
        font-size: 2.8rem;
        font-weight: 800;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        letter-spacing: -0.5px;
        line-height: 1.2;
        text-align: center;
    }
    .main-header p {
        display: none;
    }
    .stMetric {
        background: transparent !important;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: none;
    }
    .prediction-card {
        background: transparent !important;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: none;
        border-left: 4px solid #4caf50;
        margin-bottom: 1rem;
    }
    .location-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    .image-container {
        background: transparent !important;
        padding: 1rem;
        border-radius: 12px;
        box-shadow: none;
        margin-bottom: 1rem;
    }
    .analysis-button {
        width: 100%;
        padding: 1rem;
        font-size: 1.2rem;
        font-weight: 600;
        border-radius: 8px;
        margin-top: 1rem;
    }
    [data-testid="stSidebar"] {
        display: none !important;
    }
    section[data-testid="stSidebar"] {
        display: none !important;
    }
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 100%;
        background: transparent !important;
    }
    /* Dark background for main area */
    .main {
        background: #1e1e1e !important;
    }
    .stApp {
        background: #1e1e1e !important;
    }
    header[data-testid="stHeader"] {
        display: none;
    }
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
    h3 {
        color: #ffffff !important;
        font-weight: 700;
        margin-top: 1.5rem;
    }
    .stExpander {
        background: white;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    /* Make text black on white backgrounds */
    .stExpander, .stExpander * {
        color: #1a1a1a !important;
    }
    .stExpander label, .stExpander p, .stExpander div, .stExpander span {
        color: #1a1a1a !important;
    }
    .streamlit-expanderHeader {
        color: #1a1a1a !important;
    }
    /* Any element with white background should have black text */
    [style*="background: white"], [style*="background-color: white"],
    [style*="background:white"], [style*="background-color:white"] {
        color: #1a1a1a !important;
    }
    [style*="background: white"] *, [style*="background-color: white"] *,
    [style*="background:white"] *, [style*="background-color:white"] * {
        color: #1a1a1a !important;
    }
    /* Change all text to white (except Sustainable Farming heading which stays green) */
    h1:not(.main-header h1), h2, h3, h4, h5, h6 {
        color: #ffffff !important;
    }
    p, span, div, label {
        color: #ffffff !important;
    }
    .stMarkdown, .stMarkdown p, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: #ffffff !important;
    }
    /* File uploader text */
    .stFileUploader label, .stFileUploader p {
        color: #ffffff !important;
    }
    [data-testid="stFileUploader"] label, [data-testid="stFileUploader"] p {
        color: #ffffff !important;
    }
    /* Selectbox text */
    .stSelectbox label, .stSelectbox div {
        color: #ffffff !important;
    }
    [data-testid="stSelectbox"] label, [data-testid="stSelectbox"] div {
        color: #ffffff !important;
    }
    /* Caption text */
    .stCaption {
        color: #ffffff !important;
    }
    /* Button text */
    .stButton>button {
        color: white !important;
    }
    /* Metric labels and values */
    [data-testid="stMetricLabel"], [data-testid="stMetricValue"] {
        color: #ffffff !important;
    }
    /* Text input labels */
    .stTextInput label {
        color: #ffffff !important;
    }
    [data-testid="stTextInput"] label {
        color: #ffffff !important;
    }
    /* Expander text */
    .streamlit-expanderHeader {
        color: #ffffff !important;
    }
    /* All text elements */
    * {
        color: inherit;
    }
    body, .main {
        color: #ffffff !important;
    }
    /* Exception for buttons - keep white text on buttons */
    button, .stButton>button {
        color: white !important;
    }
    /* Exception for input fields - keep default text color */
    input, textarea, select {
        color: #1a1a1a !important;
    }
    /* File name display */
    [data-testid="stFileUploaderFileName"] {
        color: #ffffff !important;
    }
    /* Streamlit widget labels */
    [data-baseweb="select"] label, [data-baseweb="input"] label {
        color: #ffffff !important;
    }
    /* All text in main content */
    .main .element-container, .main .element-container * {
        color: #ffffff !important;
    }
    /* Override for specific Streamlit components */
    .stText, .stMarkdownContainer, .stMarkdownContainer * {
        color: #ffffff !important;
    }
    /* File uploader drag text */
    .uploadedFile {
        color: #ffffff !important;
    }
    /* Keep input field text dark for readability */
    input[type="text"], input[type="number"], textarea, select option {
        color: #1a1a1a !important;
    }
    /* But keep labels white */
    label, .stTextInput label, .stSelectbox label, .stFileUploader label {
        color: #ffffff !important;
    }
    /* Remove white backgrounds from file uploader */
    [data-testid="stFileUploader"] {
        background: transparent !important;
    }
    [data-testid="stFileUploader"] > div {
        background: transparent !important;
    }
    .stFileUploader > div {
        background: transparent !important;
    }
    /* Remove white background from upload dropzone */
    [data-testid="stFileUploaderDropzone"] {
        background: transparent !important;
        border: 2px dashed #4caf50 !important;
    }
    /* Remove white background from metric containers */
    [data-testid="stMetricContainer"] {
        background: transparent !important;
    }
    /* Remove white background from all element containers */
    .element-container {
        background: transparent !important;
    }
    /* Remove white background from block containers */
    .block-container {
        background: transparent !important;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header"><h1>🌱 Sustainable Farming</h1></div>', unsafe_allow_html=True)

# Default values for kNN model settings
data_dir = "data"
accept = 0.78
show_top1_on_lowconf = True

# Location & Climate section in main area (collapsible)
with st.expander("📍 Location & Climate Settings", expanded=False):
    st.markdown("Enter your farm location to get personalized crop recommendations based on local climate data.")
    loc_col1, loc_col2 = st.columns(2)
    with loc_col1:
        lat_in = st.text_input("Farm Latitude", value="44.6488", help="Enter the latitude of your farm location", key="lat_input")
    with loc_col2:
        lon_in = st.text_input("Farm Longitude", value="-63.5752", help="Enter the longitude of your farm location", key="lon_input")

# Load / build index (automatic)
try:
        with st.spinner("Loading index..."):
            X, meta = ensure_index(data_dir)
except Exception as e:
    st.error(f"Index error: {e}")
    st.stop()

labels = meta["labels"]
paths = meta["paths"]
classes = meta["classes"]

# Upload image section
st.markdown("### 📸 Upload Soil Image")
upload_col1, upload_col2 = st.columns([2, 1])
with upload_col1:
    uploaded = st.file_uploader("Upload a soil image (JPG/PNG)", type=["jpg","jpeg","png"], label_visibility="collapsed")
with upload_col2:
    sample_choice = st.selectbox("Or pick a sample", ["(none)"] + paths, label_visibility="visible")

img_to_use = None
img_origin_path = None

if uploaded is not None:
    img_to_use = Image.open(uploaded)
    img_origin_path = "(uploaded)"
elif sample_choice != "(none)":
    try:
        img_to_use = Image.open(sample_choice)
        img_origin_path = sample_choice
    except Exception as e:
        st.error(f"Failed to open selected sample: {e}")

if img_to_use is not None:
    # Main content layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<div class="image-container">', unsafe_allow_html=True)
        st.image(img_to_use, caption="Selected image", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Complete Analysis button under the image
        button_clicked = st.button("🔬 Complete Analysis", type="primary", use_container_width=True, key="complete_analysis")
        
        if button_clicked:
            st.session_state.run_analysis = True
        
        if 'run_analysis' not in st.session_state:
            st.session_state.run_analysis = False

    # Predict
    idxs, sims = knn_predict(img_to_use, X, labels, k=3)
    top_idx = idxs[0]
    top_label = labels[top_idx]
    top_sim = float(sims[0])
    low_conf = top_sim < accept

    label_to_show = "unknown" if (low_conf and not show_top1_on_lowconf) else top_label

    src = uploaded.name if uploaded is not None else (img_origin_path or "(unknown)")
    log_prediction("predictions_log.csv", src, label_to_show, top_label, top_sim, low_conf)

    with col2:
        st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
        st.subheader("🔍 kNN Soil Prediction")
        note = " (low confidence)" if (low_conf and label_to_show != "unknown") else ""
        
        pred_col1, pred_col2 = st.columns(2)
        with pred_col1:
            st.metric("Label", f"{label_to_show}{note}")
        with pred_col2:
            confidence_color = "#4caf50" if not low_conf else "#ff9800"
            st.metric("Similarity", f"{top_sim:.3f}")
        
        th_caption = f"Threshold: {accept:.2f} • {'showing top-1 only' if (low_conf and show_top1_on_lowconf) else 'unknown below threshold'}"
        st.caption(th_caption)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("### Top-3 Nearest Examples")
        ncols = st.columns(3)
        for i, (j, s) in enumerate(zip(idxs, sims)):
            try:
                with ncols[i]:
                    st.image(Image.open(paths[j]), caption=f"{labels[j]} · sim {float(s):.3f}", use_container_width=True)
            except Exception:
                pass

        # --- AI Agronomy Engine ---
        st.markdown("<br>", unsafe_allow_html=True)
        st.divider()
    
    # Run analysis when button is clicked (moved outside columns for centering)
    if st.session_state.get('run_analysis', False):
        if label_to_show == "unknown":
            st.error("Cannot generate AI plan: kNN confidence is too low. Please add this image to memory first.")
            st.session_state.run_analysis = False
        else:
            # Center the AI-Powered Agronomy Plan section
            st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
            st.subheader("🤖 AI-Powered Agronomy Plan")
            st.markdown("</div>", unsafe_allow_html=True)
            
            with st.spinner(f"Fetching climate data for ({lat_in}, {lon_in})..."):
                climate_data = get_climate_data(lat_in, lon_in)

            if climate_data:
                st.success("✅ Climate data fetched!")
                
                # Display climate charts
                create_climate_charts(climate_data)

                try:
                    model = get_gemini_model()
                except Exception as e:
                    st.error(f"Error initializing Gemini: {e}")
                    st.session_state.run_analysis = False
                    st.stop()

                climate_str = json.dumps(climate_data)

                # Enhanced prompt with more fields for better UI
                master_prompt = f"""
Return ONLY a single JSON object that matches the schema below.
Rules: no prose, no Markdown fences, no comments, no trailing commas,
minified (single line), and VALID JSON. Do not omit the outermost object.

You are an expert agronomist for Eastern Canada (Nova Scotia).
Use the inputs to produce comprehensive crop recommendations.

INPUTS:
1. Soil Type (from kNN model): "{top_label}"
2. Location: Latitude {lat_in}, Longitude {lon_in}
3. Climate Data (JSON): {climate_str}

SCHEMA:
{{
  "ai_soil_validation": "Brief validation of soil type suitability (2-3 sentences)",
  "ai_recommendations": [
    {{
      "crop": "Crop name (e.g., Forage Mix, Oats, Wheat)",
      "reason": "Why this crop is suitable (1-2 sentences)",
      "score": 85,
      "yield": "Expected yield (e.g., 9.5 t/ha, 3.8 t/ha)",
      "water_requirement": "Low/Medium/High",
      "npk": "Fertilizer NPK ratio (e.g., 50-50-100, 60-30-20)",
      "seed_rate": "Seed rate per acre (e.g., 18-24 lb/ac, 80-100 lb/ac)",
      "costs": {{
        "seed": 150,
        "fertilizer": 325,
        "other": 0,
        "total": 475
      }},
      "timeline": {{
        "planting_window": "Apr-May",
        "harvest_window": "Jun, Aug, Sep"
      }}
    }}
  ],
  "top_plan": {{
    "crop_name": "Full crop name (e.g., Forage Mix (Timothy/Alfalfa/Clover))",
    "seed_rate_per_acre": "Detailed seed rate",
    "fertilizer_npk_per_acre": "NPK ratio",
    "est_cost_cad_per_acre": {{
      "seed": 150,
      "fertilizer": 325,
      "total": 475
    }},
    "schedule": {{
      "planting_window": "Apr-May",
      "harvest_window": "Jun-Aug"
    }},
    "ai_insights": "Detailed insights about why this is the top recommendation (2-3 sentences)"
  }}
}}

Provide at least 2-3 crop recommendations in ai_recommendations array.
"""

                with st.spinner("Generating analysis..."):
                    try:
                        response = model.generate_content(
                            master_prompt,
                            generation_config={
                                "response_mime_type": "application/json",
                                "temperature": 0
                            }
                        )
                        parsed_json = parse_llm_json_output(getattr(response, "text", ""))

                        if parsed_json:
                            st.success("✨ AI Plan Generated!")
                            st.markdown("<br>", unsafe_allow_html=True)
                            # Render beautiful cards instead of raw JSON
                            render_recommendation_cards(parsed_json)
                            
                            # Optional: Show raw JSON in expander for debugging
                            with st.expander("🔧 View Raw JSON (Debug)"):
                                st.json(parsed_json)
                            
                            # Reset button to run analysis again
                            if st.button("🔄 Run Analysis Again", key="reset_analysis"):
                                st.session_state.run_analysis = False
                                st.rerun()
                        else:
                            st.error("Failed to parse AI JSON. See raw output below.")
                            st.text_area("Raw AI Output", getattr(response, "text", ""), height=240)
                            if st.button("🔄 Try Again", key="retry_analysis"):
                                st.session_state.run_analysis = False
                                st.rerun()

                    except Exception as e:
                        st.error(f"An error occurred while calling the Gemini API: {e}")
                        try:
                            error_output = getattr(response, "text", "No response")
                        except:
                            error_output = "No response available"
                        st.text_area("Raw AI Output (on error)", error_output, height=240)
                        if st.button("🔄 Try Again", key="retry_analysis_error"):
                            st.session_state.run_analysis = False
                            st.rerun()

else:
    st.info("Upload an image or pick a sample from your data folder to run a prediction.")
