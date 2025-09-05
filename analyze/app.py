import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import os
from groq import Groq
from dotenv import load_dotenv

# ==============================
# ✅ Setup & Config
# ==============================
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

st.set_page_config(page_title="Budget vs. Actuals AI", page_icon="📊", layout="wide")
st.title("📊 Budget vs. Actuals AI – Variance Analysis & Commentary")
st.write("Upload your Budget vs. Actuals file and get AI-driven financial insights!")

if not GROQ_API_KEY:
    st.error("🚨 API Key is missing! Set it in Streamlit Secrets or a .env file.")
    st.stop()

# Current supported production model IDs from Groq docs:
# - llama-3.1-8b-instant (Meta)
# - llama-3.3-70b-versatile (Meta)
# - openai/gpt-oss-20b (OpenAI)
# - openai/gpt-oss-120b (OpenAI)
# Docs: https://console.groq.com/docs/models
MODEL_OPTIONS = [
    "llama-3.1-8b-instant",
    "llama-3.3-70b-versatile",
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b",
]

with st.sidebar:
    st.header("⚙️ Settings")
    selected_model = st.selectbox("Model (Groq)", MODEL_OPTIONS, index=0)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.05)
    max_tokens = st.slider("Max tokens", 256, 4096, 1200, 64)

# File uploader
uploaded_file = st.file_uploader("📂 Upload your dataset (Excel format)", type=["xlsx"]) 

if uploaded_file:
    # ==============================
    # 🧮 Read & Validate Data
    # ==============================
    try:
        df = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Gagal membaca file Excel: {e}")
        st.stop()

    required_columns = ["Category", "Budget", "Actual"]
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        st.error("⚠️ The uploaded file must contain 'Category', 'Budget', and 'Actual' columns! Missing: " + ", ".join(missing))
        st.stop()

    # Coerce numeric columns
    for col in ["Budget", "Actual"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows with all NaNs in numeric cols
    df = df.dropna(subset=["Budget", "Actual"], how="all").copy()

    # ==============================
    # 📊 Calculations
    # ==============================
    df["Variance"] = df["Actual"] - df["Budget"]
    # Handle division by zero safely for Variance %
    df["Variance %"] = np.where(
        df["Budget"].replace({0: np.nan}).notna(),
        (df["Variance"] / df["Budget"]) * 100,
        np.nan,
    )

    # Optional: sort by largest absolute variance
    df = df.reindex(df["Variance"].abs().sort_values(ascending=False).index)

    # ==============================
    # 👀 Preview
    # ==============================
    st.subheader("📊 Data Preview with Variance Calculation")
    st.dataframe(df, use_container_width=True)

    # ==============================
    # 📈 Charts
    # ==============================
    st.subheader("📈 Budget vs. Actual Variance Analysis")

    try:
        fig_bar = px.bar(
            df,
            x="Category",
            y="Variance",
            color="Variance",
            title="📊 Variance by Category",
            text_auto=".2s",
            color_continuous_scale=["red", "yellow", "green"],
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    except Exception:
        # Fallback without text_auto for older Plotly
        fig_bar = px.bar(
            df,
            x="Category",
            y="Variance",
            color="Variance",
            title="📊 Variance by Category",
            color_continuous_scale=["red", "yellow", "green"],
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    fig_line = px.line(
        df,
        x="Category",
        y=["Budget", "Actual"],
        markers=True,
        title="📉 Budget vs. Actual Performance",
    )
    st.plotly_chart(fig_line, use_container_width=True)

    # ==============================
    # 🤖 AI Section
    # ==============================
    st.subheader("🤖 AI-Powered Variance Analysis")

    # Summarize data to keep prompts compact & cheap
    df_summary = df.describe(include="all").transpose().fillna("").to_string()
    df_sample = df.head(15).to_string(index=False)

    base_prompt = f"""
You are a senior FP&A analyst.
Analyze the following Budget vs Actuals table. Identify 5-8 key insights (drivers, trends, spikes), call out top positive/negative categories, and give actionable recommendations to improve performance next period. Keep it concise and bullet-based.

Summary stats:
{df_summary}

Sample rows (first 15):
{df_sample}
"""

    client = Groq(api_key=GROQ_API_KEY)

    def call_groq(messages):
        return client.chat.completions.create(
            model=selected_model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    try:
        resp = call_groq([
            {"role": "system", "content": "You are an AI financial analyst for variance analysis."},
            {"role": "user", "content": base_prompt},
        ])
        st.write(resp.choices[0].message.content)
    except Exception as e:
        st.error(f"Groq API error: {e}")

    # ==============================
    # 🗣️ Q&A Chat
    # ==============================
    st.subheader("🗣️ Chat with AI About Variance Analysis")
    user_query = st.text_input("🔍 Ask the AI about your variance data:")

    if user_query:
        chat_prompt = f"""
You are supporting a CFO in a quick review. Be specific and numeric where possible.

Summary stats:
{df_summary}

Sample rows:
{df_sample}

User question: {user_query}
"""
        try:
            chat_resp = call_groq([
                {"role": "system", "content": "You are an AI financial analyst helping users understand variance analysis."},
                {"role": "user", "content": chat_prompt},
            ])
            st.write(chat_resp.choices[0].message.content)
        except Exception as e:
            st.error(f"Groq API error (chat): {e}")
