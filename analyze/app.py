import streamlit as st
import pandas as pd
import plotly.express as px
import os
from groq import Groq
from dotenv import load_dotenv

#######################################
# PAGE SETUP
#######################################
st.set_page_config(page_title="Budget vs. Actuals AI", page_icon="📊", layout="wide")
st.title("📊 Budget vs. Actuals AI – Variance Analysis & Commentary")
st.write("Upload your Budget vs. Actuals file and get AI-driven financial insights!")

#######################################
# LOAD API
#######################################
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("🚨 API Key is missing! Set it in Streamlit Secrets or a .env file.")
    st.stop()

client = Groq(api_key=GROQ_API_KEY)

#######################################
# FILE UPLOAD
#######################################
uploaded_file = st.file_uploader("📂 Upload your dataset (Excel format)", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)

    # Check required columns
    required_columns = ["Category", "Budget", "Actual"]
    if not all(col in df.columns for col in required_columns):
        st.error("⚠️ The uploaded file must contain 'Category', 'Budget', and 'Actual' columns!")
        st.stop()

    #######################################
    # VARIANCE CALCULATION
    #######################################
    df["Variance"] = df["Actual"] - df["Budget"]
    df["Variance %"] = (df["Variance"] / df["Budget"]) * 100

    st.subheader("📊 Data Preview with Variance Calculation")
    st.dataframe(df)

    #######################################
    # VISUALIZATION
    #######################################
    st.subheader("📈 Budget vs. Actual Variance Analysis")

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

    fig_line = px.line(
        df,
        x="Category",
        y=["Budget", "Actual"],
        markers=True,
        title="📉 Budget vs. Actual Performance",
    )
    st.plotly_chart(fig_line, use_container_width=True)

    #######################################
    # AI COMMENTARY (Initial Summary)
    #######################################
    st.subheader("🤖 AI-Powered Variance Analysis")

    variance_preview = df.head(20).to_string(index=False)
    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are an AI financial analyst providing variance analysis insights on budget vs. actuals."},
            {"role": "user", "content": f"Here is the budget vs. actual variance summary:\n{variance_preview}\nWhat are the key insights and recommendations?"}
        ],
        model="llama-3.1-8b-instant",
    )
    ai_summary = response.choices[0].message.content
    st.write(ai_summary)

    #######################################
    # AI CHAT MODE
    #######################################
    st.subheader("💬 Chat with AI About Variance Analysis")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            {"role": "system", "content": "You are an AI financial analyst helping users understand their budget vs. actual variance analysis."},
            {"role": "assistant", "content": ai_summary}
        ]

    # tampilkan riwayat chat
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.chat_message("user").write(msg["content"])
        elif msg["role"] == "assistant":
            st.chat_message("assistant").write(msg["content"])

    # input pertanyaan baru
    if question := st.chat_input("Tanyakan sesuatu tentang variance data..."):
        st.session_state.chat_history.append({"role": "user", "content": question})
        st.chat_message("user").write(question)

        try:
            chat_response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=st.session_state.chat_history,
                temperature=0.7
            )
            answer = chat_response.choices[0].message.content
            st.session_state.chat_history.append({"role": "assistant", "content": answer})
            st.chat_message("assistant").write(answer)
        except Exception as e:
            st.error(f"❌ Error chat: {e}")
