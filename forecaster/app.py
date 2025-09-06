import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import os
from groq import Groq
from dotenv import load_dotenv

# Load API key securely
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("🚨 API Key is missing! Set it in Streamlit Secrets or a .env file.")
    st.stop()

# Init Groq client
client = Groq(api_key=GROQ_API_KEY)

# Streamlit App UI
st.set_page_config(page_title="Scenario Planning AI", page_icon="📊", layout="wide")
st.title("📊 Scenario Planning AI – Simulate Financial Scenarios")
st.write("Upload financial data and enter a scenario prompt to simulate different projections!")

# Model selector
selected_model = st.selectbox(
    "🤖 Select AI Model",
    ["llama-3.1-8b-instant", "llama-3.3-70b-versatile", "openai/gpt-oss-120b"],
    index=0
)

# File uploader
uploaded_file = st.file_uploader("📂 Upload your dataset (Excel format)", type=["xlsx"])

if uploaded_file:
    # Read the Excel file
    df = pd.read_excel(uploaded_file)

    # Check for required columns
    required_columns = ["Category", "Base Forecast"]
    if not all(col in df.columns for col in required_columns):
        st.error("⚠️ The uploaded file must contain 'Category' and 'Base Forecast' columns!")
        st.stop()

    # Scenario Input
    scenario_prompt = st.text_area(
        "📝 Enter a financial scenario (e.g., 'Revenue drops 10%', 'Costs increase by 5%'):"
    )

    if st.button("🚀 Generate Scenarios"):
        # Generate Different Scenario Projections
        df["Optimistic"] = df["Base Forecast"] * np.random.uniform(1.1, 1.3, len(df))
        df["Pessimistic"] = df["Base Forecast"] * np.random.uniform(0.7, 0.9, len(df))
        df["Worst Case"] = df["Base Forecast"] * np.random.uniform(0.5, 0.7, len(df))

        # Layout: 2 columns
        col1, col2 = st.columns([2, 1])

        with col1:
            # Display scenario data
            st.subheader("📊 Scenario-Based Projections")
            st.dataframe(df)

            # Plot Scenario Analysis
            fig_scenarios = px.bar(
                df,
                x="Category",
                y=["Base Forecast", "Optimistic", "Pessimistic", "Worst Case"],
                title="📉 Scenario Planning: Financial Projections",
                barmode="group",
                text_auto=".2s",
            )
            st.plotly_chart(fig_scenarios, use_container_width=True)

        with col2:
            # AI Section
            st.subheader("🤖 AI Insights & Discussion")

            # AI Summary of Scenario Data (limit rows to avoid token overload)
            df_preview = df.head(20).to_string(index=False)

            try:
                response = client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": "You are an AI financial analyst providing scenario planning insights."},
                        {"role": "user", "content": f"Here are the scenario projections:\n{df_preview}\nScenario: {scenario_prompt}\nPlease summarize the key insights and recommendations."}
                    ],
                    model=selected_model,
                )
                st.markdown("**AI Analysis:**")
                st.write(response.choices[0].message.content)

            except Exception as e:
                st.error(f"⚠️ AI request failed: {e}")

            # Persistent chat messages
            if "chat_history" not in st.session_state:
                st.session_state.chat_history = []

            user_query = st.text_input("💬 Ask AI something:")
            if st.button("Send") and user_query:
                try:
                    chat_response = client.chat.completions.create(
                        messages=[
                            {"role": "system", "content": "You are an AI financial strategist helping users with scenario-based financial modeling."},
                            *st.session_state.chat_history,
                            {"role": "user", "content": user_query}
                        ],
                        model=selected_model,
                    )

                    ai_answer = chat_response.choices[0].message.content

                    # Save to session state
                    st.session_state.chat_history.append({"role": "user", "content": user_query})
                    st.session_state.chat_history.append({"role": "assistant", "content": ai_answer})

                except Exception as e:
                    st.error(f"⚠️ AI chat request failed: {e}")

            # Show chat history
            if st.session_state.chat_history:
                for msg in st.session_state.chat_history:
                    if msg["role"] == "user":
                        st.markdown(f"**👤 You:** {msg['content']}")
                    else:
                        st.markdown(f"**🤖 AI:** {msg['content']}")
