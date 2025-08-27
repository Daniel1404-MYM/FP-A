import streamlit as st
import pandas as pd
import plotly.express as px
import duckdb

# ===========================
# PAGE SETUP
# ===========================
st.set_page_config(page_title="📊 Dashboard Pro", layout="wide")

st.title("📊 AI-Powered Dashboard Maker")
st.write("Prototype v1.0 - Streamlit + DuckDB + Plotly")

# ===========================
# FILE UPLOAD
# ===========================
uploaded_file = st.file_uploader("📂 Upload Excel File", type=["xlsx"])

if uploaded_file:
    try:
        df = pd.read_excel(uploaded_file)
        st.subheader("📜 Data Preview")
        st.dataframe(df.head())

        # Buat table di DuckDB
        duckdb.sql("CREATE OR REPLACE TABLE sales AS SELECT * FROM df")

        # =======================
        # CHART 1: Sales per Region
        # =======================
        st.subheader("🌍 Sales by Region")
        sales_region = duckdb.sql("""
            SELECT region, SUM(amount) as total_sales
            FROM sales
            GROUP BY region
            ORDER BY total_sales DESC
        """).df()

        fig1 = px.bar(sales_region, x="region", y="total_sales", text_auto=True)
        st.plotly_chart(fig1, use_container_width=True)

        # =======================
        # CHART 2: Sales per Product
        # =======================
        st.subheader("📦 Sales by Product")
        sales_product = duckdb.sql("""
            SELECT product, SUM(amount) as total_sales
            FROM sales
            GROUP BY product
            ORDER BY total_sales DESC
        """).df()

        fig2 = px.pie(sales_product, names="product", values="total_sales")
        st.plotly_chart(fig2, use_container_width=True)

        # =======================
        # CHART 3: Trend by Month
        # =======================
        st.subheader("📈 Monthly Sales Trend")
        sales_month = duckdb.sql("""
            SELECT strftime(date, '%Y-%m') as month, SUM(amount) as total_sales
            FROM sales
            GROUP BY month
            ORDER BY month
        """).df()

        fig3 = px.line(sales_month, x="month", y="total_sales", markers=True)
        st.plotly_chart(fig3, use_container_width=True)

    except Exception as e:
        st.error(f"🚨 Error saat membaca data: {e}")

else:
    st.info("⬆️ Upload file Excel (contoh: dummy_sales.xlsx) untuk mulai membuat dashboard.")
