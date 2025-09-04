import streamlit as st
import pdfplumber
import pandas as pd
import io

# Function to extract tables from PDF
def extract_tables_from_pdf(pdf_file):
    all_tables = []
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            tables = page.extract_tables()
            for table in tables:
                df = pd.DataFrame(table)
                all_tables.append(df)
    return all_tables

# Function to save tables to an Excel file
def save_to_excel(tables):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for i, table in enumerate(tables):
            table.to_excel(writer, sheet_name=f"Sheet{i+1}", index=False, header=False)
    output.seek(0)
    return output

# Streamlit App
st.title("PDF to Excel Converter")

st.write("Upload a PDF file, and the app will extract tabular data and convert it into an Excel file.")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file is not None:
    st.write("Extracting tables from the PDF...")
    tables = extract_tables_from_pdf(uploaded_file)

    if tables:
        st.write(f"Extracted {len(tables)} tables from the PDF.")

        excel_file = save_to_excel(tables)

        st.download_button(
            label="Download Excel File",
            data=excel_file,
            file_name="extracted_data.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    else:
        st.write("No tables found in the PDF.")
