import streamlit as st
import pdfplumber
import io
import pandas as pd
import numpy as np
from pdf2image import convert_from_bytes
from paddleocr import PaddleOCR
import paddle
import re

st.set_page_config(page_title="Smart Invoice Parser", layout="wide")
st.title("📄 Smart Invoice Data Extractor")

st.sidebar.title("📄 Smart Invoice Parser")
st.sidebar.markdown("AI Powered Invoice Data Extraction")
st.sidebar.markdown("---")
st.sidebar.info("Upload invoice PDF to extract structured data")

uploaded = st.sidebar.file_uploader(
    "📤 Upload Invoice PDF",
    type=["pdf"]
)

paddle.set_flags({'FLAGS_use_mkldnn': False})

ocr = PaddleOCR(
    lang="en",
    use_textline_orientation=True
)

FIELDS = {
    "GSTIN": r"GSTIN\s*[:\-]?\s*([A-Z0-9]{15})",
    "Invoice No": r"(?i)\b(?:invoice:|:invoice no.:|bill)\b\s*(?:no.:\.?|number|#)?\s*[:\-–]?\s*([A-Z0-9]+(?:[\/\-][A-Z0-9]+)*)",
    "Invoice Date": r"(?i)\b(?:date|dated|invoice\s*date)?\b\s*[:\-–]?\s*(\d{1,4}[\/\-.]\d{1,2}[\/\-.]\d{2,4})",
    "Customer Name": r"Customer\s*Name\s*[:\-]?\s*(.+?)(?=Invoice|GSTIN|Booking|$)",
    "Phone": r"Phone\s*[:\-]?\s*(\d{10})",
    "Aadhaar No": r"Aadhaar\s*No\.?\s*[:\-]?\s*(\d{12})"
}

def extract_text_pdf(pdf_bytes):
    text = ""
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            if page.extract_text():
                text += page.extract_text() + "\n"
    return text.strip()

def extract_text_ocr(pdf_bytes):
    text = ""
    images = convert_from_bytes(pdf_bytes, dpi=300)
    for img in images:
        img_np = np.array(img).astype("float32")
        result = ocr.ocr(img_np)
        for line in result[0]:
            text += line[1][0] + " "
    return text.strip()

def extract_invoice_fields(full_text):
    data = {}
    clean_text = " ".join(full_text.split())

    for field, pattern in FIELDS.items():
        match = re.search(pattern, clean_text, re.IGNORECASE)
        data[field] = match.group(1).strip() if match else "null"

    return data

def extract_generic_key_values(text):
    pairs = {}
    lines = [line.strip() for line in text.split("\n") if line.strip()]

    for line in lines:
        match = re.match(r"^([A-Za-z][A-Za-z /]{2,})\s*[:\-]\s*(.+)$", line)
        if match:
            pairs[match.group(1).strip()] = match.group(2).strip()

    return pairs

if uploaded:
    pdf_bytes = uploaded.getvalue()

    full_text = extract_text_pdf(pdf_bytes)

    if not full_text:
        st.warning("📄 Scanned PDF detected → Running OCR")
        full_text = extract_text_ocr(pdf_bytes)

    if not full_text:
        st.error("❌ Unable to extract text from PDF")
        st.stop()
    st.text("If you want to see full Extracted details in Raw Formate")
    if st.button("Full Details"):
        with st.expander("🔎 View Full Extracted Text"):
            st.text(full_text)

    invoice_data = extract_invoice_fields(full_text)
    filtered_invoice_data = {
    k: v for k, v in invoice_data.items()
    if v and v.lower() != "null"
}
    st.success("Default Invoice Details")

    if filtered_invoice_data:
        df_invoice = pd.DataFrame(
            filtered_invoice_data.items(),
            columns=["Field", "Value"]
        )
        st.table(df_invoice)
    else:
        st.info("No default invoice fields detected")

# -----
    generic_kv = extract_generic_key_values(full_text)

    st.subheader("Additional Details Acc. to Invoice")

    if generic_kv:
        df_generic = pd.DataFrame(
            generic_kv.items(),
            columns=["Field", "Value"]
        )
        st.table(df_generic)
    else:
        st.warning("⚠ No key-value pairs detected")
