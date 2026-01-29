import streamlit as st
import pdfplumber
import io
import pandas as pd
import numpy as np
from pdf2image import convert_from_bytes
from paddleocr import PaddleOCR
import paddle
import re
import time
st.markdown("""
<style>

/* App background */
.stApp {
    background: linear-gradient(180deg, #FFFFFF, #C8DCFA);
}

/* Main title */
h1 {
    color: #1f2937;
    font-weight: 700;
}

/* Sub headers */
h2, h3 {
    color: #111827;
}

/* Sidebar styling */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #FFFFFF, #C8DCFA);
    color: white;
}

/* Sidebar text */
[data-testid="stSidebar"] * {
    color: white;
}

/* Upload box */
.stFileUploader {
    border: 2px dashed #2563eb;
    border-radius: 12px;
    padding: 10px;
    background-color: #0f172a;
}


/* Dataframe */
[data-testid="stDataFrame"] {
    border-radius: 12px;
    padding: 10px;
    overflow: hidden;
    background-color: #0f172a;

}
/* Metric cards */
.metric-card {
    background: green;
    padding: 18px;
    border-radius: 16px;
    box-shadow: 0 10px 25px rgba(0,0,0,0.08);
    text-align: center;
}

/* Section container */
.section-box {
    background: white;
    padding: 25px;
    border-radius: 18px;
    box-shadow: 0 15px 30px rgba(0,0,0,0.08);
    margin-bottom: 25px;
}

</style>
""", unsafe_allow_html=True)

# ---------------------

st.markdown("""
<style>
.card {
    background:white;
    border-radius:16px;
    padding:18px;
    box-shadow:0 10px 25px rgba(0,0,0,0.08);
    transition: all 0.3s ease;
}
.card:hover {
    transform: translateY(-4px);
    box-shadow:0 20px 40px rgba(0,0,0,0.12);
}
</style>

<div class="card">
<b>Invoice No:</b> INV-202510-33-3397
</div>
""", unsafe_allow_html=True)


# section box le liye
st.set_page_config(page_title=" Smart Document Parser", layout="wide")
st.markdown("""
<div class="section-box">
    <h1 style="color:#6950D3;">📄 Smart Document Data Extractor</h1>
    <p style="color:black;font-size:17px;">
        AI-powered system to extract structured invoice data using
        <b>PDF parsing + OCR + NLP</b>
    </p>
</div>
""", unsafe_allow_html=True)

# info box ke liye
st.markdown("""
<style>

/* Sidebar info alert dark */
[data-testid="stSidebar"] div[data-testid="stAlert"] {
    background-color: #0f172a !important;
    color: #ecfdf5 !important;
    border-radius: 12px;
    border: 1px solid #10b981;
}

/* Text inside alert */
[data-testid="stSidebar"] div[data-testid="stAlert"] p {
    color: #ecfdf5 !important;
}

/* Icon color */
[data-testid="stSidebar"] div[data-testid="stAlert"] svg {
    color: #34d399 !important;
}

</style>
""", unsafe_allow_html=True)


# st.sidebar.title("📄:violet[**Smart Invoice Parser**]")
st.sidebar.markdown(
    "<b style='color:#2563eb;font-size:18px;'>Smart Document Parser</b>",
    unsafe_allow_html=True
)
st.sidebar.markdown(":blue[AI Powered Document Data Extraction]")
st.sidebar.markdown("---")
st.sidebar.info("Upload PDF to extract structured data")
# file uploader ke liye

st.markdown("""
<style>

/* OUTER SIDEBAR FILE UPLOADER */
[data-testid="stSidebar"] [data-testid="stFileUploader"] {
    background: linear-gradient(145deg, #020617, #020617);
    border: 2px dashed #38bdf8;
    border-radius: 18px;
    padding: 16px;
    box-shadow: 0 0 0 rgba(56,189,248,0.0);
    transition: all 0.35s ease-in-out;
}

/* HOVER GLOW */
[data-testid="stSidebar"] [data-testid="stFileUploader"]:hover {
    box-shadow: 0 0 18px rgba(56,189,248,0.55);
    border-color: #7dd3fc;
}

/* INNER DROP ZONE (WHITE AREA FIX) */
[data-testid="stSidebar"] [data-testid="stFileUploader"] section {
    background-color: #020617 !important;
    border-radius: 14px;
    padding: 18px;
    transition: background 0.3s ease;
}

/* TEXT INSIDE DROP ZONE */
[data-testid="stSidebar"] [data-testid="stFileUploader"] section div {
    color: #e5e7eb !important;
    font-weight: 500;
}

/* BROWSE FILE BUTTON */
[data-testid="stSidebar"] [data-testid="stFileUploader"] button {
    background: linear-gradient(135deg, #2563eb, #38bdf8) !important;
    color: white !important;
    border-radius: 10px;
    padding: 6px 14px;
    border: none;
    transition: all 0.3s ease;
}

/* BUTTON HOVER */
[data-testid="stSidebar"] [data-testid="stFileUploader"] button:hover {
    transform: scale(1.05);
    background: linear-gradient(135deg, #1d4ed8, #0ea5e9) !important;
}

/* REMOVE DEFAULT WHITE BACK */
[data-testid="stSidebar"] [data-testid="stFileUploader"] > div > div {
    background-color: transparent !important;
}

</style>
""", unsafe_allow_html=True)


uploaded = st.sidebar.file_uploader(
    "📤:blue[ Upload PDF]",
    type=["pdf"]
)


paddle.set_flags({'FLAGS_use_mkldnn': False})

ocr = PaddleOCR(lang="en", use_textline_orientation=True)

FIELDS = {
    "GSTIN": r"GSTIN\s*[:\-]?\s*([A-Z0-9]{15})",
    # "Invoice No.": r"(?i)\binvoice\s*no\.?\s*[:\-–]\s*([A-Z0-9\-\/]+)",
    "Invoice No.": r"(?i)\binvoice\s*(?:no\.?|number|#)\s*[:\-–]?\s*([A-Z0-9\-\/]+)",
    # "Invoice No .": r"(?i)\b(?:invoice|inv|bill)\b\s*(?:no.\.?|number|#)?\s*[:\-–]?\s*([A-Z0-9\/\-]+)",
    "Invoice Date": r"(?i)\b(?:date|dated|invoice\s*date)\b\s*[:\-–]?\s*(\d{1,4}[\/\-.]\d{1,2}[\/\-.]\d{2,4})",
    "Customer Name": r"(?im)\b(?:customer\s*name|to|bill\s*to)\b\s*:?\s*(?:\n\s*)?([A-Z][A-Za-z&.\s]{2,})",
    "Phone": r"(?i)\b(?:contact|phone|mobile|tel)\b\s*[:\-–]?\s*((?:\+?\d[\d\s\-]{8,14}\d))",
    "pod_pattern": r'POD\s*[:\-]\s*(?:[A-Z]{3,5}\s*[-–]\s*)?([A-Za-z\s]+)',
    "POL": r'POL\s*[:\-]\s*[A-Z]{3,5}\s*[-–]\s*([A-Za-z ]+)'


}
# (?i)\b(?:invoice|inv|bill)\b\s*(?:no.\.?|number|#)?\s*[:\-–]?\s*([A-Z0-9\/\-]+)
STANDARD_FIELDS = {
    "line_no": ["s.no", "sr", "no", "#"],
    "description": ["description", "particular", "item"],
    "hsn_sac": ["hsn", "sac"],
    "quantity": ["qty", "quantity"],
    "unit_price": ["rate", "unit", "price"],
    "taxable_amount": ["taxable", "tax"],
    "cgst": ["cgst"],
    "sgst": ["sgst", "sgst utgst", "sgst/utgst", "utgst"],
    "igst": ["igst"],
    "total_amount": ["total", "amount", "value", "TOTAL PRICE (INR)", "price"],
   
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

def extract_invoice_fields(text):
    data = {}
    clean_text = " ".join(text.split())
    for field, pattern in FIELDS.items():
        match = re.search(pattern, clean_text, re.IGNORECASE)
        data[field] = match.group(1).strip() if match else ""
    return data

def clean(val):
    if not val:
        return None
    return re.sub(r"\s+", " ", val).strip()

def is_valid_line_no(val):
    return bool(val and re.fullmatch(r"\d+", val))

def is_table_end(text):
    stop_words = ["grand total","amount in words","authorised","bank","branch","ifsc"]
    return any(w in text.lower() for w in stop_words)

def extract_table_rows(pdf_bytes):
    rows_data = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            words = page.extract_words(use_text_flow=True)
            rows = {}
            for w in words:
                y = int(w["top"] // 8)
                rows.setdefault(y, []).append(w)
            rows = dict(sorted(rows.items()))
            header = None
            column_x = {}
            for row in rows.values():
                row_text = " ".join(w["text"].lower() for w in row)
                if any(k in row_text for k in ["description", "particular", "item"]) and any(k in row_text for k in ["amount", "total", "value", "amt"]):
                    header = row
                    break
            if not header:
                continue
            for w in header:
                t = w["text"].lower()
                for col, keys in STANDARD_FIELDS.items():
                    if any(k in t for k in keys):
                        column_x[col] = w["x0"]
            header_y = int(header[0]["top"] // 8)
            for y, row in rows.items():
                if y <= header_y:
                    continue
                row_text = " ".join(w["text"] for w in row)
                if is_table_end(row_text):
                    break
                item = {}
                for col, x in column_x.items():
                    nearest, dist = None, 999
                    for w in row:
                        d = abs(w["x0"] - x)
                        if d < dist and d < 60:
                            nearest = w["text"]
                            dist = d
                    val = clean(nearest)
                    if val:
                        item[col] = val
                if "description" not in item:
                    continue
                if "line_no" in item and not is_valid_line_no(item["line_no"]):
                    continue
                rows_data.append(item)
    return rows_data

import base64

def show_pdf(pdf_bytes):
    base64_pdf = base64.b64encode(pdf_bytes).decode("utf-8")
    pdf_display =f"""
    <iframe
        src="data:application/pdf;base64,{base64_pdf}"
        width="100%"
        height="700"
        style="border: none; border-radius: 12px;"
    ></iframe>
    """
    st.markdown(pdf_display, unsafe_allow_html=True)

# har invoice or resume ke hisab se alag alag informaton
def extract_generic_key_values(text):
    pairs = {}
    
    lines = [line.strip() for line in text.split("\n") if line.strip()]

    i = 0
    while i < len(lines):

        line = lines[i]

        match = re.match(r"^([A-Za-z][A-Za-z /]{2,})\s*[:]\s*(.+)$", line)
        if match:
            pairs[match.group(1).strip()] = match.group(2).strip()
            i += 1
            continue

        match = re.match(r"^([A-Za-z][A-Za-z /]{2,})\s*[-]\s*(.+)$", line)
        if match:
            pairs[match.group(1).strip()] = match.group(2).strip()
            i += 1
            continue

        match = re.match(r"^([A-Za-z][A-Za-z /]{2,})\s{2,}(.+)$", line)
        if match:
            pairs[match.group(1).strip()] = match.group(2).strip()
            i += 1
            continue

        i += 1

    return pairs

if uploaded:
    with st.spinner("🔍 Analyzing document using OCR & NLP..."):
        time.sleep(2)

    pdf_bytes = uploaded.getvalue()

    col1, col2 = st.columns([1.2,1])

    with col1:
        st.subheader("📄:blue[PDF Preview]")
        show_pdf(pdf_bytes)

    with col2:

        text = extract_text_pdf(pdf_bytes)
        if not text:
            st.warning("Scanned PDF detected → Running OCR")
            text = extract_text_ocr(pdf_bytes)
        if not text:
            st.error("❌ Unable to extract text")
            st.stop()

        invoice_data = extract_invoice_fields(text)
        st.subheader("📑 :blue[Summary]")
        summary_df = pd.DataFrame(list(invoice_data.items()), columns=["Field", "Value"]).replace("", np.nan).dropna()
        st.dataframe(summary_df, use_container_width=True)

        rows = extract_table_rows(pdf_bytes)
        st.subheader("📄:blue[Line Items]")
        if rows:
            for i, row in enumerate(rows, start=1):
                st.markdown(f"**Item {i}**")
                row_df = pd.DataFrame(list(row.items()), columns=["Field", "Value"])
                st.dataframe(row_df, use_container_width=True)
        else:
            st.warning("❌ No valid invoice rows found")


    

    used_fields = set()
    # used_values = set()
    for _, row in summary_df.iterrows():
        used_fields.add(str(row["Field"]).lower())
        used_fields.add(str(row["Value"]).lower())

    for row in rows:
        for k, v in row.items():
            used_fields.add(str(k).lower())
            used_fields.add(str(v).lower())

    generic_kv = extract_generic_key_values(text)

    filtered_kv = {
        k: v
        for k, v in generic_kv.items()
        if k.lower() not in used_fields
        and v.lower() not in used_fields
    }

    st.subheader("🧠 Extra Details of Documents..")

    if filtered_kv:
        default_df = pd.DataFrame(filtered_kv.items(), columns=["Field", "Value"])
        st.dataframe(default_df, use_container_width=True)
    else:
        st.success("✅ All key fields & values already captured in Summary / Line Items")

