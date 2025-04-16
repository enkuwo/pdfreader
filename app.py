from flask import Flask, request, render_template, send_file
import fitz  # PyMuPDF
import requests
import re
from collections import Counter
from langdetect import detect
from pdf2image import convert_from_bytes
import pytesseract
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:/Program Files/Tesseract-OCR/tesseract.exe"

from PIL import Image
from fpdf import FPDF
from docx import Document
import io
import unicodedata
app = Flask(__name__)

# Hugging Face API keys
HEADERS = {"Authorization": "Bearer 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"}

# Stopwords
STOPWORDS = {
    "the", "and", "to", "of", "in", "a", "is", "it", "that", "on", "for", "with", "as", "was",
    "are", "this", "an", "by", "be", "from", "at", "or", "which", "you", "not", "but", "we"
}

# --- Extraction Utilities ---

def extract_text_with_ocr(pdf_bytes):
    images = convert_from_bytes(pdf_bytes.read())
    text_pages = [pytesseract.image_to_string(img) for img in images]
    return text_pages

def extract_text_from_pdf(pdf):
    doc = fitz.open(stream=pdf.read(), filetype="pdf")
    return [page.get_text() for page in doc]

def detect_language(text_pages):
    full_text = " ".join(text_pages)
    try:
        return detect(full_text)
    except:
        return "en"

# --- Summarization Utilities ---

def summarize_text(text, lang="en"):
    if lang == "en":
        model_url = "https://api-inference.huggingface.co/models/google/pegasus-cnn_dailymail"
    else:
        model_url = "https://api-inference.huggingface.co/models/facebook/mbart-large-cc25"

    payload = {"inputs": text[:1000]}  # Truncate if too long
    response = requests.post(model_url, headers=HEADERS, json=payload)

    if response.status_code == 200:
        return response.json()[0]["summary_text"]
    return f"(Summary error: {response.text})"

def summarize_full(text_pages, lang):
    return summarize_text(" ".join(text_pages), lang)

def summarize_by_page(text_pages, lang):
    return "\n".join(
        f"Page {i+1}:\n{summarize_text(page, lang)}\n"
        for i, page in enumerate(text_pages) if len(page.strip()) > 50
    )

def summarize_in_groups(text_pages, lang, group_size=3):
    summaries = []
    for i in range(0, len(text_pages), group_size):
        chunk = " ".join(text_pages[i:i+group_size])
        if len(chunk.strip()) > 50:
            summaries.append(f"Pages {i+1}-{min(i+group_size, len(text_pages))}:\n{summarize_text(chunk, lang)}\n")
    return "\n".join(summaries)

def extract_numbers(text_pages):
    numbers = re.findall(r'\b\d+(?:\.\d+)?\b', " ".join(text_pages))
    return "Found numbers:\n" + ", ".join(numbers)

def search_and_summarize_by_keyword(text_pages, keyword, lang):
    summaries = []
    keyword = keyword.lower()
    for i, page_text in enumerate(text_pages):
        if keyword in page_text.lower():
            summaries.append(f"Page {i+1}:\n{summarize_text(page_text, lang)}\n")
    return "\n".join(summaries) if summaries else f"No results found for '{keyword}'."

def find_most_frequent_keyword(text_pages, lang):
    full_text = " ".join(text_pages).lower()
    words = re.findall(r'\b[a-zA-Z]{3,}\b', full_text)
    filtered = [word for word in words if word not in STOPWORDS]

    if not filtered:
        return "(No valid keywords found.)"

    top_keyword, _ = Counter(filtered).most_common(1)[0]
    summaries = [
        f"Page {i+1}:\n{summarize_text(page, lang)}\n"
        for i, page in enumerate(text_pages) if top_keyword in page.lower()
    ]

    return f"Most frequent keyword: **{top_keyword}**\n\n" + "\n".join(summaries)

# --- Export Utilities ---

def remove_non_latin(text):
    return unicodedata.normalize('NFKD', text).encode('latin-1', 'ignore').decode('latin-1')

def export_summary_to_pdf(summary_text):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", size=12)
    
    clean_text = remove_non_latin(summary_text)
    
    for line in clean_text.split('\n'):
        pdf.multi_cell(0, 10, line)
    
    output = io.BytesIO()
    pdf.output(output)
    output.seek(0)
    return output

def export_summary_to_word(summary_text):
    doc = Document()
    doc.add_heading('PDF Summary', 0)
    for line in summary_text.split('\n'):
        doc.add_paragraph(line)
    output = io.BytesIO()
    doc.save(output)
    output.seek(0)
    return output

# --- Flask Routes ---

@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        pdf = request.files["file"]
        mode = request.form.get("mode")
        keyword = request.form.get("keyword", "").strip()

        if not pdf:
            return render_template("upload.html", error="Please upload a PDF.")

        # Try extracting text first
        try:
            text_pages = extract_text_from_pdf(pdf)
            if not any(text.strip() for text in text_pages):
                raise ValueError("Empty text, using OCR")
        except:
            pdf.seek(0)
            text_pages = extract_text_with_ocr(pdf)

        lang = detect_language(text_pages)

        if mode == "full":
            summary = summarize_full(text_pages, lang)
        elif mode == "page":
            summary = summarize_by_page(text_pages, lang)
        elif mode == "group":
            summary = summarize_in_groups(text_pages, lang)
        elif mode == "numbers":
            summary = extract_numbers(text_pages)
        elif mode == "keyword":
            if not keyword:
                return render_template("upload.html", error="Please enter a keyword.")
            summary = search_and_summarize_by_keyword(text_pages, keyword, lang)
        elif mode == "most_used":
            summary = find_most_frequent_keyword(text_pages, lang)
        else:
            summary = "(Invalid mode selected.)"

        return render_template("result.html", summary=summary)

    return render_template("upload.html")

@app.route("/download", methods=["POST"])
def download_summary():
    summary = request.form.get("summary")
    file_type = request.form.get("filetype", "pdf")

    if file_type == "word":
        output = export_summary_to_word(summary)
        return send_file(output, as_attachment=True, download_name="summary.docx", mimetype="application/vnd.openxmlformats-officedocument.wordprocessingml.document")
    else:
        output = export_summary_to_pdf(summary)
        return send_file(output, as_attachment=True, download_name="summary.pdf", mimetype="application/pdf")

# --- Run the App ---
if __name__ == "__main__":
    app.run(debug=True)