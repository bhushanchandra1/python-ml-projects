import fitz  # PyMuPDF
import docx
from io import BytesIO

def extract_text_from_pdf(uploaded_file) -> str:
    text = ""
    file_bytes = BytesIO(uploaded_file.read())
    doc = fitz.open(stream=file_bytes.read(), filetype="pdf")
    for page in doc:
        text += page.get_text()
    return text

def extract_text_from_docx(uploaded_file) -> str:
    file_bytes = BytesIO(uploaded_file.read())
    doc = docx.Document(file_bytes)
    return "\n".join([p.text for p in doc.paragraphs])
