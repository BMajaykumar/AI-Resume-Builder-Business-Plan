import PyPDF2

def load_pdf(file_obj):
    text = ""
    reader = PyPDF2.PdfReader(file_obj)
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text
    return text
