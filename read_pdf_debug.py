
import os
import sys

def extract_text(pdf_path):
    text = ""
    try:
        import pypdf
        print(f"Using pypdf for {pdf_path}")
        reader = pypdf.PdfReader(pdf_path)
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    except ImportError:
        pass

    try:
        import PyPDF2
        print(f"Using PyPDF2 for {pdf_path}")
        with open(pdf_path, 'rb') as f:
            reader = PyPDF2.PdfFileReader(f)
            for page_num in range(reader.numPages):
                page = reader.getPage(page_num)
                text += page.extractText() + "\n"
        return text
    except ImportError:
        pass
        
    try:
        import pdfplumber
        print(f"Using pdfplumber for {pdf_path}")
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text() + "\n"
        return text
    except ImportError:
        pass

    print("No suitable PDF library found (pypdf, PyPDF2, pdfplumber).")
    return None

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    docs_dir = os.path.join(base_dir, 'docs')
    
    files_to_read = ['2026_MCM_Problem_C.pdf', '终稿.pdf']
    output_file = os.path.join(docs_dir, 'extracted_content.txt')
    
    with open(output_file, 'w', encoding='utf-8') as out:
        for fname in files_to_read:
            path = os.path.join(docs_dir, fname)
            if os.path.exists(path):
                print(f"Reading {fname}...")
                content = extract_text(path)
                if content:
                    out.write(f"=== START OF {fname} ===\n")
                    out.write(content)
                    out.write(f"\n=== END OF {fname} ===\n\n")
                else:
                    print(f"Failed to extract text from {fname}")
            else:
                print(f"File not found: {path}")

if __name__ == "__main__":
    main()
