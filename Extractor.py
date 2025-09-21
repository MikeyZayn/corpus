#THis is standalone as per the lecturers requirements you just need to install the dependancies.
"""THese are the dependancies:
# PDF extraction
pymupdf>=1.23.0

# Hugging Face + Transformers
transformers>=4.44.0
sentencepiece>=0.1.99
sacremoses>=0.0.53

# Language detection
langdetect>=1.0.9

# Core dependencies
torch>=2.0.0
"""

#Install them and please start testing.

import re
import sys
import fitz  # PyMuPDF for PDF reading

class Extractor:
    def __init__(self, input_file, output_file, sentences_per_page=100):
        self.input_file = input_file
        self.output_file = output_file
        self.sentences_per_page = sentences_per_page

    def extract_text_from_pdf(self):
        text = ""
        with fitz.open(self.input_file) as doc:
            for page in doc:
                text += page.get_text("text") + "\n"
        return text

    def clean_text(self, text):
        # Fix spacing and ligatures
        text = re.sub(r"\s+", " ", text)  
        text = re.sub(r"([a-z])([A-Z])", r"\1. \2", text)  # lowercase->uppercase = sentence break
        text = text.replace("ﬁ", "fi").replace("ﬂ", "fl")
        return text.strip()

    def split_sentences(self, text):
        # Split on ., ?, ! followed by space or end of line
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    def group_into_pages(self, sentences):
        pages = []
        for i in range(0, len(sentences), self.sentences_per_page):
            pages.append(sentences[i:i+self.sentences_per_page])
        return pages

    def save_output(self, pages):
        with open(self.output_file, "w", encoding="utf-8") as f:
            for idx, page in enumerate(pages, 1):
                f.write(f"=== Page {idx} ===\n")
                for s in page:
                    f.write(s + "\n")
                f.write("\n")

    def extract(self):
        print(f"Extracting text from: {self.input_file}")
        raw_text = self.extract_text_from_pdf()
        clean = self.clean_text(raw_text)
        sentences = self.split_sentences(clean)
        pages = self.group_into_pages(sentences)
        self.save_output(pages)
        print(f"Done! Output saved to {self.output_file}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python Extractor.py input.pdf output.txt [sentences_per_page]")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    sentences_per_page = int(sys.argv[3]) if len(sys.argv) > 3 else 100

    extractor = Extractor(input_file, output_file, sentences_per_page)
    extractor.extract()

