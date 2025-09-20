import os
import re
import json
import datetime
import unicodedata
import pdfplumber
import docx
import logging
import asyncio
from dotenv import load_dotenv
from groq import Groq
import langdetect

# ---------------- DeepTranslator Imports ----------------
from deep_translator import MyMemoryTranslator, GoogleTranslator

# ---------- Load environment ----------
load_dotenv()
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

# ---------- Logging ----------
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s: %(message)s")

# ---------- Corpus Processor Class ----------
class CorpusProcessor:
    def __init__(self, groq_api_key=GROQ_API_KEY, fallback_lang="en-ZA"):
        self.groq_client = Groq(api_key=groq_api_key)
        self.fallback_lang = fallback_lang # This implies en-ZA is a preferred English variant

        # Initialize MyMemoryTranslator for English <-> Zulu
        # Using specific language codes as required by MyMemoryTranslator
        # From the error list: 'english south africa': 'en-ZA' and 'zulu': 'zu-ZA'
        self.mymemory_en_zu = MyMemoryTranslator(source='en-ZA', target='zu-ZA')
        self.mymemory_zu_en = MyMemoryTranslator(source='zu-ZA', target='en-ZA')
        
        # Google Translator can usually handle generic 'en' and 'zu', but we'll keep
        # the specific codes for consistency if needed, though 'en' and 'zu' are fine here.
        self.google_translator = GoogleTranslator() 

    # The rest of your class methods remain the same
    # ...
    # ---------------- Text Extraction ----------------
    def extract_text(self, filename):
        logging.info(f"Extracting text from {filename}")
        text = []
        if filename.endswith(".pdf"):
            with pdfplumber.open(filename) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text.append(page_text)
        elif filename.endswith(".docx"):
            doc = docx.Document(filename)
            text = [para.text for para in doc.paragraphs if para.text.strip()]
        else:
            raise ValueError(f"Unsupported file type: {filename}")
        return "\n".join(text)

    # ---------------- Text Cleaning ----------------
    @staticmethod
    def clean_text(text):
        text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
        text = re.sub(r'(OKUQUKETHWE|TABLE OF CONTENTS|CONTENTS|IZITHUNGO|INTRODUCTION|IMIQULU).*', '', text, flags=re.IGNORECASE)
        text = re.sub(r'[\t\r\f\v]+', ' ', text)
        text = re.sub(r'\n\s*\n', '\n', text)
        text = re.sub(r'\s{2,}', ' ', text)
        return text.strip()

    @staticmethod
    def normalize_text(text):
        return unicodedata.normalize("NFKD", text)

    @staticmethod
    def split_sentences(text):
        sentences = re.split(r'(?<=[.!?])\s+|\n', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        sentences = [s for s in sentences if len(s) > 10 and re.search(r'[a-zA-Z0-9]', s)]
        return sentences

    # ---------------- Language Detection ----------------
    @staticmethod
    def get_language(text):
        try:
            code = langdetect.detect(text)
            if code == "zu":
                return "zulu"
            elif code == "en":
                return "english"
            else:
                return "other"
        except Exception:
            return "other"

    # ---------------- Translation (Async, DeepTranslator) ----------------
    async def _translate_sentence(self, sentence, source_lang):
        translation = {"english": "", "zulu": ""}
        # Use MyMemoryTranslator's specific source/target settings or fallback to GoogleTranslator
        mymemory_source_lang = 'zu-ZA' if source_lang == 'zulu' else 'en-ZA'
        mymemory_target_lang = 'en-ZA' if source_lang == 'zulu' else 'zu-ZA'

        try:
            if source_lang == "zulu":
                translation["zulu"] = sentence
                # Zulu -> English using MyMemory
                translated_text = self.mymemory_zu_en.translate(sentence)
                if not translated_text: # Fallback if MyMemory fails or returns empty
                    translated_text = self.google_translator.translate(sentence, source='zu', target='en')
                translation["english"] = translated_text
            elif source_lang == "english":
                translation["english"] = sentence
                # English -> Zulu using MyMemory
                translated_text = self.mymemory_en_zu.translate(sentence)
                if not translated_text: # Fallback if MyMemory fails or returns empty
                    translated_text = self.google_translator.translate(sentence, source='en', target='zu')
                translation["zulu"] = translated_text
            else: # If language is not English or Zulu, just keep the original sentence
                translation["english"] = sentence
                translation["zulu"] = sentence # Best effort, might not be accurate
        except Exception as e:
            logging.warning(f"Translation failed for sentence: '{sentence}' ({source_lang}) -> {e}. Using original.")
            translation["english"] = sentence
            translation["zulu"] = sentence
        return translation

    async def translate_batch(self, sentences, source_lang, batch_size=10):
        results = []
        total = min(len(sentences), 300)  # Test mode: limit to 100 sentences
        for i in range(0, total, batch_size):
            batch = sentences[i:i + batch_size]
            batch_results = await asyncio.gather(*(self._translate_sentence(s, source_lang) for s in batch))
            results.extend(batch_results)
            logging.info(f"Translated {min(i+batch_size, total)} / {total} sentences")
        return results

    # ---------------- Classification (Async) ----------------
    async def _classify_sentence(self, sentence):
        prompt = """
        Classify each sentence into one cultural domain (Ubuntombi, Ukushada, Ukubingelela,
        Ukuvakashela, Izaga & Izisho, Ubudlelwano, Imicimbi, Ukunxiba & Ubuhle, Ukudla,
        Umculo & Umhubhe, Ezemfundo, Ezempilo, Ezomnotho, Ezesizwe, Ezemvelo, Okujwayelekile).
        Only return the domain name.
        """
        try:
            completion = self.groq_client.chat.completions.create(
                model="openai/gpt-oss-20b", # Using a more suitable Groq model if available
                messages=[{"role": "user", "content": f"{prompt}\nSentence: {sentence}"}],
                temperature=0,
                max_tokens=100
            )
            return completion.choices[0].message.content.strip()
        except Exception as e:
            logging.error(f"Classification failed: {e}")
            return "Okujwayelekile"

    async def classify_batch(self, sentences, batch_size=10):
        results = []
        total = len(sentences)
        for i in range(0, total, batch_size):
            batch = sentences[i:i + batch_size]
            batch_results = await asyncio.gather(*(self._classify_sentence(s) for s in batch))
            results.extend(batch_results)
            logging.info(f"Classified {min(i+batch_size, total)} / {total} sentences")
        return results

    # ---------------- Process File ----------------
    async def process_file(self, filename):
        logging.info(f"Processing file: {filename}")
        raw = self.extract_text(filename)
        cleaned = self.clean_text(raw)
        normalized = self.normalize_text(cleaned)
        sentences = self.split_sentences(normalized)
        detected_lang = self.get_language(normalized)
        logging.info(f"Detected language: {detected_lang}, Total sentences: {len(sentences)}")

        # Limit sentences for translation and classification to the first 100 for testing
        sentences_for_processing = sentences[:300] # Apply the limit here

        translations = await self.translate_batch(sentences_for_processing, detected_lang)
        domains = await self.classify_batch(sentences_for_processing) # Apply the limit here too!

        entries = []
        # Ensure that translations and domains lists are not shorter than sentences after slicing
        num_to_process = min(len(sentences_for_processing), len(translations), len(domains))

        for i in range(num_to_process):
            entry = {
                "zulu": translations[i].get("zulu", ""),
                "english": translations[i].get("english", ""),
                "translations": translations[i],
                "domain": domains[i],
                "original_sentence": sentences_for_processing[i], # Use the limited list here
                "source_language_detected": detected_lang,
                "origin_filename": os.path.basename(filename),
                "created_at": datetime.datetime.utcnow().isoformat(),
                "added_by": "batch_import"
            }
            entries.append(entry)
        logging.info(f"Completed processing {len(entries)} sentences from {filename}")
        return entries
    
    # ---------------- Optional: Process multiple files ----------------
    async def process_folder(self, folder_path):
        all_entries = []
        for file in os.listdir(folder_path):
            if file.endswith((".pdf", ".docx")):
                entries = await self.process_file(os.path.join(folder_path, file))
                all_entries.extend(entries)
        return all_entries


# ---------------- Example usage ----------------
if __name__ == "__main__":
    processor = CorpusProcessor()
    input_files = ["The Role of Old Women in Zulu Culture - M Brindley (1).pdf"] 

    async def main():
        all_entries = []
        for f in input_files:
            entries = await processor.process_file(f)
            all_entries.extend(entries)

        with open("corpus_export.json", "w", encoding="utf-8") as f:
            json.dump(all_entries, f, ensure_ascii=False, indent=2)
        logging.info(f"Exported {len(all_entries)} entries to corpus_export.json")

    asyncio.run(main())