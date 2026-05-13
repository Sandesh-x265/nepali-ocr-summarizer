# 🇳🇵 Nepali OCR + Translation + Summarization

## 📌 Overview

This project is a **Nepali text processing pipeline** that extracts text from images and converts it into meaningful English summaries.

It combines OCR (Optical Character Recognition) with machine translation and text summarization to make Nepali text in images easier to understand and process.

---

## ⚙️ What It Does

Given an image containing Nepali text, the system:

- Extracts Nepali text using OCR (`pytesseract`)
- Translates the extracted text into English
- Generates a short summary of the translated text
- Displays results through a simple web interface (Gradio)

---

## 🧠 Pipeline

```
Image
↓
OCR (Nepali text extraction)
↓
English Translation
↓
Text Summarization
↓
Final Output (UI)
```

---

## 🛠 Tech Stack

- Python
- `pytesseract` (OCR)
- Hugging Face Transformers
- Gradio (UI)
- PyTorch

---

## 📂 Project Structure

```
images/
└── sample.jpg
src/
 ├── ocr_service.py
 ├── translation_service.py
 ├── summarization_service.py
 └── pipeline.py
app.py
requirements.txt
README.md
```

---

## 🚀 How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the app
```bash
python app.py
```

### 3. Open in browser
```
http://127.0.0.1:7860
```

---

## 📷 Input Example

Upload an image containing Nepali text (e.g. signboard, document, news clipping).

---

## 📤 Output

- Extracted Nepali text
- English translation
- Short summary
- Processing metrics

---

## 🎯 Purpose

This project demonstrates:

- NLP pipeline design
- OCR integration for low-resource languages (Nepali)
- End-to-end ML application development
- Model orchestration using Python services

---

## 📌 Notes

- Works best with clear printed Nepali text
- Handwritten text accuracy may vary
- First run may take time due to model downloads

---

## 👤 Author

**Sandesh**  
GitHub: [https://github.com/Sandesh-x265](https://github.com/Sandesh-x265)
```

