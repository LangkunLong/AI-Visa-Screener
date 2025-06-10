# Visa Screener API

This project implements an AI-powered visa screening tool that evaluates CVs against the O-1A visa qualification criteria. It uses a RESTful API (built with FastAPI) to process uploaded resumes and return a qualification rating (`low`, `medium`, or `high`) based on evidence extracted using a transformer-based Named Entity Recognition (NER) model.

---

## 🧠 Overview

The system performs the following steps:
1. Accepts a CV upload via API
2. Extracts and preprocesses text from the document
3. Applies Named Entity Recognition to identify relevant qualifications
4. Evaluates how many O-1A criteria the candidate meets
5. Returns a structured JSON response with the overall rating and supporting evidence

---

## 🔍 O-1A Criteria Evaluated

The following 8 qualification categories are extracted and scored:
- Awards
- Membership in associations
- Press coverage
- Judging work
- Original contributions
- Scholarly articles
- Critical employment
- High salary

Each category is scored equally in the current implementation.

---

## 🧪 Evaluation Strategy

Based on how many criteria are matched:
- **Low**: Fewer than 4 criteria matched
- **Medium**: 4–5 criteria matched
- **High**: 6 or more criteria matched

Entity recognition is performed using the [`bert-large-cased-finetuned-conll03-english`](https://huggingface.co/dbmdz/bert-large-cased-finetuned-conll03-english) model.

---

## ⚙️ Tech Stack

- **Backend**: FastAPI
- **Model**: BERT (transformer for NER)
- **Preprocessing**: PDF text extraction, stopword removal, text normalization

---

## 📈 Planned Improvements

- 🔧 Fine-tune the BERT model on visa-specific examples
- ⚖️ Learn optimal weights for each criterion based on historical acceptance data
- 🤖 Integrate additional models (e.g., E5) for name/entity disambiguation
- 📊 Replace hard thresholds with learned classification models
- 📄 Expand input format support (e.g., .docx) and add robust error handling
- ☁️ Deploy model inference on GPU-backed cloud infrastructure for scalability

---

## 📁 Example Response
```json
{
  "score": "medium",
  "criteria_matched": ["awards", "press", "original contributions", "judging"],
  "evidence": {
    "awards": ["Best Research Paper 2022"],
    "press": ["Interviewed in TechCrunch"],
    ...
  }
}
