# AI Leadership Insights Agent

## Overview

This project enables leadership teams to ask natural-language questions about their organization and receive precise, document-grounded answers.

## Features

- **Document Ingestion:** Extracts and processes text from company PDFs.
- **Semantic Chunking:** Breaks long documents into meaningful, context-aware text segments.
- **Hybrid Information Retrieval:** Combines state-of-the-art vector search with keyword-based BM25 retrieval for maximum coverage.
- **LLM Reranking:** Uses a language model to rerank results and produce concise, grounded answers.
- **Source Linking:** Every answer references its originating document(s).
- **Interactive CLI:** Ask questions and get answers instantly from your leadership corpus.


## Folder Structure

```
AI-LEADERSHIP_INSIGHT_AGENT/
  ├── data/documents/
  ├── chroma_store/
  ├── ingestion/            
  ├── retrieval/            
  ├── generation/           
  ├── main.py               
  ├── config.py             
  ├── requirements.txt
  └── README.md             
```

---

## Quickstart

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

Edit `config.py` to add your OpenAI API key

### 3. Documents

Added documents related to Adobe dummy financial report.

### 4. Run the Agent

```bash
python main.py
```

You’ll see prompts to enter your leadership questions interactively.

---

## Limitations & Next Steps

- Only PDF documents are currently supported (expandable to other formats).
- No web UI/API yet—CLI only.
- Chunk metadata could be extended with more granular context (e.g., page numbers).
- No automated evaluation or traceability beyond citing document names.

---
