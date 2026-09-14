# Solve the Case — An Interactive Agentic RAG Investigation

**Solve the Case** is an interactive multi-document investigation system that uses **Retrieval-Augmented Generation (RAG)**, **LLM agents**, and an **evidence graph** to investigate a real-world-style case from a public news corpus.

The system is designed to make AI-assisted investigation **evidence-grounded, inspectable, and resistant to unsupported claims**.

---

## Overview

The user is presented with an investigation case containing a set of candidate entities and supporting documents.

Instead of directly asking an LLM to answer the case, the system builds an evidence pipeline:

```text
News Corpus
    ↓
Preprocessing
    ↓
Hybrid Retrieval
(BM25 + Semantic Embeddings)
    ↓
Investigator Agent
    ↓
Evidence Graph
    ↓
Fact-Checker Agent
    ↓
Citation Validation
    ↓
Final Verdict
```

The system also allows the investigator to make an initial prediction and compare it against the final AI-assisted investigation.

---

## Key Features

* **Hybrid evidence retrieval**

  * BM25 lexical retrieval
  * SentenceTransformer semantic retrieval
  * Combined hybrid ranking

* **Agentic investigation**

  * Investigator Agent performs bounded investigation and query reformulation
  * Fact-Checker Agent performs adversarial evidence searches

* **Evidence grounding**

  * Agent responses are tied to retrieved document evidence
  * Deterministic citation validation checks referenced evidence

* **Verified vs unverified evidence**

  * Evidence can be explicitly marked as verified or unverified
  * Unverified claims are prevented from being treated as confirmed facts

* **Evidence graph**

  * NetworkX graph representing entities and relationships extracted from the corpus
  * Graph context can be inspected during investigation

* **Candidate interrogation**

  * Candidates can be individually interrogated using the investigation pipeline

* **Interactive investigation**

  * Users can inspect evidence, investigate candidates, make predictions, and submit a final verdict

* **Execution traces**

  * Investigation and fact-checking steps can be inspected to understand how the system reached its conclusions

---

## Case

The primary demonstration case is:

### The Afghanistan Security Network

The case investigates a set of candidate entities in connection with events described across multiple news documents.

### Candidates

* Taliban
* Zabiullah Mujahid
* NATO
* Hamid Karzai

The investigation uses four selected DWIE documents as the primary case corpus.

An intentionally unverified social-media claim is also included as a reliability test. The system explicitly marks this claim as unverified rather than allowing it to become a confirmed fact merely because an LLM can generate a plausible explanation around it.

---

## Dataset

The project uses the **DWIE (Deutsche Welle Information Extraction) dataset**, a public multi-document news corpus containing annotated entities, mentions, relations, and article content.

The raw dataset is used locally during preprocessing and is **not committed to this repository**.

The repository contains the processed runtime artifacts required by the deployed application.

---

## Retrieval Pipeline

The retrieval system combines two complementary approaches.

### 1. BM25 Retrieval

BM25 provides lexical retrieval based on term matching between the investigation query and document chunks.

This is useful when the query contains specific names, organisations, locations, or phrases appearing directly in the source documents.

### 2. Semantic Retrieval

The project uses:

```text
sentence-transformers/all-MiniLM-L6-v2
```

to generate dense embeddings for document chunks.

Queries are embedded using the same model, and semantic similarity is used to retrieve conceptually related evidence even when exact words do not match.

### 3. Hybrid Retrieval

The final retrieval process combines lexical and semantic evidence to improve robustness across different query types.

---

## Agent Architecture

### Investigator Agent

The Investigator Agent is responsible for conducting the main investigation.

It can:

1. Interpret the investigation objective
2. Generate evidence queries
3. Retrieve relevant documents
4. Inspect retrieved evidence
5. Reformulate queries when necessary
6. Build an evidence-backed investigation result

The investigation is bounded to prevent uncontrolled agent execution.

### Fact-Checker Agent

The Fact-Checker Agent acts as an adversarial second pass.

Rather than simply accepting the Investigator's conclusions, it searches for evidence that may:

* contradict a claim
* weaken a conclusion
* provide an alternative explanation
* reveal unsupported assumptions

This makes the system less dependent on a single LLM reasoning pass.

---

## Evidence Graph

The system constructs an evidence graph using **NetworkX**.

The graph represents relationships between entities and other extracted information from the corpus.

This allows the investigation to be viewed not only as a collection of retrieved text snippets, but also as a connected network of entities and relationships.

The frontend exposes graph statistics and graph/neighborhood information for inspection.

---

## Reliability and Citation Validation

A central design goal is to prevent the LLM from presenting unsupported claims as established facts.

The system therefore includes:

* source-backed evidence retrieval
* citation validation
* explicit verified/unverified evidence status
* adversarial fact checking
* deterministic validation of generated citations

The system does **not** assume that an LLM response is automatically factual simply because it sounds convincing.

---

## Project Structure

```text
Solve-the-Case/
│
├── backend/
│   ├── case_manager.py
│   ├── citations_validator.py
│   ├── evidence_graph.py
│   ├── fact_checker.py
│   ├── findclusters.py
│   ├── inspect_dwie.py
│   ├── interrogator.py
│   ├── investigator.py
│   ├── llm_client.py
│   ├── main.py
│   ├── preprocess.py
│   ├── retrieval.py
│   ├── schemas.py
│   └── verdict.py
│
├── cases/
│   └── case_001.json
│
├── data/
│   └── processed/
│       └── processed runtime evidence artifacts
│
├── frontend/
│   └── app.py
│
├── requirements.txt
├── .gitignore
└── README.md
```

The raw DWIE dataset is intentionally excluded from Git using `.gitignore`.

---

## Local Setup

### 1. Clone the repository

```bash
git clone https://github.com/TanzilaFirdousP/Solve-the-Case.git
cd Solve-the-Case
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_api_key_here
GEMINI_MODEL=gemini-3.6-flash
```

**Never commit `.env` or expose the API key publicly.**

### 5. Start the FastAPI backend

```powershell
python -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

The backend will be available at:

```text
http://127.0.0.1:8000
```

Health check:

```text
http://127.0.0.1:8000/health
```

### 6. Start the Streamlit frontend

In a second terminal:

```powershell
streamlit run frontend/app.py
```

The Streamlit interface will then provide the interactive investigation workflow.

---

## API

The FastAPI backend exposes endpoints for the major investigation operations, including:

```text
GET  /health
GET  /cases
POST /predict
POST /search_evidence
POST /interrogate
POST /investigate
POST /fact_check
POST /submit_verdict
GET  /graph/stats
GET  /graph
GET  /graph/neighborhood
```

The API provides the interface between the Streamlit frontend and the investigation pipeline.

---

## Deployment

The intended deployment architecture separates the frontend and backend:

```text
User
 │
 ▼
Streamlit Community Cloud
 │
 │ HTTP
 ▼
FastAPI Backend
 │
 ├── Hybrid Retrieval
 ├── Evidence Graph
 ├── Investigator Agent
 ├── Fact-Checker Agent
 └── Gemini API
```

The backend is deployed as a FastAPI service and the Streamlit application communicates with it using the deployed backend URL.

API credentials are supplied through deployment environment variables/secrets rather than committed to the repository.

---

## Technologies

| Component                  | Technology           |
| -------------------------- | -------------------- |
| Backend                    | FastAPI              |
| Frontend                   | Streamlit            |
| LLM                        | Google Gemini        |
| Semantic Embeddings        | SentenceTransformers |
| Lexical Retrieval          | BM25                 |
| Evidence Graph             | NetworkX             |
| Numerical Processing       | NumPy                |
| Machine Learning Utilities | scikit-learn         |
| Dataset                    | DWIE                 |
| Server                     | Uvicorn              |
| Language                   | Python               |

---

## Bonus / Extended Features

The implementation goes beyond a basic RAG pipeline through:

* Hybrid lexical + semantic retrieval
* Bounded query reformulation
* Adversarial fact checking
* Evidence graph visualization and inspection
* Agent execution/search traces
* Candidate interrogation
* Verified/unverified evidence handling
* Deterministic citation validation
* User prediction before AI-assisted investigation
* Final verdict evaluation

---

## AI-Assisted Development

AI coding assistants were used during development for code generation, debugging, and implementation support.

The final system was tested and integrated manually, with the developer responsible for understanding the architecture, reviewing generated code, and validating the investigation workflow.

---

## Limitations

The system has several limitations:

* It relies on the quality and coverage of the underlying corpus.
* Semantic retrieval may miss relevant evidence or retrieve superficially similar text.
* LLM-generated reasoning can still contain errors.
* Citation validation verifies citation references, but does not guarantee that every generated conclusion is factually correct.
* The evidence graph depends on the information extracted from the source corpus.
* The deployed free-tier infrastructure may have cold starts and resource limitations.

---

## Future Improvements

Potential extensions include:

* More sophisticated reranking of retrieved evidence
* Larger and more diverse corpora
* Improved entity resolution
* More advanced graph-based retrieval
* Stronger automated faithfulness evaluation
* Persistent investigation sessions
* More sophisticated agent planning and tool selection

---

## License

This project is intended as an academic/project submission and demonstration of an agentic RAG investigation system.
