# TaxonNote 🩺

**Real-Time AI Clinical Documentation Supervision & ICD-10-CM Coding Ambiguity Prevention**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg)](https://fastapi.tiangolo.com)
[![Knowledge Base](https://img.shields.io/badge/ICD--10--CM-2027%20Index%20DAG-emerald.svg)](https://github.com/jeromemassot/Medical-Coding-based-on-ICD10-CM-Graph)
[![Tests](https://img.shields.io/badge/tests-6%20passed-success.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

TaxonNote is a clinical intelligence application designed to assist physicians while drafting clinical documentation. Grounded in a comprehensive directed acyclic graph (DAG) constructed from the official ICD-10-CM index, TaxonNote continuously supervises clinical notes in real time to prevent **medical coder confusion**, **documentation queries**, and **claim submission delays**.

---

## 🌟 The Problem & Solution

### The Challenge
When medical coders translate clinical notes into ICD-10-CM billing codes, they frequently encounter documentation gaps:
- **Missing Laterality**: E.g., documenting *"knee pain"* instead of *"right knee pain"* forcing coders to choose between query loops or non-specific default codes (`M25.569`).
- **Missing Acuity**: E.g., documenting *"bronchitis"* without specifying *acute* (`J20.9`) vs. *chronic* (`J42`).
- **Missing Etiology / Manifestation**: E.g., documenting *"diabetes"* and *"foot ulcer"* without explicit linkage.

### The TaxonNote Solution
TaxonNote acts as an intelligent supervisor sitting alongside the physician:
1. **Live Entity Extraction**: Detects clinical conditions, anatomical sites, and acuity markers as the note is drafted.
2. **Trajectory Traversal**: Matches extracted entities against over **103,000 graph nodes** and **64,000 coding trajectories** from the ICD-10-CM Index.
3. **Fork Ambiguity Detection**: Detects when a note halts at a branching point (e.g., knee pain branching into right, left, bilateral, or unspecified).
4. **Anticipated Coder Queries**: Formulates the exact query a medical coder would have to ask the physician.
5. **1-Click Enrichments**: Suggests precise, context-aware phrases that physicians can accept with a single click to instantly resolve ambiguities and update readiness scores.

---

## 🚀 Key Features

- **⚡ Real-Time Debounced Supervision**: Automatically evaluates notes 450ms after typing pauses, minimizing latency and API overhead.
- **📊 Coder Readiness Scoring (0–100%)**: Computes a transparent, grade-based score (Grade A–F) reflecting note specificity and coder query risk.
- **🎯 1-Click Note Enrichment**: Physicians can accept proposed enrichments directly into the clinical text, instantly triggering an immediate re-evaluation and confusion assessment.
- **🌳 Interactive Cytoscape Fork Inspector**: Visualizes the exact hierarchical decision tree in ICD-10-CM, letting physicians inspect downstream branches and select enrichments directly from nodes.
- **📋 Claim Preview**: Simulates how medical coders will map the note into billable ICD-10-CM diagnosis codes, with real-time specificity badges.
- **📁 Clinical Scenario Presets**: Preloaded with illustrative cases across Orthopedics, Cardiology, Pulmonology, and Endocrinology demonstrating common documentation pitfalls.
- **🌓 3-Way Mode Switcher**: Polished modern interface supporting **Light**, **Dark** (Slate-950), and **Auto** (OS system preference) modes with smooth transitions.

---

## 🏗️ Architecture

```
TaxonNote/
├── backend/
│   ├── graph_engine.py       # XML ingestion, NetworkX DAG indexing (103k+ nodes, 64k+ trajectories)
│   ├── supervision_engine.py # Entity extraction, offset reconciliation, fork detection & scoring
│   ├── main.py               # FastAPI application & REST endpoints
│   ├── test_app.py           # Comprehensive pytest suite (6/6 passing)
│   └── static/
│       └── index.html        # Modern dual-pane UI (Tailwind CSS, Cytoscape.js, Dagre layout)
├── data/
│   ├── xml_documents/        # Official ICD-10-CM 2027 Index XML (icd10cm-index-2027.xml)
│   └── xml_schemas/          # Validation schema (icd10cm-index.xsd)
├── requirements.txt          # Python dependencies
├── run.py                    # Application launcher
└── README.md
```

### Knowledge Base Grounding
The knowledge graph is parsed from `icd10cm-index-2027.xml`, indexed using NetworkX into hierarchical root-to-leaf paths. Root concepts branch into specific anatomical sites, laterality indicators, and acuity modifiers. Each branching point represents an opportunity for proactive ambiguity prevention.

Upstream ICD-10-CM graph reference: [Medical-Coding-based-on-ICD10-CM-Graph](https://github.com/jeromemassot/Medical-Coding-based-on-ICD10-CM-Graph).

---

## 🛠️ Quick Start

### Prerequisites
- Python 3.10+
- (Optional) `GEMINI_API_KEY` for LLM-powered entity extraction. If omitted, TaxonNote falls back to an integrated clinical regex terminology parser.

### 1. Clone the Repository
```bash
git clone https://github.com/jeromemassot/TaxonNote.git
cd TaxonNote
```

### 2. Set Up Virtual Environment & Dependencies
```bash
python -m venv .venv

# On Windows:
.\.venv\Scripts\activate

# On macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. (Optional) Configure Environment
```bash
cp .env.example .env
# Edit .env to add GEMINI_API_KEY if desired
```

### 4. Run the Application
```bash
python run.py
```
The server will start at **http://127.0.0.1:8000**. Open your browser to begin drafting clinical notes with live supervisor assistance.

---

## 🧪 Running Tests

The test suite validates the system status, template endpoints, clinical ambiguity detection for orthopedics and cardiology, coder-ready benchmark scoring, and Cytoscape fork subgraphs:

```bash
pytest backend/test_app.py -v
```

All 6 tests run and pass synchronously against the indexed ICD-10-CM graph.

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/status` | Returns system health, node count, and trajectory index metrics |
| `POST` | `/api/supervise` | Evaluates a clinical note and returns ambiguities, readiness score, and preview |
| `GET` | `/api/templates` | Provides clinical benchmark scenarios |
| `GET` | `/api/fork-subgraph` | Returns Cytoscape-formatted JSON for a specific trajectory fork |
| `GET` | `/api/node-detail/{id}` | Inspects detailed hierarchy and metadata for any ICD-10 node |

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
