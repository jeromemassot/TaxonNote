from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from backend.graph_engine import XMLGraphWorkspace
from backend.supervision_engine import SupervisionEngine, SupervisionResult

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
XML_DOC_PATH = DATA_DIR / "xml_documents" / "icd10cm-index-2027.xml"
XML_XSD_PATH = DATA_DIR / "xml_schemas" / "icd10cm-index.xsd"
STATIC_DIR = Path(__file__).resolve().parent / "static"

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm up workspace index on startup
    get_workspace()
    get_supervision_engine()
    yield

app = FastAPI(
    title="TaxonNote: AI Supervised Medical Note Redaction",
    description="Real-time AI agentic supervision system for medical notes to eliminate coder confusion in ICD-10-CM mapping.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

_WORKSPACE: Optional[XMLGraphWorkspace] = None
_SUPERVISION_ENGINE: Optional[SupervisionEngine] = None


def get_workspace() -> XMLGraphWorkspace:
    global _WORKSPACE
    if _WORKSPACE is None:
        if not XML_DOC_PATH.exists() or not XML_XSD_PATH.exists():
            raise HTTPException(
                status_code=500,
                detail="ICD-10-CM XML or XSD files not found in data directory.",
            )
        _WORKSPACE = XMLGraphWorkspace(XML_DOC_PATH, XML_XSD_PATH)
    return _WORKSPACE


def get_supervision_engine() -> SupervisionEngine:
    global _SUPERVISION_ENGINE
    if _SUPERVISION_ENGINE is None:
        ws = get_workspace()
        _SUPERVISION_ENGINE = SupervisionEngine(ws)
    return _SUPERVISION_ENGINE



@app.get("/", response_class=HTMLResponse)
def index():
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        return HTMLResponse("<h1>TaxonNote server running. Frontend index.html loading...</h1>")
    return FileResponse(str(index_file))


@app.get("/api/status")
def get_system_status():
    ws = get_workspace()
    summary = ws.get_summary()
    return {
        "status": "ready",
        "app_name": "TaxonNote",
        "version": "1.0.0",
        "knowledge_base": {
            "xml_filename": summary["xml_filename"],
            "total_nodes": summary["total_graph_nodes"],
            "total_trajectories": summary["total_trajectories"],
            "is_schema_valid": summary["is_valid"],
            "load_time_ms": summary["load_time_ms"],
        },
    }


class SuperviseRequest(BaseModel):
    note: str
    patient_context: Optional[Dict[str, Any]] = None


@app.post("/api/supervise", response_model=SupervisionResult)
def supervise_note(req: SuperviseRequest):
    engine = get_supervision_engine()
    return engine.supervise_note(req.note)


@app.get("/api/fork-subgraph")
def get_fork_subgraph(
    tids: str = Query(..., description="Comma-separated trajectory IDs, e.g. 102,103,105"),
    fork_node_id: Optional[str] = Query(None, description="Optional fork node id to highlight"),
):
    ws = get_workspace()
    try:
        trajectory_ids = [int(x.strip()) for x in tids.split(",") if x.strip().isdigit()]
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid trajectory IDs parameter.") from exc

    return ws.get_fork_subgraph(trajectory_ids, highlight_fork_node_id=fork_node_id)


@app.get("/api/node-detail/{node_id:path}")
def get_node_detail(node_id: str):
    ws = get_workspace()
    return ws.get_node_detail(node_id)


@app.get("/api/templates")
def get_clinical_templates():
    return {
        "templates": [
            {
                "id": "ortho_ambiguity",
                "title": "Orthopedic Consultation (Knee Pain)",
                "category": "Orthopedics",
                "description": "Demonstrates missing laterality on knee pain and meniscus tear.",
                "text": (
                    "Chief Complaint: Patient presents with persistent knee pain for 3 weeks.\n\n"
                    "History of Present Illness: 48-year-old male reports sudden onset pain while twisting during recreational tennis. "
                    "Reports joint line tenderness, mild swelling, and catching sensation. Suspected meniscus tear.\n\n"
                    "Assessment & Plan:\n"
                    "1. Severe knee pain, likely acute medial meniscus tear.\n"
                    "2. Order MRI and prescribe conservative NSAID therapy."
                ),
            },
            {
                "id": "cardio_ambiguity",
                "title": "Cardiology Progress Note (Heart Failure)",
                "category": "Cardiology",
                "description": "Demonstrates missing acuity and chronicity on heart failure and hypertension.",
                "text": (
                    "Subjective: 71-year-old female admitted with worsening dyspnea on exertion and 3+ pitting edema.\n\n"
                    "Physical Exam: Crackles bilaterally at lung bases, elevated JVP. Blood pressure 162/95.\n\n"
                    "Assessment:\n"
                    "1. Decompensated congestive heart failure with volume overload.\n"
                    "2. Essential hypertension.\n"
                    "Plan: Increase IV furosemide, titrate beta-blocker, obtain echocardiogram to evaluate ejection fraction."
                ),
            },
            {
                "id": "diabetes_ambiguity",
                "title": "Endocrinology Follow-Up (Diabetes)",
                "category": "Endocrinology",
                "description": "Demonstrates missing diabetes subtype classification (Type 1 vs Type 2) and foot ulcer linkage.",
                "text": (
                    "HPI: 55-year-old patient with long-standing diabetes presents for routine diabetic foot check.\n"
                    "Exam reveals non-healing foot ulcer on plantar aspect of first metatarsal.\n"
                    "Also reports peripheral numbness and tingling in lower extremities.\n\n"
                    "Impression:\n"
                    "1. Diabetes mellitus with neuropathy and active foot ulcer.\n"
                    "2. Refer to podiatry for wound debridement."
                ),
            },
            {
                "id": "pulmonary_ambiguity",
                "title": "Pulmonary Clinic (Bronchitis & Sinusitis)",
                "category": "Pulmonology",
                "description": "Demonstrates acuity gaps on respiratory conditions.",
                "text": (
                    "HPI: 34-year-old patient presents with productive cough, discolored sputum, and facial pressure lasting 10 days.\n\n"
                    "Assessment:\n"
                    "1. Bronchitis.\n"
                    "2. Sinusitis.\n"
                    "Plan: Supportive care, inhaled bronchodilator PRN, nasal saline rinses."
                ),
            },
            {
                "id": "coder_ready_example",
                "title": "Coder-Ready Benchmark (Grade A)",
                "category": "Multispecialty",
                "description": "Demonstrates fully specified clinical note with 100% Coder Readiness.",
                "text": (
                    "Patient: 62-year-old female presents for urgent evaluation.\n\n"
                    "HPI: Evaluated for acute right knee pain after a stumble at home yesterday.\n"
                    "Past Medical History: Known type 2 diabetes mellitus without complications, and essential hypertension.\n\n"
                    "Assessment & Plan:\n"
                    "1. Acute severe right knee pain, rule out acute sprain.\n"
                    "2. Type 2 diabetes mellitus, well-controlled on metformin.\n"
                    "3. Primary essential hypertension, continue lisinopril."
                ),
            },
        ]
    }
