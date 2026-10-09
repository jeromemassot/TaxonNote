from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional, Set, Tuple
from pydantic import BaseModel

from backend.graph_engine import XMLGraphWorkspace

try:
    from google import genai
    from google.genai import types
    import google.auth
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False


class ClinicalSpan(BaseModel):
    start: int
    end: int
    text: str


class CompetingCode(BaseModel):
    code: str
    title: str
    trajectory_id: int
    breadcrumb: str
    is_unspecified: bool = False


class QuickFix(BaseModel):
    id: str
    label: str
    action: str  # "replace" or "insert"
    target_start: int
    target_end: int
    original_text: str = ""
    replacement_text: str
    branch_node_label: Optional[str] = None
    branch_code: Optional[str] = None
    node_id: Optional[str] = None


class AmbiguityReport(BaseModel):
    id: str
    condition_name: str
    span: Dict[str, int]  # {"start": ..., "end": ...}
    category: str  # laterality, acuity, etiology, manifestation, anatomical_subsite, encounter
    severity: str  # "high", "medium", "low"
    reason: str
    coder_query: str
    unspecified_code_risk: Optional[str] = None
    competing_codes: List[CompetingCode]
    trajectory_ids: List[int]
    quick_fixes: List[QuickFix]
    fork_node_id: Optional[str] = None


class ExtractedCondition(BaseModel):
    id: str
    condition_name: str
    span: Dict[str, int]
    keywords: List[str]
    has_ambiguity: bool
    ambiguity_ids: List[str]
    resolved_code: Optional[str] = None
    resolved_breadcrumb: Optional[str] = None
    trajectory_id: Optional[int] = None


class CoderPreviewItem(BaseModel):
    code: str
    description: str
    condition_source: str
    is_specific: bool
    risk_level: str  # "clean", "warning", "audit_risk"
    risk_message: Optional[str] = None
    trajectory_id: Optional[int] = None


class SupervisionResult(BaseModel):
    note_text: str
    coder_readiness_score: int
    readiness_grade: str
    total_conditions: int
    resolved_conditions: int
    ambiguity_count: int
    conditions: List[ExtractedCondition]
    ambiguities: List[AmbiguityReport]
    coder_preview: List[CoderPreviewItem]
    model_used: str


# Common medical concepts subject to strict ICD-10-CM specificity rules
LATERALITY_SITES = {
    "knee": {"terms": ["knee", "patella", "meniscus"], "organs": ["right", "left", "bilateral"]},
    "shoulder": {"terms": ["shoulder", "rotator cuff", "clavicle", "acromioclavicular"], "organs": ["right", "left", "bilateral"]},
    "ankle": {"terms": ["ankle", "malleolus", "achilles"], "organs": ["right", "left", "bilateral"]},
    "wrist": {"terms": ["wrist", "carpal", "radius", "ulna"], "organs": ["right", "left", "bilateral"]},
    "hip": {"terms": ["hip", "femur", "acetabulum", "trochanter"], "organs": ["right", "left", "bilateral"]},
    "foot": {"terms": ["foot", "toe", "metatarsal", "heel", "calcaneus"], "organs": ["right", "left", "bilateral"]},
    "hand": {"terms": ["hand", "finger", "thumb", "metacarpal"], "organs": ["right", "left", "bilateral"]},
    "elbow": {"terms": ["elbow", "olecranon", "epicondyle"], "organs": ["right", "left", "bilateral"]},
    "ear": {"terms": ["ear", "tympanic", "otitis"], "organs": ["right", "left", "bilateral"]},
    "eye": {"terms": ["eye", "cornea", "retina", "conjunctiva", "eyelid"], "organs": ["right", "left", "bilateral"]},
    "breast": {"terms": ["breast", "mammary"], "organs": ["right", "left", "bilateral"]},
    "kidney": {"terms": ["kidney", "renal"], "organs": ["right", "left", "bilateral"]},
}

ACUITY_CONDITIONS = {
    "heart failure": {
        "keywords": ["heart failure", "chf"],
        "options": ["acute systolic", "chronic systolic", "acute on chronic systolic", "acute diastolic", "chronic diastolic", "acute on chronic diastolic"],
        "default_unspecified": "I50.9 (Heart failure, unspecified)",
    },
    "bronchitis": {
        "keywords": ["bronchitis"],
        "options": ["acute", "chronic", "with bronchospasm"],
        "default_unspecified": "J40 (Bronchitis, not specified as acute or chronic)",
    },
    "pancreatitis": {
        "keywords": ["pancreatitis"],
        "options": ["acute", "chronic", "acute on chronic", "alcohol-induced"],
        "default_unspecified": "K85.90 (Acute pancreatitis, unspecified)",
    },
    "sinusitis": {
        "keywords": ["sinusitis"],
        "options": ["acute", "chronic", "recurrent"],
        "default_unspecified": "J32.9 (Chronic sinusitis, unspecified)",
    },
    "gout": {
        "keywords": ["gout", "podagra"],
        "options": ["acute", "chronic with tophus", "chronic without tophus"],
        "default_unspecified": "M10.9 (Gout, unspecified)",
    },
    "cystitis": {
        "keywords": ["cystitis"],
        "options": ["acute", "chronic interstitial", "with hematuria", "without hematuria"],
        "default_unspecified": "N30.90 (Cystitis, unspecified without hematuria)",
    },
    "respiratory failure": {
        "keywords": ["respiratory failure"],
        "options": ["acute with hypoxia", "acute with hypercapnia", "chronic with hypoxia", "acute on chronic with hypoxia"],
        "default_unspecified": "J96.90 (Respiratory failure, unspecified)",
    },
    "otitis media": {
        "keywords": ["otitis media"],
        "options": ["acute serous", "acute suppurative", "chronic serous", "chronic mucoid"],
        "default_unspecified": "H66.90 (Otitis media, unspecified)",
    },
}

DIABETES_SUBTYPES = {
    "keywords": ["diabetes", "dm", "diabetic"],
    "subtypes": ["type 1", "type 2", "gestational", "drug-induced"],
    "manifestations": [
        ("neuropathy", "with neuropathy"),
        ("nephropathy", "with diabetic kidney disease / CKD"),
        ("retinopathy", "with retinopathy"),
        ("ulcer", "with foot ulcer"),
        ("hyperglycemia", "with hyperglycemia"),
        ("hypoglycemia", "with hypoglycemia"),
        ("without complication", "without complications"),
    ],
}


class SupervisionEngine:
    """
    AI Agentic Supervisor:
    Continuously monitors clinical note composition, detects ambiguity forks in the
    ICD-10-CM trajectory graph, predicts medical coder queries, and generates 1-click fixes.
    """

    def __init__(self, workspace: XMLGraphWorkspace):
        self.workspace = workspace
        self._genai_client = None

    def _get_genai_client(self):
        if not HAS_GENAI:
            return None
        if self._genai_client is not None:
            return self._genai_client

        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if api_key:
            self._genai_client = genai.Client(api_key=api_key)
            return self._genai_client

        project = os.environ.get("GOOGLE_CLOUD_PROJECT") or "ai-for-good-projects"
        location = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
        try:
            self._genai_client = genai.Client(vertexai=True, project=project, location=location)
            return self._genai_client
        except Exception:
            return None

    def supervise_note(self, note_text: str) -> SupervisionResult:
        """
        Main supervision entry point.
        Analyzes clinical note, identifies conditions, checks against ICD-10-CM trajectories,
        diagnoses coder confusion risks, and computes readiness score.
        """
        clean_text = (note_text or "").strip()
        if not clean_text:
            return SupervisionResult(
                note_text=note_text,
                coder_readiness_score=100,
                readiness_grade="A",
                total_conditions=0,
                resolved_conditions=0,
                ambiguity_count=0,
                conditions=[],
                ambiguities=[],
                coder_preview=[],
                model_used="None (empty note)",
            )

        # 1. Extract clinical conditions and exact spans
        extracted_raw, model_name = self._extract_conditions(clean_text)

        conditions: List[ExtractedCondition] = []
        ambiguities: List[AmbiguityReport] = []
        coder_preview_items: List[CoderPreviewItem] = []

        ambiguity_counter = 0

        # 2. For each extracted condition, evaluate against ICD-10 graph trajectories
        for idx, item in enumerate(extracted_raw):
            c_id = f"cond_{idx + 1}"
            c_name = item["condition"]
            span = item["span"]
            keywords = item["keywords"]

            # Query the ICD-10-CM graph
            query_str = "; ".join(keywords) if keywords else c_name
            search_res = self.workspace.search_subgraph(query=query_str, max_results=12)
            trajectories = search_res.get("trajectories", [])

            # Check for ambiguity forks
            c_ambiguities = self._detect_ambiguity_forks(
                note_text=clean_text,
                condition_name=c_name,
                span=span,
                keywords=keywords,
                trajectories=trajectories,
                counter_start=ambiguity_counter,
            )

            ambiguity_ids = [a.id for a in c_ambiguities]
            ambiguity_counter += len(c_ambiguities)
            ambiguities.extend(c_ambiguities)

            has_amb = len(c_ambiguities) > 0
            resolved_code = None
            resolved_breadcrumb = None
            top_tid = None

            if trajectories:
                top_traj = trajectories[0]
                top_tid = top_traj["trajectory_id"]
                resolved_code = top_traj["code"]
                resolved_breadcrumb = top_traj["breadcrumb"]

                # Generate Coder Preview Item
                is_unspecified = self._is_code_unspecified(resolved_code) or has_amb
                risk_lvl = "audit_risk" if has_amb else ("warning" if is_unspecified else "clean")
                risk_msg = None
                if has_amb:
                    risk_msg = f"Ambiguity detected: {c_ambiguities[0].reason}"
                elif is_unspecified:
                    risk_msg = "Coded with unspecified ICD-10 modifier; may face claim inquiries."

                coder_preview_items.append(
                    CoderPreviewItem(
                        code=resolved_code,
                        description=f"{c_name} ({resolved_breadcrumb.split('→')[-1].strip()})",
                        condition_source=c_name,
                        is_specific=not is_unspecified,
                        risk_level=risk_lvl,
                        risk_message=risk_msg,
                        trajectory_id=top_tid,
                    )
                )

            conditions.append(
                ExtractedCondition(
                    id=c_id,
                    condition_name=c_name,
                    span=span,
                    keywords=keywords,
                    has_ambiguity=has_amb,
                    ambiguity_ids=ambiguity_ids,
                    resolved_code=resolved_code,
                    resolved_breadcrumb=resolved_breadcrumb,
                    trajectory_id=top_tid,
                )
            )

        # 3. Calculate Coder Readiness Score
        total_cond = len(conditions)
        high_sev_amb = sum(1 for a in ambiguities if a.severity == "high")
        med_sev_amb = sum(1 for a in ambiguities if a.severity == "medium")
        resolved_count = sum(1 for c in conditions if not c.has_ambiguity)

        if total_cond == 0:
            readiness_score = 100
        else:
            base_score = int((resolved_count / total_cond) * 100)
            penalty = (high_sev_amb * 15) + (med_sev_amb * 8)
            readiness_score = max(10, min(100, base_score - penalty + (15 if resolved_count > 0 else 0)))

        if readiness_score >= 90:
            grade = "A"
        elif readiness_score >= 75:
            grade = "B"
        elif readiness_score >= 50:
            grade = "C"
        else:
            grade = "D"

        return SupervisionResult(
            note_text=clean_text,
            coder_readiness_score=readiness_score,
            readiness_grade=grade,
            total_conditions=total_cond,
            resolved_conditions=resolved_count,
            ambiguity_count=len(ambiguities),
            conditions=conditions,
            ambiguities=ambiguities,
            coder_preview=coder_preview_items,
            model_used=model_name,
        )

    def _extract_conditions(self, note_text: str) -> Tuple[List[Dict[str, Any]], str]:
        """
        Extracts clinical entities, diagnoses, and symptoms with character offsets.
        Attempts Gemini 2.5 Flash first if available, falls back to clinical NLP pattern matcher.
        """
        client = self._get_genai_client()
        if client is not None:
            try:
                prompt = (
                    "You are a clinical documentation and ICD-10-CM coding AI supervisor.\n"
                    "Analyze the following medical note and identify all medical conditions, diagnoses, or clinical findings.\n"
                    "For each condition:\n"
                    "1. `condition`: exact or concise clinical term as written in the note.\n"
                    "2. `start_char`: 0-indexed character offset where the condition begins in the note.\n"
                    "3. `end_char`: 0-indexed character offset where the condition ends in the note.\n"
                    "4. `keywords`: list of atomic ICD-10-CM index search terms (main term, anatomical site, acuity, subtype).\n\n"
                    f"Medical Note:\n{note_text}"
                )

                class ExtractedEntityItem(BaseModel):
                    condition: str
                    start_char: int
                    end_char: int
                    keywords: List[str]

                class ExtractionOutput(BaseModel):
                    entities: List[ExtractedEntityItem]

                resp = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=ExtractionOutput,
                        temperature=0.1,
                    ),
                )
                parsed = getattr(resp, "parsed", None)
                if parsed is None and resp.text:
                    parsed = ExtractionOutput.model_validate_json(resp.text)

                if parsed and parsed.entities:
                    res = []
                    lower_note = note_text.lower()
                    last_pos = 0
                    for e in parsed.entities:
                        c_text = e.condition.strip()
                        c_lower_text = c_text.lower()
                        # Search for actual character occurrence in note text
                        found_idx = lower_note.find(c_lower_text, last_pos)
                        if found_idx == -1:
                            found_idx = lower_note.find(c_lower_text)
                        if found_idx != -1:
                            s = found_idx
                            e_char = found_idx + len(c_text)
                            last_pos = e_char
                        else:
                            s = max(0, min(e.start_char, len(note_text)))
                            e_char = max(s, min(e.end_char, len(note_text)))
                        res.append(
                            {
                                "condition": c_text,
                                "span": {"start": s, "end": e_char},
                                "keywords": [k.strip() for k in e.keywords if k.strip()],
                            }
                        )
                    return res, "Gemini 2.5 Flash"
            except Exception:
                pass  # Fall back to deterministic clinical pattern extractor

        # Deterministic Clinical Pattern Extractor fallback
        return self._fast_clinical_pattern_extractor(note_text), "TaxonNote Clinical Parser"

    def _fast_clinical_pattern_extractor(self, text: str) -> List[Dict[str, Any]]:
        """
        Fast deterministic clinical entity and span detector using medical lexicon & regex matching.
        """
        results: List[Dict[str, Any]] = []
        lower_text = text.lower()

        # Clinical patterns and condition triggers
        patterns = [
            (r"\b(type\s*2\s*diabetes(?:\s*mellitus)?)\b", ["diabetes", "type 2"]),
            (r"\b(type\s*1\s*diabetes(?:\s*mellitus)?)\b", ["diabetes", "type 1"]),
            (r"\b(diabetes(?:\s*mellitus)?)\b", ["diabetes"]),
            (r"\b(hypertension|essential\s*hypertension|htn)\b", ["hypertension"]),
            (r"\b(knee\s*pain|pain\s+in\s+knee)\b", ["pain", "joint", "knee"]),
            (r"\b(shoulder\s*pain|pain\s+in\s+shoulder)\b", ["pain", "joint", "shoulder"]),
            (r"\b(hip\s*pain|pain\s+in\s+hip)\b", ["pain", "joint", "hip"]),
            (r"\b(ankle\s*pain|pain\s+in\s+ankle)\b", ["pain", "joint", "ankle"]),
            (r"\b(wrist\s*pain|pain\s+in\s+wrist)\b", ["pain", "joint", "wrist"]),
            (r"\b(congestive\s*heart\s*failure|heart\s*failure|chf)\b", ["failure", "heart"]),
            (r"\b(osteoarthritis|oa)\b", ["osteoarthritis"]),
            (r"\b(meniscus\s*tear|tear\s+of\s+meniscus)\b", ["tear", "meniscus"]),
            (r"\b(rotator\s*cuff\s*tear|rotator\s*cuff\s*syndrome)\b", ["rotator cuff"]),
            (r"\b(distal\s*radius\s*fracture|radius\s*fracture|fracture\s+of\s+radius)\b", ["fracture", "radius"]),
            (r"\b(femur\s*fracture|fracture\s+of\s+femur)\b", ["fracture", "femur"]),
            (r"\b(foot\s*ulcer|ulcer\s+of\s+foot|diabetic\s*ulcer)\b", ["ulcer", "foot"]),
            (r"\b(pneumonia)\b", ["pneumonia"]),
            (r"\b(acute\s*bronchitis|bronchitis)\b", ["bronchitis"]),
            (r"\b(acute\s*pancreatitis|pancreatitis)\b", ["pancreatitis"]),
            (r"\b(otitis\s*media)\b", ["otitis", "media"]),
            (r"\b(chronic\s*kidney\s*disease|ckd)\b", ["disease", "kidney", "chronic"]),
            (r"\b(headache|migraine)\b", ["migraine"]),
            (r"\b(asthma)\b", ["asthma"]),
            (r"\b(cellulitis)\b", ["cellulitis"]),
            (r"\b(urinary\s*tract\s*infection|uti)\b", ["infection", "urinary"]),
            (r"\b(appendicitis)\b", ["appendicitis"]),
        ]

        seen_spans: List[Tuple[int, int]] = []
        for pat_str, kws in patterns:
            for m in re.finditer(pat_str, lower_text):
                start, end = m.start(), m.end()
                # Check overlapping
                if any(s <= start < e or s < end <= e for s, e in seen_spans):
                    continue
                seen_spans.append((start, end))

                matched_str = text[start:end]
                # Context enrichment (check for preceding acute/chronic/right/left)
                pre_window = lower_text[max(0, start - 25) : start]
                enriched_kws = list(kws)
                if "right" in pre_window or "rt" in pre_window:
                    enriched_kws.append("right")
                elif "left" in pre_window or "lt" in pre_window:
                    enriched_kws.append("left")
                elif "bilateral" in pre_window:
                    enriched_kws.append("bilateral")

                if "acute" in pre_window:
                    enriched_kws.append("acute")
                elif "chronic" in pre_window:
                    enriched_kws.append("chronic")

                results.append(
                    {
                        "condition": matched_str,
                        "span": {"start": start, "end": end},
                        "keywords": enriched_kws,
                    }
                )

        results.sort(key=lambda x: x["span"]["start"])
        return results

    def _detect_ambiguity_forks(
        self,
        note_text: str,
        condition_name: str,
        span: Dict[str, int],
        keywords: List[str],
        trajectories: List[Dict[str, Any]],
        counter_start: int,
    ) -> List[AmbiguityReport]:
        """
        Analyzes the ICD-10-CM trajectory matches against the clinical note to identify
        branching forks where missing modifiers cause medical coder confusion.
        """
        ambiguities: List[AmbiguityReport] = []
        c_lower = condition_name.lower()
        full_lower = note_text.lower()
        start = span["start"]
        end = span["end"]

        # Context windows around the mention
        line_start = max(0, full_lower.rfind("\n", 0, start) + 1)
        line_end = full_lower.find("\n", end)
        if line_end == -1:
            line_end = len(full_lower)
        context_line = full_lower[line_start:line_end]

        sent_start = max(0, full_lower.rfind(".", 0, start) + 1)
        sent_end = full_lower.find(".", end)
        if sent_end == -1:
            sent_end = len(full_lower)
        context_sentence = full_lower[sent_start:sent_end]
        local_window = full_lower[max(0, start - 60):min(len(full_lower), end + 60)]

        # 1. Check for Laterality Ambiguity
        for site, info in LATERALITY_SITES.items():
            matched_term = next((t for t in info["terms"] if re.search(rf"\b{re.escape(t)}\b", c_lower)), None)
            if matched_term:
                side_terms = ["right", "left", "bilateral", "rt", "lt"]
                has_side = (
                    any(re.search(rf"\b{re.escape(side)}\b", c_lower) for side in side_terms)
                    or any(re.search(rf"\b{re.escape(side)}\b", context_sentence) for side in side_terms)
                    or any(re.search(rf"\b{re.escape(side)}\b", context_line) for side in side_terms)
                    or any(re.search(rf"\b{re.escape(side)}\b", local_window) for side in side_terms)
                    or any(side in [k.lower() for k in keywords] for side in ["right", "left", "bilateral"])
                )
                if not has_side:
                    competing = self._find_competing_laterality_trajectories(site, trajectories)
                    if competing:
                        aid = f"amb_{counter_start + len(ambiguities) + 1}"
                        right_text = re.sub(rf"\b{re.escape(matched_term)}\b", f"right {matched_term}", condition_name, count=1, flags=re.IGNORECASE)
                        left_text = re.sub(rf"\b{re.escape(matched_term)}\b", f"left {matched_term}", condition_name, count=1, flags=re.IGNORECASE)
                        bilat_text = re.sub(rf"\b{re.escape(matched_term)}\b", f"bilateral {matched_term}", condition_name, count=1, flags=re.IGNORECASE)
                        right_code = next((c.code for c in competing if "right" in c.breadcrumb.lower() or c.code.endswith("1")), None)
                        left_code = next((c.code for c in competing if "left" in c.breadcrumb.lower() or c.code.endswith("2")), None)
                        bilat_code = next((c.code for c in competing if "bilateral" in c.breadcrumb.lower() or c.code.endswith("3")), None)
                        fixes = [
                            QuickFix(
                                id=f"fix_{aid}_1",
                                label=f"+ Right {matched_term.capitalize()}",
                                action="replace",
                                target_start=start,
                                target_end=end,
                                original_text=condition_name,
                                replacement_text=right_text,
                                branch_node_label=f"Right",
                                branch_code=right_code,
                            ),
                            QuickFix(
                                id=f"fix_{aid}_2",
                                label=f"+ Left {matched_term.capitalize()}",
                                action="replace",
                                target_start=start,
                                target_end=end,
                                original_text=condition_name,
                                replacement_text=left_text,
                                branch_node_label=f"Left",
                                branch_code=left_code,
                            ),
                            QuickFix(
                                id=f"fix_{aid}_3",
                                label=f"+ Bilateral {matched_term.capitalize()}",
                                action="replace",
                                target_start=start,
                                target_end=end,
                                original_text=condition_name,
                                replacement_text=bilat_text,
                                branch_node_label=f"Bilateral",
                                branch_code=bilat_code,
                            ),
                        ]
                        ambiguities.append(
                            AmbiguityReport(
                                id=aid,
                                condition_name=condition_name,
                                span=span,
                                category="laterality",
                                severity="high",
                                reason=f"Laterality not specified for {site}. Coders must select between Right, Left, or Bilateral.",
                                coder_query=f"Please specify laterality (Right, Left, or Bilateral) for the patient's {condition_name}.",
                                unspecified_code_risk=f"Will default to unspecified code (e.g. ending in .9), leading to possible insurance denial or medical necessity audit.",
                                competing_codes=competing,
                                trajectory_ids=[c.trajectory_id for c in competing],
                                quick_fixes=fixes,
                                fork_node_id=trajectories[0]["term_id"] if trajectories else None,
                            )
                        )
                    break

        # 2. Check for Acuity / Chronicity Ambiguity
        for cond_key, a_info in ACUITY_CONDITIONS.items():
            if any(re.search(rf"\b{re.escape(kw)}\b", c_lower) for kw in a_info["keywords"]):
                acuity_terms = ["acute", "chronic", "subacute", "acute on chronic", "exacerbation"]
                has_acuity = (
                    any(re.search(rf"\b{re.escape(ac)}\b", c_lower) for ac in acuity_terms)
                    or any(re.search(rf"\b{re.escape(ac)}\b", context_sentence) for ac in acuity_terms)
                    or any(re.search(rf"\b{re.escape(ac)}\b", context_line) for ac in acuity_terms)
                    or any(re.search(rf"\b{re.escape(ac)}\b", local_window) for ac in acuity_terms)
                    or any(ac in [k.lower() for k in keywords] for ac in ["acute", "chronic", "subacute"])
                )
                if not has_acuity:
                    competing = self._find_competing_acuity_trajectories(cond_key, trajectories)
                    if competing:
                        aid = f"amb_{counter_start + len(ambiguities) + 1}"
                        acute_code = next((c.code for c in competing if "acute" in c.breadcrumb.lower()), None)
                        chronic_code = next((c.code for c in competing if "chronic" in c.breadcrumb.lower()), None)
                        fixes = [
                            QuickFix(
                                id=f"fix_{aid}_1",
                                label="+ Acute",
                                action="replace",
                                target_start=start,
                                target_end=end,
                                original_text=condition_name,
                                replacement_text=f"acute {condition_name}",
                                branch_node_label="Acute",
                                branch_code=acute_code,
                            ),
                            QuickFix(
                                id=f"fix_{aid}_2",
                                label="+ Chronic",
                                action="replace",
                                target_start=start,
                                target_end=end,
                                original_text=condition_name,
                                replacement_text=f"chronic {condition_name}",
                                branch_node_label="Chronic",
                                branch_code=chronic_code,
                            ),
                            QuickFix(
                                id=f"fix_{aid}_3",
                                label="+ Acute on chronic",
                                action="replace",
                                target_start=start,
                                target_end=end,
                                original_text=condition_name,
                                replacement_text=f"acute on chronic {condition_name}",
                                branch_node_label="Acute on chronic",
                                branch_code=acute_code,
                            ),
                        ]
                        ambiguities.append(
                            AmbiguityReport(
                                id=aid,
                                condition_name=condition_name,
                                span=span,
                                category="acuity",
                                severity="medium",
                                reason=f"Acuity/chronicity is not specified for {condition_name}. Coder cannot distinguish acute from chronic.",
                                coder_query=f"Please clarify the acuity/chronicity of {condition_name} (acute, chronic, or acute on chronic).",
                                unspecified_code_risk=f"Default code: {a_info['default_unspecified']} (Unspecified acuity).",
                                competing_codes=competing,
                                trajectory_ids=[c.trajectory_id for c in competing],
                                quick_fixes=fixes,
                                fork_node_id=trajectories[0]["term_id"] if trajectories else None,
                            )
                        )
                    break

        # 3. Check for Diabetes Subtype & Manifestation Ambiguity
        if "diabetes" in c_lower or "dm" in c_lower:
            type_terms = ["type 1", "type 2", "gestational", "drug-induced"]
            has_type = (
                any(re.search(rf"\b{re.escape(t)}\b", c_lower) for t in type_terms)
                or any(re.search(rf"\b{re.escape(t)}\b", context_sentence) for t in type_terms)
                or any(re.search(rf"\b{re.escape(t)}\b", context_line) for t in type_terms)
                or any(re.search(rf"\b{re.escape(t)}\b", local_window) for t in type_terms)
                or any(t in [k.lower() for k in keywords] for t in ["type 1", "type 2"])
            )
            if not has_type:
                aid = f"amb_{counter_start + len(ambiguities) + 1}"
                competing = [
                    CompetingCode(
                        code="E11.9",
                        title="Type 2 diabetes mellitus without complications",
                        trajectory_id=trajectories[0]["trajectory_id"] if trajectories else 0,
                        breadcrumb="Diabetes › type 2 → E11.9",
                    ),
                    CompetingCode(
                        code="E10.9",
                        title="Type 1 diabetes mellitus without complications",
                        trajectory_id=trajectories[1]["trajectory_id"] if len(trajectories) > 1 else 0,
                        breadcrumb="Diabetes › type 1 → E10.9",
                    ),
                ]
                fixes = [
                    QuickFix(
                        id=f"fix_{aid}_1",
                        label="Specify Type 2",
                        action="replace",
                        target_start=start,
                        target_end=end,
                        original_text=condition_name,
                        replacement_text="type 2 diabetes",
                        branch_node_label="Type 2",
                        branch_code="E11.9",
                    ),
                    QuickFix(
                        id=f"fix_{aid}_2",
                        label="Specify Type 1",
                        action="replace",
                        target_start=start,
                        target_end=end,
                        original_text=condition_name,
                        replacement_text="type 1 diabetes",
                        branch_node_label="Type 1",
                        branch_code="E10.9",
                    ),
                ]
                ambiguities.append(
                    AmbiguityReport(
                        id=aid,
                        condition_name=condition_name,
                        span=span,
                        category="etiology",
                        severity="high",
                        reason="Diabetes type (Type 1 vs Type 2) not documented. Affects primary diagnostic category.",
                        coder_query="Please document whether the diabetes is Type 1, Type 2, or secondary.",
                        unspecified_code_risk="Defaults to E11.9 (Type 2 by coding default if adult, or triggers query).",
                        competing_codes=competing,
                        trajectory_ids=[c.trajectory_id for c in competing],
                        quick_fixes=fixes,
                    )
                )

        return ambiguities

    def _find_competing_laterality_trajectories(
        self, site: str, trajectories: List[Dict[str, Any]]
    ) -> List[CompetingCode]:
        competing: List[CompetingCode] = []
        seen_codes: Set[str] = set()

        for t in trajectories:
            code = t["code"]
            breadcrumb = t["breadcrumb"].lower()
            if code not in seen_codes:
                is_unspec = self._is_code_unspecified(code)
                is_right = "right" in breadcrumb or code.endswith("1")
                is_left = "left" in breadcrumb or code.endswith("2")
                is_bilat = "bilateral" in breadcrumb or code.endswith("3")

                if is_right or is_left or is_bilat or is_unspec:
                    seen_codes.add(code)
                    competing.append(
                        CompetingCode(
                            code=code,
                            title=t["breadcrumb"].split("→")[-1].strip(),
                            trajectory_id=t["trajectory_id"],
                            breadcrumb=t["breadcrumb"],
                            is_unspecified=is_unspec,
                        )
                    )
            if len(competing) >= 4:
                break
        return competing

    def _find_competing_acuity_trajectories(
        self, cond_name: str, trajectories: List[Dict[str, Any]]
    ) -> List[CompetingCode]:
        competing: List[CompetingCode] = []
        seen_codes: Set[str] = set()

        for t in trajectories:
            code = t["code"]
            breadcrumb = t["breadcrumb"].lower()
            if code not in seen_codes:
                is_unspec = self._is_code_unspecified(code)
                is_acute = "acute" in breadcrumb
                is_chronic = "chronic" in breadcrumb

                if is_acute or is_chronic or is_unspec:
                    seen_codes.add(code)
                    competing.append(
                        CompetingCode(
                            code=code,
                            title=t["breadcrumb"].split("→")[-1].strip(),
                            trajectory_id=t["trajectory_id"],
                            breadcrumb=t["breadcrumb"],
                            is_unspecified=is_unspec,
                        )
                    )
            if len(competing) >= 4:
                break
        return competing

    @staticmethod
    def _is_code_unspecified(code: str) -> bool:
        if not code:
            return False
        clean = code.strip()
        return clean.endswith(".9") or clean.endswith("9") or clean.endswith(".90") or clean.endswith(".99")
