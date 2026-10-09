from pathlib import Path
from fastapi.testclient import TestClient

from backend.main import app

client = TestClient(app)


def test_status_endpoint():
    response = client.get("/api/status")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ready"
    assert data["knowledge_base"]["total_nodes"] > 100000
    assert data["knowledge_base"]["total_trajectories"] > 50000


def test_templates_endpoint():
    response = client.get("/api/templates")
    assert response.status_code == 200
    data = response.json()
    assert "templates" in data
    assert len(data["templates"]) >= 4


def test_supervise_ortho_ambiguity():
    payload = {"note": "Patient presents with persistent severe knee pain for 3 weeks."}
    response = client.post("/api/supervise", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["coder_readiness_score"] < 80
    assert len(data["ambiguities"]) >= 1

    lat_amb = next((a for a in data["ambiguities"] if a["category"] == "laterality"), None)
    assert lat_amb is not None
    assert "knee" in lat_amb["reason"].lower()
    assert len(lat_amb["quick_fixes"]) >= 2
    assert any("right" in f["label"].lower() for f in lat_amb["quick_fixes"])


def test_supervise_cardio_ambiguity():
    payload = {"note": "Patient admitted with decompensated congestive heart failure."}
    response = client.post("/api/supervise", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert len(data["ambiguities"]) >= 1

    acuity_amb = next((a for a in data["ambiguities"] if a["category"] == "acuity"), None)
    assert acuity_amb is not None
    assert any("acute" in f["label"].lower() for f in acuity_amb["quick_fixes"])


def test_supervise_clean_note():
    payload = {"note": "Patient has acute right knee pain and essential hypertension."}
    response = client.post("/api/supervise", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["coder_readiness_score"] >= 90
    assert data["readiness_grade"] == "A"
    assert len(data["ambiguities"]) == 0


def test_fork_subgraph_endpoint():
    # First get an ambiguity with trajectory ids
    payload = {"note": "Patient has severe knee pain."}
    sup_res = client.post("/api/supervise", json=payload).json()
    assert len(sup_res["ambiguities"]) > 0
    tids = sup_res["ambiguities"][0]["trajectory_ids"]

    tids_param = ",".join(str(t) for t in tids)
    response = client.get(f"/api/fork-subgraph?tids={tids_param}")
    assert response.status_code == 200
    graph = response.json()
    assert "nodes" in graph
    assert "edges" in graph
    assert len(graph["nodes"]) > 0
