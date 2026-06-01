"""Test del servizio API (saltati se le dipendenze FastAPI non sono installate)."""
import os
import sys
import tempfile
import uuid

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# DB isolato per i test (prima di importare i moduli api)
os.environ["DATABASE_URL"] = f"sqlite:///{os.path.join(tempfile.gettempdir(), 'fnova_test_' + uuid.uuid4().hex + '.db')}"
os.environ.setdefault("API_JWT_SECRET", "test-secret")

import pytest  # noqa: E402

for _mod in ("fastapi", "sqlalchemy", "jose", "passlib", "multipart", "httpx"):
    pytest.importorskip(_mod)

from fastapi.testclient import TestClient  # noqa: E402

from api.db import Job, SessionLocal, User, init_db  # noqa: E402
from api.main import app  # noqa: E402

init_db()
client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_requires_auth():
    assert client.get("/v1/me").status_code == 401


def test_auth_and_job_flow():
    email = f"user_{uuid.uuid4().hex[:8]}@test.it"
    password = "password123"
    assert client.post("/auth/register", json={"email": email, "password": password}).status_code == 201

    token = client.post("/auth/token", data={"username": email, "password": password}).json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}
    assert client.get("/v1/me", headers=headers).json()["email"] == email

    r = client.post("/v1/books", headers=headers, json={"title": "Test", "genre": "fantasy", "target_pages": 50})
    assert r.status_code == 202
    job = client.get(f"/v1/jobs/{r.json()['id']}", headers=headers).json()
    assert job["status"] in ("pending", "running", "completed")


def test_process_job_generates_and_persists(tmp_path):
    """Verifica deterministica della logica di generazione + export + persistenza."""
    from api import services

    db = SessionLocal()
    user = User(email=f"d_{uuid.uuid4().hex[:8]}@test.it", hashed_password="x")
    db.add(user)
    db.commit()
    db.refresh(user)
    job_id = uuid.uuid4().hex
    db.add(Job(id=job_id, user_id=user.id, status="pending"))
    db.commit()
    db.close()

    services.process_job(job_id, user.id, {
        "title": "Romanzo Test", "genre": "fiction", "target_pages": 50, "chapter_structure": "misto",
    })

    db = SessionLocal()
    job = db.get(Job, job_id)
    assert job.status == "completed"
    assert job.book_id is not None
    db.close()
