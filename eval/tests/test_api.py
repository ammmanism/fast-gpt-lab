import pytest
from fastapi.testclient import TestClient
from deploy.api import app, MODEL_STATE

# Initialize a dummy state for testing without loading the full model
@pytest.fixture(autouse=True)
def bypass_model_load():
    MODEL_STATE["model"] = "mocked"
    MODEL_STATE["tokenizer"] = "mocked"
    MODEL_STATE["device"] = "cpu"

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "engine": "FastGPT-Lab Kernel"}

def test_stream_completions_requires_auth_or_params():
    # Attempting a stream without valid payload should 422 Unprocessable Entity
    response = client.post("/v1/completions/stream", json={})
    assert response.status_code == 422
