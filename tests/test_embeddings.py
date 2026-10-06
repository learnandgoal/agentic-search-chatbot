"""Azure OpenAI embedder, tested with a mocked client (no credentials or network needed)."""

from __future__ import annotations

import langchain_openai
import numpy as np
import pytest

from hr_chatbot.embeddings import (
    AzureOpenAIEmbedder, HashingEmbedder, OpenAIEmbedder, default_embedder_kind, get_embedder,
)
from hr_chatbot.ingest import Ingestor
from hr_chatbot.sample_data import generate
from hr_chatbot.store import HRStore, IndexMismatchError, Scope

ENV_VARS = [
    "HR_EMBEDDER", "AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_EMBEDDING_DEPLOYMENT",
    "AZURE_OPENAI_EMBEDDING_DIMENSIONS", "AZURE_OPENAI_EMBEDDING_API_VERSION", "OPENAI_API_VERSION",
    "OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_EMBEDDING_MODEL", "OPENAI_EMBEDDING_DIMENSIONS",
]


class FakeAzureEmbeddings:
    """Stands in for langchain_openai.AzureOpenAIEmbeddings. Returns unnormalised hash vectors."""

    instances: list["FakeAzureEmbeddings"] = []
    ignore_dimensions = False
    broken_shape = False

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.calls: list[list[str]] = []
        dim = 96 if self.ignore_dimensions else kwargs.get("dimensions", 96)
        self.inner = HashingEmbedder(dim=dim)
        FakeAzureEmbeddings.instances.append(self)

    def embed_documents(self, texts):
        self.calls.append(list(texts))
        if self.broken_shape:
            return [[0.1, 0.2]] * (len(texts) + 1)  # one vector too many
        return (self.inner.embed(texts) * 3.0).tolist()


@pytest.fixture(autouse=True)
def azure_env(monkeypatch):
    for name in ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.openai.azure.com")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-key")
    FakeAzureEmbeddings.instances = []
    FakeAzureEmbeddings.ignore_dimensions = False
    FakeAzureEmbeddings.broken_shape = False
    monkeypatch.setattr(langchain_openai, "AzureOpenAIEmbeddings", FakeAzureEmbeddings)
    monkeypatch.setattr(langchain_openai, "OpenAIEmbeddings", FakeAzureEmbeddings)  # same fake for the OpenAI client


def test_azure_is_the_default_when_the_endpoint_is_set():
    embedder = get_embedder()
    assert isinstance(embedder, AzureOpenAIEmbedder)
    kwargs = FakeAzureEmbeddings.instances[0].kwargs
    assert kwargs["azure_endpoint"] == "https://example.openai.azure.com"
    assert kwargs["azure_deployment"] == "text-embedding-3-large"  # default deployment name
    assert kwargs["api_key"] == "test-key"
    assert kwargs["check_embedding_ctx_length"] is False  # avoids a runtime tokenizer download
    assert "dimensions" not in kwargs
    assert embedder.name == "azure:text-embedding-3-large"


def test_explicit_choice_beats_the_azure_default(monkeypatch):
    monkeypatch.setenv("HR_EMBEDDER", "hashing")
    assert isinstance(get_embedder(), HashingEmbedder)
    assert not FakeAzureEmbeddings.instances


def test_deployment_dimensions_and_api_version_come_from_the_environment(monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "my-embeddings")
    monkeypatch.setenv("AZURE_OPENAI_EMBEDDING_DIMENSIONS", "256")
    monkeypatch.setenv("AZURE_OPENAI_EMBEDDING_API_VERSION", "2025-01-01-preview")
    embedder = get_embedder("azure")
    kwargs = FakeAzureEmbeddings.instances[0].kwargs
    assert kwargs["azure_deployment"] == "my-embeddings" and kwargs["dimensions"] == 256
    assert kwargs["api_version"] == "2025-01-01-preview"
    assert embedder.dim == 256 and embedder.name == "azure:my-embeddings"


def test_vector_size_is_read_from_the_first_response():
    embedder = get_embedder("azure")
    assert embedder.dim == 96  # not hard-coded: whatever the deployment returns
    assert FakeAzureEmbeddings.instances[0].calls == [["dimension probe"]]


def test_vectors_are_float32_l2_normalised_and_batched():
    embedder = get_embedder("azure")
    vectors = embedder.embed(["python developer", "kafka streaming", "sql reporting"])
    assert vectors.dtype == np.float32 and vectors.shape == (3, 96)
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)
    assert FakeAzureEmbeddings.instances[0].calls[-1] == ["python developer", "kafka streaming", "sql reporting"]
    assert embedder.embed([]).shape == (0, 96)


def test_missing_credentials_fail_with_a_clear_message(monkeypatch):
    monkeypatch.delenv("AZURE_OPENAI_API_KEY")
    with pytest.raises(RuntimeError, match="AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY"):
        get_embedder("azure")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "k")
    monkeypatch.delenv("AZURE_OPENAI_ENDPOINT")
    with pytest.raises(RuntimeError, match="AZURE_OPENAI_ENDPOINT"):
        get_embedder("azure")


def test_dimension_mismatch_is_an_error_not_a_silent_surprise(monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_EMBEDDING_DIMENSIONS", "256")
    FakeAzureEmbeddings.ignore_dimensions = True  # a deployment that ignores the request and returns 96
    with pytest.raises(RuntimeError, match="96-dimensional vectors but 256 were requested"):
        get_embedder("azure")


def test_unexpected_response_shape_is_an_error():
    FakeAzureEmbeddings.broken_shape = True
    with pytest.raises(RuntimeError, match="Unexpected embeddings response shape"):
        get_embedder("azure")


def test_unknown_kind_lists_the_choices():
    with pytest.raises(ValueError, match="azure"):
        get_embedder("nonsense")


def test_openai_embedder_uses_a_gateway_url_and_model_from_the_environment(monkeypatch):
    monkeypatch.setenv("HR_EMBEDDER", "openai")  # explicit choice beats the Azure endpoint that is also set
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
    monkeypatch.setenv("OPENAI_EMBEDDING_MODEL", "openai/text-embedding-3-large")
    embedder = get_embedder()
    assert isinstance(embedder, OpenAIEmbedder)
    kwargs = FakeAzureEmbeddings.instances[0].kwargs
    assert kwargs["model"] == "openai/text-embedding-3-large" and kwargs["base_url"] == "https://openrouter.ai/api/v1"
    assert kwargs["api_key"] == "sk-test" and kwargs["check_embedding_ctx_length"] is False
    assert embedder.name == "openai:openai/text-embedding-3-large" and embedder.dim == 96
    vectors = embedder.embed(["python", "kafka"])
    assert vectors.shape == (2, 96) and np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)


def test_openai_embedder_defaults_and_dimensions(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("OPENAI_EMBEDDING_DIMENSIONS", "128")
    embedder = get_embedder("openai")
    kwargs = FakeAzureEmbeddings.instances[0].kwargs
    assert kwargs["model"] == "text-embedding-3-large" and "base_url" not in kwargs and kwargs["dimensions"] == 128
    assert embedder.dim == 128


def test_openai_embedder_is_never_chosen_automatically(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.delenv("AZURE_OPENAI_ENDPOINT")
    assert default_embedder_kind() == "sentence-transformers"
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.openai.azure.com")
    assert default_embedder_kind() == "azure"


def test_openai_embedder_needs_a_key():
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        get_embedder("openai")


def test_ingest_and_hybrid_search_work_end_to_end_with_azure_vectors(settings):
    generate(settings.data_dir)
    store = HRStore(settings.db_path, get_embedder("azure"), settings)
    report = Ingestor(store, settings.data_dir, settings).ingest_all()
    assert not report.warnings and store.stats()["embedder"] == "azure:text-embedding-3-large"

    hits = store.search(Scope("candidate-001", "job-001"), "kafka streaming experience", 5)
    assert hits and any("Kafka" in h["text"] for h in hits)

    # An index built with Azure vectors must not be reopened with a different embedder
    with pytest.raises(IndexMismatchError):
        HRStore(settings.db_path, HashingEmbedder(), settings)
