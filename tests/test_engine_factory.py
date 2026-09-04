import httpx
import pytest

from ultron.core.agents.react import ReActAgent
from ultron.core.agents.simple import SimpleAgent
from ultron.core.config import settings
from ultron.core.engine import BaseEngine, LlamaCppEngine, get_engine


def test_factory_returns_llamacpp_engine_by_default():
    engine = get_engine()
    assert isinstance(engine, LlamaCppEngine)
    assert isinstance(engine, BaseEngine)
    assert engine.base_url == settings.llama_cpp_base_url.rstrip("/")
    assert engine.model == settings.model


def test_factory_with_custom_model_name():
    engine = get_engine("custom-model-123")
    assert isinstance(engine, LlamaCppEngine)
    assert engine.model == "custom-model-123"


def test_agents_receive_base_engine():
    engine = get_engine()
    simple_agent = SimpleAgent(engine=engine)
    assert isinstance(simple_agent.engine, BaseEngine)
    assert simple_agent.engine is engine

    react_agent = ReActAgent(engine=engine)
    assert isinstance(react_agent.engine, BaseEngine)
    assert react_agent.engine is engine


@pytest.mark.anyio
async def test_factory_engine_generate(monkeypatch):
    engine = get_engine()

    async def mock_post(client, url, json=None, timeout=None):
        req = httpx.Request("POST", url)
        return httpx.Response(
            status_code=200,
            json={"choices": [{"message": {"role": "assistant", "content": "Factory generation success"}}]},
            request=req,
        )

    monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)
    res = await engine.generate([{"role": "user", "content": "ping"}])
    assert res == "Factory generation success"


@pytest.mark.anyio
async def test_factory_engine_stream(monkeypatch):
    engine = get_engine()

    class MockStreamContext:
        async def __aenter__(self):
            class MockResponse:
                def raise_for_status(self):
                    pass

                async def aiter_lines(self):
                    yield 'data: {"choices":[{"delta":{"content":"Chunk1"}}]}'
                    yield 'data: {"choices":[{"delta":{"content":" Chunk2"}}]}'
                    yield "data: [DONE]"

            return MockResponse()

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

    def mock_stream(client, method, url, json=None, timeout=None):
        return MockStreamContext()

    monkeypatch.setattr(httpx.AsyncClient, "stream", mock_stream)
    chunks = []
    async for chunk in engine.stream([{"role": "user", "content": "ping"}]):
        chunks.append(chunk)

    assert "".join(chunks) == "Chunk1 Chunk2"
