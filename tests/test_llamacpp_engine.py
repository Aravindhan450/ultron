import httpx
import pytest

from ultron.core.engine.base import BaseEngine
from ultron.core.engine.llama_cpp import LlamaCppEngine


def test_llamacpp_engine_inherits_base_engine():
    engine = LlamaCppEngine()
    assert isinstance(engine, BaseEngine)
    assert engine.model == "default"
    engine.set_model("qwen2.5-coder")
    assert engine.model == "qwen2.5-coder"


@pytest.mark.anyio
async def test_generate_success(monkeypatch):
    engine = LlamaCppEngine(base_url="http://testserver")

    async def mock_post(client, url, json=None, timeout=None):
        assert url == "http://testserver/v1/chat/completions"
        assert json["model"] == "default"
        assert json["stream"] is False
        assert json["messages"] == [{"role": "user", "content": "Hello"}]
        req = httpx.Request("POST", url)
        return httpx.Response(
            status_code=200,
            json={
                "choices": [{"message": {"role": "assistant", "content": "Hi there!"}}]
            },
            request=req,
        )

    monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)
    res = await engine.generate([{"role": "user", "content": "Hello"}])
    assert res == "Hi there!"


@pytest.mark.anyio
async def test_stream_success(monkeypatch):
    engine = LlamaCppEngine(base_url="http://testserver")

    class MockStreamContext:
        async def __aenter__(self):
            class MockResponse:
                def raise_for_status(self):
                    pass

                async def aiter_lines(self):
                    yield 'data: {"choices":[{"delta":{"content":"Hello"}}]}'
                    yield ""
                    yield 'data: {"choices":[{"delta":{"content":" world"}}]}'
                    yield "data: [DONE]"

            return MockResponse()

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

    def mock_stream(client, method, url, json=None, timeout=None):
        assert method == "POST"
        assert url == "http://testserver/v1/chat/completions"
        assert json["stream"] is True
        return MockStreamContext()

    monkeypatch.setattr(httpx.AsyncClient, "stream", mock_stream)
    chunks = []
    async for chunk in engine.stream([{"role": "user", "content": "Hello"}]):
        chunks.append(chunk)

    assert chunks == ["Hello", " world"]


@pytest.mark.anyio
async def test_http_failure_raises(monkeypatch):
    engine = LlamaCppEngine(base_url="http://testserver")

    async def mock_post(client, url, json=None, timeout=None):
        req = httpx.Request("POST", url)
        resp = httpx.Response(status_code=500, request=req)
        resp.raise_for_status()

    monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)
    with pytest.raises(httpx.HTTPStatusError):
        await engine.generate([{"role": "user", "content": "Hello"}])


@pytest.mark.anyio
async def test_connection_failure_raises(monkeypatch):
    engine = LlamaCppEngine(base_url="http://testserver")

    async def mock_post(client, url, json=None, timeout=None):
        raise httpx.ConnectError("Connection refused")

    monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)
    with pytest.raises(httpx.ConnectError):
        await engine.generate([{"role": "user", "content": "Hello"}])


@pytest.mark.anyio
async def test_malformed_response_raises(monkeypatch):
    engine = LlamaCppEngine(base_url="http://testserver")

    async def mock_post(client, url, json=None, timeout=None):
        req = httpx.Request("POST", url)
        return httpx.Response(status_code=200, json={"choices": []}, request=req)

    monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)
    with pytest.raises(ValueError, match="empty choices"):
        await engine.generate([{"role": "user", "content": "Hello"}])




@pytest.mark.anyio
async def test_list_models(monkeypatch):
    engine = LlamaCppEngine(base_url="http://testserver")

    async def mock_get(client, url, timeout=None):
        assert url == "http://testserver/v1/models"
        return httpx.Response(
            status_code=200,
            json={"data": [{"id": "qwen2.5-coder"}, {"id": "llama-3"}]},
        )

    monkeypatch.setattr(httpx.AsyncClient, "get", mock_get)
    models = await engine.list_models()
    assert models == ["qwen2.5-coder", "llama-3"]


@pytest.mark.anyio
async def test_supports_images(monkeypatch):
    engine = LlamaCppEngine(base_url="http://testserver")

    async def mock_get(client, url, timeout=None):
        assert url == "http://testserver/props"
        return httpx.Response(status_code=200, json={"has_mmproj": True})

    monkeypatch.setattr(httpx.AsyncClient, "get", mock_get)
    supported = await engine.supports_images()
    assert supported is True


@pytest.mark.anyio
async def test_generate_with_image_formats_openai_payload(monkeypatch):
    engine = LlamaCppEngine(base_url="http://testserver")

    async def mock_post(client, url, json=None, timeout=None):
        assert url == "http://testserver/v1/chat/completions"
        assert len(json["messages"]) == 1
        msg = json["messages"][0]
        assert msg["role"] == "user"
        assert isinstance(msg["content"], list)
        assert msg["content"][0] == {"type": "text", "text": "Describe this image"}
        assert msg["content"][1]["type"] == "image_url"
        assert msg["content"][1]["image_url"]["url"].startswith("data:image/jpeg;base64,")
        req = httpx.Request("POST", url)
        return httpx.Response(
            status_code=200,
            json={"choices": [{"message": {"role": "assistant", "content": "A blue sky."}}]},
            request=req,
        )

    monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)
    res = await engine.generate([{
        "role": "user",
        "content": "Describe this image",
        "images": ["aW1hZ2VkYXRh"]
    }])
    assert res == "A blue sky."


@pytest.mark.anyio
async def test_supports_images_with_modalities_dict(monkeypatch):
    engine = LlamaCppEngine(base_url="http://testserver")

    async def mock_get(client, url, timeout=None):
        assert url == "http://testserver/props"
        return httpx.Response(status_code=200, json={"modalities": {"vision": False, "audio": False}})

    monkeypatch.setattr(httpx.AsyncClient, "get", mock_get)
    supported = await engine.supports_images()
    assert supported is False
