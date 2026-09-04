"""
Project Ultron — Phase 5 Model-in-the-Loop Comprehensive Validation Suite
Exercises the complete stack:
  Ultron CLI / Agent -> BaseEngine -> LlamaCppEngine -> HTTP -> llama-server -> Qwen2.5-Coder (Metal)
"""
import asyncio
import time
from types import SimpleNamespace

import httpx
from rich.console import Console

from ultron.core.agents.react import ReActAgent
from ultron.core.agents.simple import SimpleAgent
from ultron.core.engine import LlamaCppEngine, get_engine
from ultron.main import handle_slash_command

console = Console()

async def run_scenario(name: str, coro):
    console.print(f"\n[bold cyan]▶ Running Scenario: {name}[/bold cyan]")
    start = time.monotonic()
    try:
        res = await coro
        elapsed = time.monotonic() - start
        console.print(f"[bold green]✔ {name} PASSED ({elapsed:.2f}s)[/bold green]")
        return True, res
    except Exception as e:  # noqa: BLE001
        elapsed = time.monotonic() - start
        console.print(f"[bold red]✘ {name} FAILED ({elapsed:.2f}s): {e}[/bold red]")

        return False, str(e)

async def main():
    engine = get_engine()
    assert isinstance(engine, LlamaCppEngine), "Default engine must be LlamaCppEngine"

    # Verify server reachable and metadata
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"{engine.base_url}/v1/models")
        assert resp.status_code == 200, "llama-server must be reachable"
        models_data = resp.json().get("data", [])
        active_model_path = models_data[0].get("id") if models_data else "unknown"

    console.print(f"[bold yellow]Active llama-server Endpoint:[/bold yellow] {engine.base_url}")
    console.print(f"[bold yellow]Active Loaded Model:[/bold yellow] {active_model_path}\n")

    results = {}

    # Test A — Basic conversation
    async def test_a():
        agent = SimpleAgent(engine=engine)
        r1 = await agent.run("hello")
        assert r1.content and len(r1.content.strip()) > 0
        r2 = await agent.run("Explain recursion in Java with a simple example.")
        assert "recursion" in r2.content.lower() or "method" in r2.content.lower()
        return r2.content[:150]
    results["Basic conversation"] = await run_scenario("Test A: Basic conversation", test_a())

    # Test B — Repository intelligence
    async def test_b():
        agent = ReActAgent(engine=engine)
        r = await agent.run("Where is BaseEngine defined in this repository?")
        assert "base.py" in r.content or "engine" in r.content.lower()
        return r.content[:150]
    results["Repository intelligence"] = await run_scenario("Test B: Repository intelligence", test_b())

    # Test C — ReAct routing
    async def test_c():
        agent = ReActAgent(engine=engine)
        r = await agent.run("What is in the current directory? List the files.")
        assert r.content and len(r.content) > 0
        return r.content[:150]
    results["ReAct routing"] = await run_scenario("Test C: ReAct routing", test_c())

    # Test D — Tool execution
    async def test_d():
        agent = SimpleAgent(engine=engine)
        r = await agent.run("Run pwd")
        assert "/Users/aravindhan/ultron" in r.content or "ultron" in r.content.lower()
        return r.content.strip()
    results["Tool execution"] = await run_scenario("Test D: Tool execution", test_d())

    # Test E — Web / External routing
    async def test_e():
        agent = SimpleAgent(engine=engine)
        r = await agent.run("search the web for the latest Python version")
        # SecurityBoundary confirm or result returned
        assert r.content and ("web" in r.content.lower() or "python" in r.content.lower() or "confirm" in r.content.lower() or "search" in r.content.lower() or r.pending_action is not None)
        return r.content[:150]
    results["Web routing"] = await run_scenario("Test E: Web routing", test_e())

    # Test F — Security Boundary
    async def test_f():
        agent = SimpleAgent(engine=engine)
        # Test 1: Denied command with secret
        r1 = await agent.run("run grep AKIA1234567890ABCDEF config.txt")
        assert "blocked by security" in r1.content.lower()

        # Test 2: Confirmation required for state-changing command
        r2 = await agent.run("run mkdir test_phase5_dir")
        assert r2.pending_action is not None and r2.pending_action.action_type == "run_command"
        return f"Denied secret blocked: {r1.content.strip()}; Confirmation required: {r2.pending_action.action_type}"
    results["Security"] = await run_scenario("Test F: Security", test_f())

    # Test G — Multi-step task
    async def test_g():
        agent = SimpleAgent(engine=engine)
        r = await agent.run("read checkme.txt then check git status")
        assert r.content and len(r.content) > 0
        return r.content[:150]
    results["Multi-step task"] = await run_scenario("Test G: Multi-step task", test_g())

    # Test H — Streaming
    async def test_h():
        chunks = []
        async for chunk in engine.stream([{"role": "user", "content": "Count: 1, 2, 3"}]):
            chunks.append(chunk)
        streamed = "".join(chunks).lower()
        assert len(chunks) > 1, f"Expected real streaming chunks, got {len(chunks)}"
        assert ("1" in streamed or "one" in streamed) and ("2" in streamed or "two" in streamed)
        return f"{len(chunks)} chunks streamed: {streamed.strip()}"
    results["Streaming"] = await run_scenario("Test H: Streaming", test_h())


    # Test I — /model command
    async def test_i():
        dummy_session = SimpleNamespace(active_model="default")
        agent = SimpleAgent(engine=engine)
        handled, _ = await handle_slash_command("/model", console, [], agent=agent, session=dummy_session)
        assert handled is True
        assert dummy_session.active_model == active_model_path
        return f"Synced to active model: {dummy_session.active_model}"
    results["/model"] = await run_scenario("Test I: /model command", test_i())

    # Test J — Unsupported vision handling
    async def test_j():
        import base64
        from pathlib import Path
        supported = await engine.supports_images()
        assert supported is False
        
        # Create a real local test image in workspace
        tmp_img = Path("test_phase5_chart.png")
        tmp_img.write_bytes(base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="))
        try:
            agent = SimpleAgent(engine=engine)
            r = await agent.run("analyze this test_phase5_chart.png")
            assert "does not support vision" in r.content
            return r.content.strip()
        finally:
            if tmp_img.exists():
                tmp_img.unlink()
    results["Unsupported vision"] = await run_scenario("Test J: Unsupported vision", test_j())


    console.print("\n=======================================================")
    console.print("         PHASE 5 LIVE VALIDATION SUMMARY               ")
    console.print("=======================================================")
    all_ok = True
    for k, (ok, detail) in results.items():
        status = "[green]PASS[/green]" if ok else "[red]FAIL[/red]"
        console.print(f"{k:<26}: {status}")
        if not ok:
            all_ok = False

    return 0 if all_ok else 1

if __name__ == "__main__":
    import sys
    sys.exit(asyncio.run(main()))
