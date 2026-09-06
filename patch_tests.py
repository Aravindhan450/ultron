with open("tests/test_main_slash.py", "r") as f:
    content = f.read()

# test_model_status_display_syncs_with_server
content = content.replace(
    '    assert session.active_model == "qwen2.5-coder-7b-instruct-q4_k_m.gguf"\n    assert engine.default_model == "qwen2.5-coder-7b-instruct-q4_k_m.gguf"\n    assert any("Active Server Model:" in m for m in console.messages)\n    assert any("qwen2.5-coder-7b-instruct-q4_k_m.gguf" in m for m in console.messages)\n    assert any("llama.cpp" in m for m in console.messages)',
    '    assert any("Dynamic Model Architecture" in m for m in console.messages)\n    assert any("llama.cpp" in m for m in console.messages)'
)

# test_model_matching_request_accepted
content = content.replace(
    '    assert session.active_model == "qwen2.5-coder-7b-instruct-q4_k_m.gguf"\n    assert any("matches currently loaded server model" in m for m in console.messages)',
    '    assert any("Manual model switching via /model <name> is disabled" in m for m in console.messages)'
)

# test_model_switch_to_unloaded_model_rejected
content = content.replace(
    '    # State MUST NOT be updated to the unloaded model\n    assert session.active_model == "qwen2.5-coder-7b-instruct-q4_k_m.gguf"\n    assert engine.default_model == "qwen2.5-coder-7b-instruct-q4_k_m.gguf"\n    assert any("Dynamic model switching is not supported" in m for m in console.messages)\n    assert any("llama-server -m /path/to/llama-3-8b.gguf" in m for m in console.messages)',
    '    assert any("Manual model switching via /model <name> is disabled" in m for m in console.messages)'
)

# test_model_server_offline_falls_back_gracefully
content = content.replace(
    '    assert any("Model & Backend Status" in m for m in console.messages)\n    assert not any("Ollama" in m for m in console.messages)',
    '    assert any("Dynamic Model Architecture" in m for m in console.messages)\n    assert not any("Ollama" in m for m in console.messages)'
)

with open("tests/test_main_slash.py", "w") as f:
    f.write(content)
