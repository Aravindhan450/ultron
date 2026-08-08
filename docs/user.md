# Ultron – User Guide

## What is Ultron?

Ultron is a local-first, privacy-respecting personal AI assistant.  
It can hear you, talk back, use tools, control your computer (with permission), and remember things — all while keeping sensitive data under your control.

## Core Principles

- **Local by default** – Models run on your machine when possible.
- **Permission first** – High-risk actions always require explicit approval.
- **Transparent** – You can see what the AI is doing and why.
- **Extensible** – Add tools via MCP or custom code.

## Interaction Modes

| Mode          | How to use                          | Best for                     |
|---------------|-------------------------------------|------------------------------|
| Voice         | Say the wake word                   | Hands-free, natural          |
| Text          | Type in the desktop/web UI or CLI   | Precise, long instructions   |
| Desktop App   | GUI window                          | Everyday use                 |
| Web UI        | Browser                             | Remote access (careful)      |
| API           | HTTP / WebSocket                    | Automation & integrations    |

## Voice Pipeline States
Idle → Listening → Thinking → Speaking → (can be Interrupted)


- **Wake Word** – Activates listening.
- **VAD** – Detects when you start/stop speaking.
- **STT** – Turns speech into text (Whisper family).
- **Barge-in** – You can interrupt the AI while it’s speaking.
- **TTS** – Natural voice reply (Kokoro / Piper).

## Permissions & Safety

Every action is risk-classified:

| Risk Tier   | Behavior                          | Examples                          |
|-------------|-----------------------------------|-----------------------------------|
| Low         | Auto-allowed                      | Read public files, search web     |
| Medium      | Soft confirm (voice or UI)        | Open apps, send non-sensitive msgs|
| High        | Explicit confirm required         | Delete files, send emails         |
| Critical    | Explicit + extra verification     | System changes, financial actions |

You always have the final say. Every approval/denial is logged: each security
verdict (allow/confirm/deny) is appended to the JSON-lines audit trail at
`~/.ultron/security_audit.jsonl`.

The same permission model applies when several tools run at once: a parallel
batch ("read config.json and notes.txt", "check example.com and example.org")
never executes a member the boundary would deny or confirm — only auto-allowed
calls run concurrently, and the results are synthesized into one analysis. See
[docs/parallel-tools.md](docs/parallel-tools.md).

## Memory

Ultron remembers:
- Conversation history
- Facts you explicitly tell it
- Tool usage patterns (for learning)

You can inspect, edit, or wipe memory at any time.

## Getting Started (Quick)

1. Install dependencies.
2. Copy `.env.example` → `.env` and set your preferences.
3. Run `ultron` (or the desktop app).
4. Say the wake word or open the UI.
5. Start talking / typing.

## Privacy Notes

- No data leaves your machine unless you explicitly enable a cloud model or external MCP tool.
- Secrets and PII are scanned and can be redacted or blocked.
- File access is governed by a strict policy.