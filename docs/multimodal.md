# Multimodal Input (Vision)

> Design doc for reasoning across modalities: analyze a graph in an uploaded
> picture and act on it (e.g. write code based on the visual data), instead of
> treating images as opaque files.

## Motivation

Ultron can read files, but a `read chart.png` returns binary garbage — images
were invisible to the model. Modern vision-capable local models (llava,
qwen2.5vl, etc., via Ollama) can interpret pictures, but the agent had no way
to *send* them: no detector, no handler, no capability check.

This upgrade closes that gap: attach an image (a diagram, chart, graph, or
handwritten sketch) and Ultron reasons across the modality — describing what
it shows, explaining the data, or writing code that implements the diagram.

## Design

### How it works

Ollama's `/api/chat` accepts an `images` array (base64-encoded) on any
message, and the engine already passes messages straight through. So the
feature is mostly agent-side:

1. **Detector** — `detect_image_intent` recognizes vision phrasing
   ("analyze this chart.png", "look at the graph in plot.png", "what's in
   screenshot.png") and requires an **image extension** (png/jpg/gif/webp/bmp/
   tiff/heic). The extension requirement is what keeps ordinary file reads
   (`read config.json`) on their existing path.
2. **Routing** — the detector runs *before* the file-read detector (Step 0.9),
   so image files always route to the vision model. Reading a binary image as
   text was never useful anyway.
3. **Handler** — `handle_image` gates the read through the security boundary
   (same path-escape/secret rules as `read_file`), checks the file exists,
   base64-encodes it, and sends it to the engine as an image part alongside a
   prompt derived from the user's request ("interpret the visual data
   precisely and act on it — e.g. write code implementing what is shown").
4. **Capability check** — `OllamaEngine.supports_images()` queries
   `/api/show` for the `vision` capability. If the active model can't see
   images, Ultron says so and gives the exact commands to switch
   (`ollama pull llava` → `/model`), instead of failing mid-request.

### Security model

Image analysis is a *read* of a local file, so it uses the existing
`read_file` classification:

- **Path escapes** are denied by the guardrails before the file is touched.
- The read is LOW risk and auto-allowed in every mode.
- Only the file's bytes leave the machine (to the local Ollama server) — the
  same trust boundary as every other tool call. The base64 image never enters
  the audit log or the reply text.

### Response style

The vision reply flows through the same response pipeline as any other answer:
`polish_response` tidies whitespace, and the shared response-style guidance in
the system prompt keeps the analysis well-mannered and structured (the
prompt tells the model to be specific and concrete about what it sees).

## CLI flow

1. User: *"analyze this chart.png and write code to reproduce the trend"*
2. `detect_image_intent` → `chart.png`
3. `handle_image` → boundary allows (LOW, in-bounds path) → base64 → engine
   call with `{"role": "user", "content": …, "images": [base64]}`
4. The model's analysis (with code) is returned, polished, and rendered.

## Edge cases

- **Non-vision model active** — friendly hint with pull/switch commands.
- **Missing file** — explicit "couldn't find the image" error, no crash.
- **File exists but unreadable / not actually an image** — the model's
  analysis or the engine error is surfaced as a message.
- **Image file named like a config** (`config.png`) — still routes to vision
  (image extension wins); the binary-content read path is not useful for
  images anyway.
- **No image phrasing** ("read notes.txt") — untouched, stays a file read.

## Future work

- Attach multiple images in one request.
- Persist the vision turn in conversation history so follow-up questions can
  reference the image.
- Drag-and-drop / paste attachment support in the CLI prompt.
- Pass the image through `retrieve`-style composition (e.g. analyze a
  screenshot of a website, then fetch the page behind it).
