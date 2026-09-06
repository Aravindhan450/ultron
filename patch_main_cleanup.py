with open("src/ultron/main.py", "r") as f:
    lines = f.readlines()

start_idx = -1
end_idx = -1
for i, line in enumerate(lines):
    if line.strip() == "reflow.start()":
        start_idx = i + 1
    if line.startswith("@app.command()"):
        if end_idx == -1 and i > start_idx:
            end_idx = i

if start_idx != -1 and end_idx != -1:
    body = lines[start_idx:end_idx]
    # Remove existing lifecycle_manager.shutdown() if any
    body = [l for l in body if "lifecycle_manager.shutdown()" not in l and "Phase 3.5: Cleanup" not in l]
    indented_body = ["    " + l if l.strip() else l for l in body]
    
    new_lines = lines[:start_idx] + ["    try:\n"] + indented_body + ["    finally:\n        lifecycle_manager.shutdown()\n\n"] + lines[end_idx:]
    with open("src/ultron/main.py", "w") as f:
        f.writelines(new_lines)
