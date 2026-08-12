"""
Fix #3 stage-1 tests: safe file editing + modification tracking.

Covers create / replace / targeted edit / append / delete / rename, the
ModificationTracker, and path-safety (ALLOWED_BASE_DIR confinement). All
tests use temporary directories — the real Ultron repository is never
modified.
"""

import subprocess

import pytest

from ultron.core.coding.edits import (
    EditAction,
    FileModification,
    ModificationTracker,
    append_to_file,
    create_file,
    delete_file,
    rename_file,
    replace_file,
    replace_in_file,
)
from ultron.core.tools import paths as tools_paths


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    monkeypatch.setattr(tools_paths, "ALLOWED_BASE_DIR", tmp_path)
    monkeypatch.chdir(tmp_path)
    return tmp_path


# ---------------------------------------------------------------------------
# create / replace
# ---------------------------------------------------------------------------


def test_create_file(sandbox):
    result = create_file("notes.txt", "hello")
    assert result.startswith("Created")
    assert (sandbox / "notes.txt").read_text(encoding="utf-8") == "hello"


def test_create_file_refuses_overwrite(sandbox):
    (sandbox / "notes.txt").write_text("existing", encoding="utf-8")
    result = create_file("notes.txt", "new")
    assert result.startswith("Error: file already exists")
    assert (sandbox / "notes.txt").read_text(encoding="utf-8") == "existing"


def test_replace_file(sandbox):
    (sandbox / "notes.txt").write_text("old content", encoding="utf-8")
    result = replace_file("notes.txt", "new content")
    assert result.startswith("Replaced")
    assert (sandbox / "notes.txt").read_text(encoding="utf-8") == "new content"


# ---------------------------------------------------------------------------
# targeted edit
# ---------------------------------------------------------------------------


def test_replace_in_file_targeted(sandbox):
    (sandbox / "app.py").write_text("x = 1\nprint(x)\n", encoding="utf-8")
    result = replace_in_file("app.py", "print(x)", "print(x * 2)")
    assert "Replaced 1 occurrence" in result
    assert (sandbox / "app.py").read_text(encoding="utf-8") == "x = 1\nprint(x * 2)\n"


def test_replace_in_file_preserves_unrelated_content(sandbox):
    (sandbox / "app.py").write_text("def a(): pass\ndef b(): pass\n", encoding="utf-8")
    replace_in_file("app.py", "def a(): pass", "def a(): return 1")
    assert "def b(): pass" in (sandbox / "app.py").read_text(encoding="utf-8")


def test_replace_in_file_missing_text_is_error(sandbox):
    (sandbox / "app.py").write_text("x = 1", encoding="utf-8")
    result = replace_in_file("app.py", "not there", "y = 2")
    assert result.startswith("Error: text not found")
    # Nothing changed.
    assert (sandbox / "app.py").read_text(encoding="utf-8") == "x = 1"


def test_replace_in_file_blank_old_is_error(sandbox):
    (sandbox / "app.py").write_text("x = 1", encoding="utf-8")
    result = replace_in_file("app.py", "", "y = 2")
    assert result.startswith("Error:")


# ---------------------------------------------------------------------------
# append / delete / rename
# ---------------------------------------------------------------------------


def test_append_to_file(sandbox):
    (sandbox / "log.txt").write_text("a", encoding="utf-8")
    result = append_to_file("log.txt", "\nb")
    assert result.startswith("Appended")
    assert (sandbox / "log.txt").read_text(encoding="utf-8") == "a\nb"


def test_append_creates_missing_file(sandbox):
    result = append_to_file("new.txt", "hello")
    assert result.startswith("Appended")
    assert (sandbox / "new.txt").read_text(encoding="utf-8") == "hello"


def test_delete_file(sandbox):
    (sandbox / "tmp.txt").write_text("x", encoding="utf-8")
    result = delete_file("tmp.txt")
    assert result.startswith("Deleted")
    assert not (sandbox / "tmp.txt").exists()


def test_delete_file_refuses_directory(sandbox):
    (sandbox / "adir").mkdir()
    result = delete_file("adir")
    assert result.startswith("Error:")
    assert (sandbox / "adir").is_dir()


def test_rename_file(sandbox):
    (sandbox / "a.txt").write_text("x", encoding="utf-8")
    result = rename_file("a.txt", "b.txt")
    assert result.startswith("Renamed")
    assert not (sandbox / "a.txt").exists()
    assert (sandbox / "b.txt").read_text(encoding="utf-8") == "x"


# ---------------------------------------------------------------------------
# path safety (ALLOWED_BASE_DIR confinement)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "op",
    [
        lambda p: create_file(str(p / "out.txt"), "x"),
        lambda p: replace_file(str(p / "out.txt"), "x"),
        lambda p: replace_in_file(str(p / "out.txt"), "a", "b"),
        lambda p: append_to_file(str(p / "out.txt"), "x"),
        lambda p: delete_file(str(p / "out.txt")),
        lambda p: rename_file(str(p / "a.txt"), str(p / "b.txt")),
    ],
)
def test_all_edits_reject_unsafe_paths(sandbox, op):
    result = op(sandbox.parent / "outside")
    assert result.startswith("Error: access denied")


# ---------------------------------------------------------------------------
# ModificationTracker
# ---------------------------------------------------------------------------


def test_tracker_records_modifications(sandbox):
    tracker = ModificationTracker()
    tracker.record("notes.txt", EditAction.CREATE, resulting_state="hello")
    tracker.record("notes.txt", EditAction.TARGETED_EDIT, original_state="hello", resulting_state="hello2")
    tracker.record("gone.txt", EditAction.DELETE, original_state="bye", success=True)

    assert len(tracker.modifications) == 3
    mod = tracker.modifications[1]
    assert mod.action is EditAction.TARGETED_EDIT
    assert mod.original_state == "hello"
    assert mod.resulting_state == "hello2"
    assert mod.success is True
    assert mod.describe().startswith("[targeted_edit]")


def test_tracker_records_failure(sandbox):
    tracker = ModificationTracker()
    tracker.record("x.txt", EditAction.REPLACE, success=False, error="boom")
    assert tracker.modifications[-1].success is False
    assert "boom" in tracker.modifications[-1].describe()


def test_tracker_recent_limit(sandbox):
    tracker = ModificationTracker()
    for i in range(15):
        tracker.record(f"f{i}.txt", EditAction.CREATE)
    assert len(tracker.recent(10)) == 10
    assert tracker.recent(10)[0].path == "f5.txt"


def test_tracker_git_status_integration(sandbox):
    try:
        subprocess.run(["git", "init", "-q"], cwd=sandbox, check=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        pytest.skip("git not available")
    (sandbox / "a.txt").write_text("hello", encoding="utf-8")
    tracker = ModificationTracker()
    tracker.record("a.txt", EditAction.CREATE, resulting_state="hello")
    status = tracker.git_status(str(sandbox))
    assert status != ""


def test_file_modification_model_serializes():
    mod = FileModification(
        path="a.txt",
        action=EditAction.CREATE,
        original_state=None,
        resulting_state="x",
        success=True,
    )
    restored = FileModification.model_validate_json(mod.model_dump_json())
    assert restored.path == "a.txt"
    assert restored.action is EditAction.CREATE


# ---------------------------------------------------------------------------
# Tool-result recording (wires the tracker into agent execution)
# ---------------------------------------------------------------------------


def test_record_tool_result_records_success():
    from ultron.core.coding.edits import record_tool_result

    tracker = ModificationTracker()
    mod = record_tool_result(tracker, "replace_in_file", "app.py", "Replaced 1 occurrence(s) in 'app.py'.", step=2)
    assert mod is not None
    assert mod.action is EditAction.TARGETED_EDIT
    assert mod.path == "app.py"
    assert mod.success is True
    assert mod.step == 2
    assert tracker.modifications == [mod]


def test_record_tool_result_records_failure_with_error():
    from ultron.core.coding.edits import record_tool_result

    tracker = ModificationTracker()
    mod = record_tool_result(
        tracker,
        "delete_file",
        "gone.txt",
        "Error: file not found at gone.txt",
    )
    assert mod is not None
    assert mod.action is EditAction.DELETE
    assert mod.success is False
    assert "file not found" in (mod.error or "")


def test_record_tool_result_ignores_non_edits():
    from ultron.core.coding.edits import record_tool_result

    tracker = ModificationTracker()
    mod = record_tool_result(tracker, "run_command", "pytest", "Exit code: 0")
    assert mod is None
    assert tracker.modifications == []
