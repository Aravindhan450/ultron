"""
ultron.core.repository.index
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Repository indexing engine with file classification, security/ignore filtering,
AST/symbol extraction, and incremental mtime/hash caching.
"""

from __future__ import annotations

import hashlib
import os
import time
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field

from ultron.core.coding.intelligence.parsers import (
    EXTENSION_LANGUAGES,
    language_for_path,
)
from ultron.core.repository.cache import RepositoryCache, get_repository_cache
from ultron.core.repository.parser import (
    ParsedImport,
    ParsedSymbol,
    parse_source_file,
)


class FileClassification(str, Enum):
    """Semantic category of a repository file."""

    SOURCE = "source"
    TEST = "test"
    CONFIG = "config"
    DOCUMENTATION = "documentation"
    ASSET = "asset"
    IGNORED = "ignored"
    SECRET = "secret"


# Directories that are never indexed
IGNORED_DIRS: frozenset[str] = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        ".venv",
        "venv",
        "env",
        ".env_dir",
        "node_modules",
        "__pycache__",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".tox",
        "build",
        "dist",
        ".egg-info",
        ".eggs",
        ".idea",
        ".vscode",
        ".next",
        ".nuxt",
        "target",  # Rust/Cargo
        "vendor",  # Go/PHP
    }
)

# Binary extensions strictly excluded from parsing and repository maps
BINARY_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".pyc",
        ".pyo",
        ".pyd",
        ".so",
        ".dylib",
        ".dll",
        ".exe",
        ".bin",
        ".o",
        ".a",
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".ico",
        ".svg",
        ".webp",
        ".pdf",
        ".zip",
        ".tar",
        ".gz",
        ".tgz",
        ".bz2",
        ".7z",
        ".rar",
        ".woff",
        ".woff2",
        ".ttf",
        ".eot",
        ".mp3",
        ".mp4",
        ".wav",
        ".sqlite",
        ".sqlite3",
        ".db",
    }
)

# Common configuration file basenames or suffixes
CONFIG_NAMES: frozenset[str] = frozenset(
    {
        "pyproject.toml",
        "setup.py",
        "setup.cfg",
        "requirements.txt",
        "pipfile",
        "pipfile.lock",
        "poetry.lock",
        "package.json",
        "package-lock.json",
        "pnpm-lock.yaml",
        "yarn.lock",
        "tsconfig.json",
        "cargo.toml",
        "cargo.lock",
        "go.mod",
        "go.sum",
        "gemfile",
        "dockerfile",
        "docker-compose.yml",
        "docker-compose.yaml",
        "makefile",
        "cmakelists.txt",
        ".flake8",
        ".eslintrc",
        ".prettierrc",
    }
)

CONFIG_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".toml",
        ".yaml",
        ".yml",
        ".json",
        ".ini",
        ".cfg",
        ".conf",
    }
)

DOC_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".md",
        ".rst",
        ".txt",
        ".adoc",
    }
)


def is_secret_file(path_str: str) -> bool:
    """Checks if a file path matches secret or credential patterns."""
    basename = Path(path_str).name.lower()
    if basename.startswith(".env") or basename.endswith(".env"):
        return True
    if any(basename.endswith(ext) for ext in (".pem", ".key", ".cert", ".pfx", ".p12", ".id_rsa")):
        return True
    return basename in ("credentials.json", "secrets.json", "id_rsa", "id_dsa", "id_ed25519")


def classify_file(rel_path: str) -> FileClassification:
    """Classifies a relative repository path."""
    basename = Path(rel_path).name.lower()
    suffix = Path(rel_path).suffix.lower()
    parts = Path(rel_path).parts

    # 1. Ignored or secret
    if is_secret_file(rel_path):
        return FileClassification.SECRET

    for part in parts[:-1]:
        if part in IGNORED_DIRS or part.startswith(".git"):
            return FileClassification.IGNORED

    if suffix in BINARY_EXTENSIONS:
        return FileClassification.IGNORED

    # 2. Test files
    if (
        "tests" in parts
        or "test" in parts
        or "specs" in parts
        or basename.startswith("test_")
        or basename.endswith(("_test.py", ".spec.ts", ".spec.js", ".test.ts", ".test.js"))
    ):
        return FileClassification.TEST

    # 3. Config files
    if basename in CONFIG_NAMES or suffix in CONFIG_EXTENSIONS:
        return FileClassification.CONFIG

    # 4. Documentation
    if suffix in DOC_EXTENSIONS:
        return FileClassification.DOCUMENTATION

    # 5. Source code
    if suffix in EXTENSION_LANGUAGES:
        return FileClassification.SOURCE

    return FileClassification.ASSET


class FileInfo(BaseModel):
    """Metadata and indexed contents of a single file in the repository."""

    rel_path: str
    abs_path: str
    classification: FileClassification
    language: str = ""
    size_bytes: int = 0
    mtime: float = 0.0
    content_hash: str = ""
    symbols: list[ParsedSymbol] = Field(default_factory=list)
    imports: list[ParsedImport] = Field(default_factory=list)
    parse_error: str | None = None

    @property
    def is_source(self) -> bool:
        return self.classification == FileClassification.SOURCE

    @property
    def is_test(self) -> bool:
        return self.classification == FileClassification.TEST

    @property
    def is_config(self) -> bool:
        return self.classification == FileClassification.CONFIG


class IndexSummary(BaseModel):
    """Summary metrics of a repository index pass."""

    total_files: int = 0
    source_files: int = 0
    test_files: int = 0
    config_files: int = 0
    total_symbols: int = 0
    total_imports: int = 0
    duration_ms: float = 0.0

    def to_summary_line(self) -> str:
        return (
            f"Indexed {self.total_files} files ({self.source_files} source, "
            f"{self.test_files} test, {self.config_files} config), "
            f"{self.total_symbols} symbols, {self.total_imports} imports in {self.duration_ms:.1f}ms"
        )


class RepositoryIndex:
    """
    Maintains an in-memory index of repository files, symbols, and imports
    with incremental mtime/size caching.
    """

    def __init__(self, root: str | Path, cache: RepositoryCache | None = None) -> None:
        self.root = Path(root).resolve()
        self.cache = cache or get_repository_cache()
        self._files: dict[str, FileInfo] = {}
        self._last_summary: IndexSummary | None = None

    @property
    def files(self) -> dict[str, FileInfo]:
        if not self._files:
            self.refresh()
        return self._files

    def refresh(self, force: bool = False) -> IndexSummary:
        """
        Incrementally scans the repository root, re-parsing only modified or new files.
        """
        start_time = time.perf_counter()
        root_data = self.cache.get_root_data(self.root)
        cached_files: dict[str, FileInfo] = root_data.get("files", {})

        current_files: dict[str, FileInfo] = {}

        for dirpath, dirnames, filenames in os.walk(self.root):
            # Prune ignored directories in-place
            dirnames[:] = [d for d in dirnames if d not in IGNORED_DIRS and not d.startswith(".git")]

            for filename in filenames:
                full_path = Path(dirpath) / filename
                try:
                    rel_path = str(full_path.relative_to(self.root))
                except ValueError:
                    continue

                classification = classify_file(rel_path)
                if classification in (FileClassification.IGNORED, FileClassification.SECRET):
                    continue

                try:
                    stat = full_path.stat()
                    mtime = stat.st_mtime
                    size_bytes = stat.st_size
                except (OSError, ValueError):
                    continue

                # Check if we can reuse cached file info
                if (
                    not force
                    and rel_path in cached_files
                    and cached_files[rel_path].mtime == mtime
                    and cached_files[rel_path].size_bytes == size_bytes
                ):
                    current_files[rel_path] = cached_files[rel_path]
                    continue

                # File modified or new: read and parse if appropriate
                lang = language_for_path(str(full_path))
                symbols: list[ParsedSymbol] = []
                imports: list[ParsedImport] = []
                parse_err: str | None = None
                content_hash = ""

                # Only parse text / source / test / config files under 512KB
                if size_bytes <= 512 * 1024:
                    try:
                        content = full_path.read_text(encoding="utf-8", errors="replace")
                        content_hash = hashlib.sha256(content.encode("utf-8", errors="ignore")).hexdigest()[:16]
                        if lang or classification in (FileClassification.SOURCE, FileClassification.TEST):
                            parse_res = parse_source_file(full_path, rel_path, content)
                            symbols = parse_res.symbols
                            imports = parse_res.imports
                            parse_err = parse_res.error
                    except Exception as exc:  # noqa: BLE001
                        parse_err = str(exc)

                file_info = FileInfo(
                    rel_path=rel_path,
                    abs_path=str(full_path),
                    classification=classification,
                    language=lang,
                    size_bytes=size_bytes,
                    mtime=mtime,
                    content_hash=content_hash,
                    symbols=symbols,
                    imports=imports,
                    parse_error=parse_err,
                )
                current_files[rel_path] = file_info

        # Update cache and internal store
        root_data["files"] = current_files
        # Reset cached graph and repo maps since index changed
        root_data["graph"] = None
        root_data["repo_maps"] = {}
        self._files = current_files

        duration_ms = (time.perf_counter() - start_time) * 1000.0
        src_count = sum(1 for f in current_files.values() if f.is_source)
        test_count = sum(1 for f in current_files.values() if f.is_test)
        config_count = sum(1 for f in current_files.values() if f.is_config)
        sym_count = sum(len(f.symbols) for f in current_files.values())
        imp_count = sum(len(f.imports) for f in current_files.values())

        summary = IndexSummary(
            total_files=len(current_files),
            source_files=src_count,
            test_files=test_count,
            config_files=config_count,
            total_symbols=sym_count,
            total_imports=imp_count,
            duration_ms=duration_ms,
        )
        self._last_summary = summary
        return summary

    def get_file(self, rel_path: str) -> FileInfo | None:
        """Retrieves FileInfo for a relative path."""
        return self.files.get(rel_path)

    def list_files(self, classification: FileClassification | None = None) -> list[FileInfo]:
        """Lists files optionally filtered by classification."""
        if classification is None:
            return list(self.files.values())
        return [f for f in self.files.values() if f.classification == classification]

    def get_symbols(self, name: str = "", kind: str = "") -> list[ParsedSymbol]:
        """Queries symbols matching name and/or kind."""
        results: list[ParsedSymbol] = []
        name_lower = name.lower().strip()
        kind_lower = kind.lower().strip()

        for finfo in self.files.values():
            for sym in finfo.symbols:
                if name_lower and sym.name.lower() != name_lower:
                    continue
                if kind_lower and sym.kind.lower() != kind_lower:
                    continue
                results.append(sym)
        return results

    def search_symbols(self, query: str) -> list[ParsedSymbol]:
        """Fuzzy/substring search across symbol names and signatures."""
        q = query.lower().strip()
        if not q:
            return []
        results: list[ParsedSymbol] = []
        for finfo in self.files.values():
            for sym in finfo.symbols:
                if q in sym.name.lower() or (sym.signature and q in sym.signature.lower()):
                    results.append(sym)
        return results

    def search_text(self, query: str, max_results: int = 30) -> list[tuple[str, int, str]]:
        """Lexical search across indexed source and test files."""
        q = query.lower().strip()
        if not q:
            return []
        matches: list[tuple[str, int, str]] = []
        for finfo in self.files.values():
            if finfo.classification not in (
                FileClassification.SOURCE,
                FileClassification.TEST,
                FileClassification.CONFIG,
            ):
                continue
            try:
                content = Path(finfo.abs_path).read_text(encoding="utf-8", errors="replace")
                for idx, line in enumerate(content.splitlines(), start=1):
                    if q in line.lower():
                        matches.append((finfo.rel_path, idx, line.strip()[:140]))
                        if len(matches) >= max_results:
                            return matches
            except Exception:  # noqa: BLE001, S112
                continue
        return matches
