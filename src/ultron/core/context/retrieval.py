"""
ultron.core.context.retrieval
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Evidence-based retrieval engine for repository files, symbols, search results,
git status, and diffs.

Guarantees:
- Never invents files or symbols (returns explicit NOT_FOUND)
- Respects security boundary path safety (is_path_safe)
- Extracts scoped regions/symbols rather than unbounded repository dumps
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from ultron.core.logging import get_logger

logger = get_logger("ultron.context.retrieval")

from ultron.core.coding.intelligence.facade import CodeIntelligence
from ultron.core.coding.workspace import (
    CodingWorkspace,
    _resolve_safe_path,
    discover_workspace,
)
from ultron.core.context.models import (
    ContextItem,
    ContextPriority,
    ContextRetrievalResult,
    ContextRetrievalStatus,
    ContextSourceType,
    estimate_tokens,
)


class RepositoryRetriever:
    """
    Read-only retrieval interface over the workspace filesystem and code index.
    """

    def __init__(self, workspace: CodingWorkspace | None = None) -> None:
        self.workspace = workspace or discover_workspace()
        self.root = Path(self.workspace.project_root)
        self._intelligence: CodeIntelligence | None = None

    @property
    def intelligence(self) -> CodeIntelligence:
        if self._intelligence is None:
            self._intelligence = CodeIntelligence(root=self.root)
        return self._intelligence

    def retrieve_file(
        self,
        file_path: str,
        start_line: int | None = None,
        end_line: int | None = None,
        max_lines: int = 250,
    ) -> ContextRetrievalResult:
        """
        Retrieves a file or bounded file region from the workspace.
        Explicitly returns NOT_FOUND if file does not exist.
        """
        target_path_str = str(file_path).strip()
        resolved = _resolve_safe_path(target_path_str)
        if resolved is None:
            # Try relative to workspace root
            candidate = self.root / target_path_str
            resolved = _resolve_safe_path(str(candidate))

        if resolved is None:
            return ContextRetrievalResult(
                target=target_path_str,
                status=ContextRetrievalStatus.ACCESS_DENIED,
                source_type=ContextSourceType.FILE_CONTENT,
                error_message=f"Path '{target_path_str}' is outside allowed directory",
                searched_locations=[target_path_str],
            )

        if not resolved.exists() or not resolved.is_file():
            return ContextRetrievalResult(
                target=target_path_str,
                status=ContextRetrievalStatus.NOT_FOUND,
                source_type=ContextSourceType.FILE_CONTENT,
                error_message=f"File not found: '{target_path_str}'",
                searched_locations=[str(resolved)],
            )

        try:
            lines = resolved.read_text(encoding="utf-8", errors="replace").splitlines()
        except Exception as exc:  # noqa: BLE001
            return ContextRetrievalResult(
                target=target_path_str,
                status=ContextRetrievalStatus.ERROR,
                source_type=ContextSourceType.FILE_CONTENT,
                error_message=f"Error reading file: {exc}",
                searched_locations=[str(resolved)],
            )

        total_lines = len(lines)
        if start_line is not None or end_line is not None:
            s = max(1, start_line or 1)
            e = min(total_lines, end_line or total_lines)
            selected_lines = lines[s - 1 : e]
            title = f"{resolved.name} (lines {s}-{e}/{total_lines})"
        else:
            if total_lines > max_lines:
                selected_lines = lines[:max_lines]
                title = f"{resolved.name} (first {max_lines}/{total_lines} lines)"
            else:
                selected_lines = lines
                title = f"{resolved.name} ({total_lines} lines)"

        content = "\n".join(selected_lines)
        item = ContextItem(
            source_type=ContextSourceType.FILE_CONTENT,
            priority=ContextPriority.DIRECT_FILE,
            title=title,
            content=content,
            target=str(resolved),
            estimated_tokens=estimate_tokens(content),
            metadata={"total_lines": total_lines, "path": str(resolved)},
        )

        return ContextRetrievalResult(
            target=target_path_str,
            status=ContextRetrievalStatus.FOUND,
            source_type=ContextSourceType.FILE_CONTENT,
            items=[item],
            searched_locations=[str(resolved)],
        )

    def retrieve_symbol(self, symbol_name: str) -> ContextRetrievalResult:
        """
        Retrieves AST symbol definitions and references via CodeIntelligence.
        Returns NOT_FOUND if the symbol does not exist in the index.
        """
        name = symbol_name.strip()
        if not name:
            return ContextRetrievalResult(
                target=symbol_name,
                status=ContextRetrievalStatus.NOT_FOUND,
                source_type=ContextSourceType.SYMBOL_DEFINITION,
                error_message="Empty symbol name provided",
            )

        defs = self.intelligence.find_definition(name)
        refs = self.intelligence.find_references(name)

        if not defs and not refs:
            # Also try case-insensitive symbol lookup
            symbols = self.intelligence.find_symbol(name, case_insensitive=True)
            if not symbols:
                return ContextRetrievalResult(
                    target=name,
                    status=ContextRetrievalStatus.NOT_FOUND,
                    source_type=ContextSourceType.SYMBOL_DEFINITION,
                    error_message=f"Symbol '{name}' not found in repository index",
                )
            defs = symbols

        items: list[ContextItem] = []
        for d in defs[:5]:
            desc = f"{d.kind.value} {d.name} in {d.rel_path}:{d.line}"
            body = d.signature or d.docstring or desc
            items.append(
                ContextItem(
                    source_type=ContextSourceType.SYMBOL_DEFINITION,
                    priority=ContextPriority.SYMBOL,
                    title=f"Definition: {d.name} ({d.rel_path})",
                    content=body,
                    target=d.name,
                    estimated_tokens=estimate_tokens(body),
                    metadata={"rel_path": d.rel_path, "line": d.line, "kind": d.kind.value},
                )
            )

        for r in refs[:5]:
            ref_line = f"Reference in {r.rel_path}:{r.line}"
            items.append(
                ContextItem(
                    source_type=ContextSourceType.SYMBOL_REFERENCE,
                    priority=ContextPriority.SYMBOL,
                    title=f"Reference: {r.name} ({r.rel_path})",
                    content=ref_line,
                    target=r.name,
                    estimated_tokens=estimate_tokens(ref_line),
                    metadata={"rel_path": r.rel_path, "line": r.line},
                )
            )

        return ContextRetrievalResult(
            target=name,
            status=ContextRetrievalStatus.FOUND,
            source_type=ContextSourceType.SYMBOL_DEFINITION,
            items=items,
        )

    def retrieve_search(self, query: str, max_results: int = 10) -> ContextRetrievalResult:
        """
        Executes lexical code search and wraps matches into structured ContextItems.
        """
        q = query.strip()
        if not q:
            return ContextRetrievalResult(
                target=query,
                status=ContextRetrievalStatus.NOT_FOUND,
                source_type=ContextSourceType.SEARCH_RESULT,
                error_message="Empty search query",
            )

        raw_search = self.intelligence.search(q, max_results=max_results)
        if not raw_search or raw_search.startswith(("No matches", "Error:")):
            return ContextRetrievalResult(
                target=q,
                status=ContextRetrievalStatus.NOT_FOUND,
                source_type=ContextSourceType.SEARCH_RESULT,
                error_message=f"No matches found for query '{q}'",
            )

        item = ContextItem(
            source_type=ContextSourceType.SEARCH_RESULT,
            priority=ContextPriority.SEARCH,
            title=f"Search matches for '{q}'",
            content=raw_search,
            target=q,
            estimated_tokens=estimate_tokens(raw_search),
        )

        return ContextRetrievalResult(
            target=q,
            status=ContextRetrievalStatus.FOUND,
            source_type=ContextSourceType.SEARCH_RESULT,
            items=[item],
        )

    def retrieve_git_context(self) -> ContextRetrievalResult:
        """
        Retrieves current git branch, dirty status, and short status diff.
        """
        if not self.workspace.is_git_repo:
            return ContextRetrievalResult(
                target="git",
                status=ContextRetrievalStatus.NOT_FOUND,
                source_type=ContextSourceType.GIT_STATE,
                error_message="Not a git repository",
            )

        status_text = self.workspace.git_status_short or "clean working tree"
        body = f"Branch: {self.workspace.git_branch or 'HEAD'}\nStatus: {status_text}"

        # Try to retrieve compact unstaged diff if available
        try:
            diff_proc = subprocess.run(
                ["git", "-C", str(self.root), "diff", "--stat"],
                capture_output=True,
                text=True,
                timeout=3,
                check=False,
            )
            if diff_proc.returncode == 0 and diff_proc.stdout.strip():
                body += f"\nDiff Stat:\n{diff_proc.stdout.strip()[:400]}"
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"Failed to read git diff: {exc}")

        item = ContextItem(
            source_type=ContextSourceType.GIT_STATE,
            priority=ContextPriority.CHANGES_AND_DIFF,
            title="Git Working Tree Status",
            content=body,
            target="git_status",
            estimated_tokens=estimate_tokens(body),
        )

        return ContextRetrievalResult(
            target="git",
            status=ContextRetrievalStatus.FOUND,
            source_type=ContextSourceType.GIT_STATE,
            items=[item],
        )
