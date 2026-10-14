"""Sandboxed file tools.

- Reads are limited to the workspace/ and knowledge/ folders.
- Writes are limited to workspace/ and require human approval (LangGraph `interrupt`).
- Paths are fully resolved (symlinks included) before the containment check, so `..`
  tricks and symlinks pointing outside the sandbox are rejected.
"""

from __future__ import annotations

from pathlib import Path

from langchain_core.tools import BaseTool, tool
from langgraph.types import interrupt
from pypdf import PdfReader

from general_agent.config import PROJECT_ROOT, get_settings
from general_agent.prompts import wrap_untrusted

# Name the tools/LLM use for the writable folder, independent of its real location.
WORKSPACE_ALIAS = "workspace"


class SandboxError(PermissionError):
    pass


class FileSandbox:
    def __init__(
        self, *, read_roots: list[Path], write_root: Path, project_root: Path, max_bytes: int
    ) -> None:
        self.read_roots = [r.resolve() for r in read_roots]
        self.write_root = write_root.resolve()
        self.project_root = project_root.resolve()
        self.max_bytes = max_bytes

    # ----------------------------------------------------------------- helpers
    def _inside(self, path: Path, roots: list[Path]) -> bool:
        return any(path == root or path.is_relative_to(root) for root in roots)

    def _resolve_existing(self, raw: str) -> Path:
        """Find `raw` relative to the project root or any read root; must stay inside."""
        raw = raw.strip()
        given = Path(raw).expanduser()
        candidates = (
            [given]
            if given.is_absolute()
            else [self.project_root / given, *(root / given for root in self.read_roots)]
        )
        for candidate in candidates:
            resolved = candidate.resolve()
            if resolved.exists() and self._inside(resolved, self.read_roots):
                return resolved
        raise SandboxError(
            f"'{raw}' was not found inside the allowed folders "
            f"({', '.join(r.name for r in self.read_roots)})."
        )

    # ---------------------------------------------------------------- actions
    def read(self, raw: str) -> str:
        path = self._resolve_existing(raw)
        if not path.is_file():
            raise SandboxError(f"'{raw}' is not a file.")
        if path.suffix.lower() == ".pdf":
            text = "\n".join(page.extract_text() or "" for page in PdfReader(path).pages)
        else:
            with path.open("rb") as fh:
                text = fh.read(self.max_bytes + 1).decode("utf-8", errors="replace")
        if len(text) > self.max_bytes:
            text = text[: self.max_bytes] + f"\n...[truncated at {self.max_bytes} bytes]"
        return text

    def list_dir(self, raw: str) -> str:
        if raw.strip() in ("", "/"):
            return "Allowed folders:\n" + "\n".join(f"{r.name}/" for r in self.read_roots)
        path = self._resolve_existing(raw)
        if not path.is_dir():
            raise SandboxError(f"'{raw}' is not a directory.")
        entries = sorted(path.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
        lines = [
            f"{p.name}/" if p.is_dir() else f"{p.name} ({p.stat().st_size} bytes)"
            for p in entries
            if not p.name.startswith(".")
        ]
        return "\n".join(lines) or "(empty directory)"

    def resolve_write_target(self, raw: str) -> Path:
        given = Path(raw.strip()).expanduser()
        if given.is_absolute():
            # Absolute paths are fine only if they already point inside the workspace.
            target = given.resolve()
            if not target.is_relative_to(self.write_root):
                raise SandboxError(
                    f"Writes are only allowed inside {WORKSPACE_ALIAS}/. "
                    "Use a relative path such as 'notes.txt'."
                )
            return target
        parts = given.parts
        if parts and parts[0] in (WORKSPACE_ALIAS, self.write_root.name):
            given = Path(*parts[1:])  # accept "workspace/foo.txt"
        elif parts and parts[0] in {r.name for r in self.read_roots if r != self.write_root}:
            raise SandboxError(
                f"'{parts[0]}/' is read-only; files can only be written to {WORKSPACE_ALIAS}/."
            )
        target = (self.write_root / given).resolve()
        if target == self.write_root or not target.is_relative_to(self.write_root):
            raise SandboxError(f"Writes are only allowed to files inside {WORKSPACE_ALIAS}/.")
        return target

    def display_path(self, target: Path) -> str:
        """Stable, user-facing path ('workspace/...'), wherever the workspace really lives."""
        return f"{WORKSPACE_ALIAS}/{target.relative_to(self.write_root)}"

    def write(self, raw: str, content: str) -> Path:
        target = self.resolve_write_target(raw)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return target


def default_sandbox() -> FileSandbox:
    s = get_settings()
    return FileSandbox(
        read_roots=[s.workspace_dir, s.knowledge_dir],
        write_root=s.workspace_dir,
        project_root=PROJECT_ROOT,
        max_bytes=s.max_file_read_bytes,
    )


def make_file_tools(sandbox: FileSandbox, *, require_approval: bool = True) -> list[BaseTool]:
    @tool
    def read_file(path: str) -> str:
        """Read a text, markdown or PDF file from the workspace/ or knowledge/ folders.
        Accepts paths like 'notes.txt', 'knowledge/guide.md' or 'workspace/out.txt'."""
        try:
            return wrap_untrusted(sandbox.read(path))
        except (SandboxError, OSError) as exc:
            return f"Error: {exc}"

    @tool
    def list_directory(path: str = "") -> str:
        """List files in a folder inside workspace/ or knowledge/. Empty path lists the
        allowed top-level folders."""
        try:
            return sandbox.list_dir(path)
        except (SandboxError, OSError) as exc:
            return f"Error: {exc}"

    @tool
    def write_file(path: str, content: str) -> str:
        """Write (create or overwrite) a UTF-8 text file inside workspace/. The user is
        asked to approve the write first."""
        try:
            target = sandbox.resolve_write_target(path)
        except SandboxError as exc:
            return f"Error: {exc}"
        if require_approval:
            # Human-in-the-loop: pauses the whole graph until the user answers.
            # Nothing with side effects happens before this line, so re-running the
            # tool on resume is safe.
            decision = interrupt(
                {
                    "type": "approval",
                    "action": "write_file",
                    "path": sandbox.display_path(target),
                    "preview": content[:500],
                    "question": f"Allow writing {len(content)} characters to "
                    f"{sandbox.display_path(target)}?",
                }
            )
            if not _is_approved(decision):
                return "Write cancelled: the user did not approve it."
        try:
            written = sandbox.write(path, content)
        except (SandboxError, OSError) as exc:
            return f"Error: {exc}"
        # Echo what was written so later steps report the real content, not a paraphrase.
        preview = content if len(content) <= 1000 else content[:1000] + "\n...[truncated]"
        return (
            f"Wrote {len(content)} characters to {sandbox.display_path(written)}. "
            f"Exact content written:\n{preview}"
        )

    return [read_file, list_directory, write_file]


def _is_approved(decision: object) -> bool:
    if isinstance(decision, bool):
        return decision
    return str(decision).strip().lower() in {"y", "yes", "approve", "approved", "ok", "true"}
